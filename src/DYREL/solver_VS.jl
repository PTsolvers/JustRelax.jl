## VARIATIONAL VISCO-ELASTIC STOKES SOLVER (DYREL)
#
# Mirror of `_solve_DYREL!` (src/DYREL/solver.jl) but taking `ϕ::JustRelax.RockRatio`
# as a positional argument after `phase_ratios`, exactly like `_solve_VS!` vs the APT
# `_solve!`. Julia dispatch routes through the same public `solve_DYREL!` entry point.
# 2D only — DYREL is 2D-only (Gershgorin_Stokes2D_SchurComplement!).

function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        ρg,
        dyrel,
        flow_bcs::AbstractFlowBoundaryConditions,
        phase_ratios::JustPIC.PhaseRatios,
        ϕ::JustRelax.RockRatio,
        rheology,
        args,
        grid::Geometry{N},
        dt,
        igg::IGG;
        air_phase::Integer = 0,
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        λ_relaxation_DR = 1,
        λ_relaxation_PH = 1,
        iterMax = 50.0e3,
        total_iterMax = 50.0e3,
        nout = 100,
        rel_drop = 1.0e-2,
        b_width = (4, 4, 0),
        verbose_PH = true,
        verbose_DR = true,
        linear_viscosity = false,
        free_surface = false,
        kwargs...,
    ) where {N}

    dim = Val(N)
    v_dofs = velocity_dofs(dim)
    p_dof = pressure_dof(dim)
    di = grid.di
    _di = grid._di
    ni = size(stokes.P)

    residuals = @residuals(stokes.R)
    fields = dyrel_fields(dyrel, dim)

    # masks: only count residuals over the valid (rock) part of the domain
    maskV = (ϕ.Vx[2:(end - 1), :] .> 0, ϕ.Vy[:, 2:(end - 1)] .> 0)
    maskP = ϕ.center .> 0

    # errors
    err = 1.0
    iter = 0

    # solver loop
    @copy stokes.P0 stokes.P
    residuals0 = fields.R0

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # reset plastic multiplier at the beginning of the time step
    stokes.λ .= 0.0
    stokes.λv .= 0.0

    # Iteration loop
    err_min = Inf
    err = 1.0
    errV0 = ntuple(_ -> 1.0, dim)
    errPt0 = 1.0
    errV00 = ntuple(_ -> 1.0, dim)
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ
    err_evo_tot = Float64[]
    err_evo_V = Float64[]
    err_evo_P = Float64[]
    err_evo_it = Float64[]
    itg = 0
    # small pressure correction θc = γ_eff·RP + ΔPψ, assembled once per iteration and read (alongside
    # the separately-differenced P) by the fused momentum kernel. Reuses the P_num scratch.
    θc = similar(stokes.P)

    # recompute all the DYREL variables
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase = air_phase)
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    DYREL!(dyrel, stokes, rheology, phase_ratios, ϕ, grid.di, dt)

    # Powell-Hestenes iterations
    for itPH in 1:1000
        # update buoyancy forces
        update_ρg!(ρg, phase_ratios, rheology, args)

        # compute divergence, deviatoric strain rate and pressure residual in one pass (masked)
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, args)

        # compute deviatoric stress
        compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, λ_relaxation_PH, dt)

        if !linear_viscosity
            update_viscosity_τII!(
                stokes,
                phase_ratios,
                args,
                rheology,
                viscosity_cutoff;
                relaxation = viscosity_relaxation,
                air_phase = air_phase,
            )
        end

        # compute velocity residuals (pressure residual stokes.R.RP already computed above;
        # free-surface stabilization via dt * free_surface)
        @parallel (@idx ni) compute_PH_residual_V!(
            residuals...,
            @velocity(stokes)...,
            stokes.P,
            stokes.ΔPψ,
            @stress(stokes)...,
            ρg...,
            ϕ,
            _di.center,
            _di.vertex,
            dt * free_surface,
        )

        # Residual check
        errV = ntuple(d -> norm_mpi(@views residuals[d][maskV[d]]) / √(v_dofs[d]), dim)
        errPt = norm_mpi(@views stokes.R.RP[maskP]) / √(p_dof)
        if isone(itPH)
            errV0 = map(x -> x + eps(), errV)
            errPt0 = errPt + eps()
        end
        if itPH == 2
            errPt0 = errPt + eps()
        end
        errV_rel = ntuple(d -> min(errV[d] / errV0[d], errV[d]), dim)
        err = maximum((errV_rel..., min(errPt / errPt0, errPt)))

        if verbose_PH && igg.me == 0
            errV_msg = join(
                ntuple(d -> @sprintf("R%d=%1.3e %1.3e", d, errV[d], errV[d] / errV0[d]), dim),
                ", ",
            )
            @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[%s, Rp=%1.3e %1.3e] \n", itPH, iter, iter / ni[1], err, errV_msg, errPt, errPt / errPt0)
        end
        igg.me == 0 && isnan(err) && error("NaN detected in outer loop")
        igg.me == 0 && err > 1.0e10 && error("Kaboom! Error > 1e10 in outer loop")
        err < ϵ && break

        # Set tolerance of velocity solve proportional to residual
        if err > err_min * 1.05
            rel_drop = max(rel_drop * 0.1, 1.0e-3)
        end
        if err_min > err
            err_min = err
        end

        ϵ_vel = err * rel_drop
        itPT = 0
        while (err > ϵ_vel && itPT ≤ iterMax)
            itPT += 1
            itg += 1
            iter += 1

            # Pseudo-old dudes (only needed by compute_λminV! on residual-check iterations)
            iszero(iter % nout) && foreach(copyto!, residuals0, residuals)

            # compute divergence, deviatoric strain rate and pressure residual in one pass (masked)
            compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, args)

            # Deviatoric stress
            compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, λ_relaxation_DR, dt)

            if !linear_viscosity
                update_viscosity_τII!(
                    stokes,
                    phase_ratios,
                    args,
                    rheology,
                    viscosity_cutoff;
                    relaxation = viscosity_relaxation,
                    air_phase = air_phase,
                )
            end

            # assemble the small pressure correction θc = γ_eff·RP + ΔPψ (masked diffs are linear, so
            # differencing θc equals differencing P_num + ΔPψ separately). ΔPψ and RP are fresh above.
            @. θc = dyrel.γ_eff * stokes.R.RP + stokes.ΔPψ

            # Velocity residual + ϕ-damped pseudo-transient velocity update (fused, masked)
            @parallel (@idx ni) compute_DR_residual_update_V!(
                residuals...,
                @velocity(stokes)...,
                fields.dVdτ...,
                stokes.P,
                θc,
                @stress(stokes)...,
                ρg...,
                fields.D...,
                fields.αV...,
                fields.βV...,
                fields.dτV...,
                ϕ,
                _di.center,
                _di.vertex,
                dt * free_surface,
            )
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)

            # Residual check
            if iszero(iter % nout)

                errV = ntuple(d -> norm_mpi(@views (fields.D[d] .* residuals[d])[maskV[d]]) / √(v_dofs[d]), dim)

                if iter == nout
                    errV00 = errV
                end

                errV_ratio = ntuple(d -> errV[d] / errV00[d], dim)
                err = maximum(errV_ratio)
                isnan(err) && igg.me == 0 && error("NaN detected in inner loop")

                push!(err_evo_tot, err)
                push!(err_evo_V, maximum(errV_ratio))
                push!(err_evo_P, errPt / errPt0)
                push!(err_evo_it, iter)

                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
                end
                λminV = compute_λminV!(fields, residuals, residuals0, ni, dim)
                @parallel (@idx ni) update_cV!(fields.cV, 2 * √(λminV) * dyrel.c_fact)

                # Optimal pseudo-time steps - can be replaced by AD
                Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, ϕ, rheology, grid.di, dt)

                # Select dτ
                update_dτV_α_β!(dyrel)
            end
        end

        # update pressure
        @. stokes.P += dyrel.γ_eff .* stokes.R.RP

        iter > total_iterMax && break
    end

    # absorb plastic pressure correction into P (mirrors APT: stokes.P .= θ = P + ΔPψ)
    @. stokes.P += stokes.ΔPψ

    # refresh the ∇V diagnostic from the converged velocity field (masked); it is no longer stored
    # inside the fused DYREL/PH loop (see compute_∇V_strain_rate_RP!)
    @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di.vertex)

    # compute vorticity
    compute_vorticity!(stokes, _di, ni, dim)

    # Interpolate shear components to cell center arrays
    shear2center!(stokes.ε)
    shear2center!(stokes.ε_pl)
    shear2center!(stokes.Δε)

    # accumulate plastic strain tensor
    accumulate_tensor!(stokes.EII_pl, stokes.ε_pl, dt)
    accumulate_vol!(stokes.EVol_pl, stokes.ε_vol_pl, dt)

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))
    copy_stress_vertices!(stokes, dim)

    return (; err_evo_it, err_evo_V, err_evo_P, err_evo_tot)

end

# legacy uniform-grid wrapper (di as a spacing tuple / named tuple)
function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        ρg,
        dyrel,
        flow_bcs::AbstractFlowBoundaryConditions,
        phase_ratios::JustPIC.PhaseRatios,
        ϕ::JustRelax.RockRatio,
        rheology,
        args,
        di::Union{NTuple{2, <:Real}, NamedTuple},
        dt,
        igg::IGG;
        kwargs...,
    )
    grid = JustRelax.legacy_uniform_grid(size(stokes.P), di)
    return _solve_DYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg; kwargs...)
end
