## VARIATIONAL VISCO-ELASTIC STOKES SOLVER (DYREL)
#
# Mirror of `_solve_DYREL!` (src/DYREL/solver.jl) but taking `ϕ::JustRelax.RockRatio`
# as a positional argument after `phase_ratios`, exactly like `_solve_VS!` vs the APT
# `_solve!`. Julia dispatch routes through the same public `solve_DYREL!` entry point.
# 2D only — DYREL is 2D-only (Gershgorin_Stokes2D_SchurComplement!).

function _solve_VariationalDYREL!(
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
        pressure_relaxation = 1,
        iterMax = nothing,
        iterMax_PH = 1.0e3,
        iterMax_DR = isnothing(iterMax) ? 50.0e3 : iterMax,
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
    _di = grid._di
    lx = grid.max_li
    ni = size(stokes.P)

    residuals = @residuals(stokes.R)
    fields = dyrel_fields(dyrel, dim)

    # Masks: only count residuals over the valid (rock) part of the domain. `similar` keeps the
    # element type a `Bool`, not a `Bit`: the kernels below write single entries from concurrent
    # threads, and `BitArray` `setindex!` is a non-atomic read-modify-write of a whole 64-bit
    # chunk, so neighbouring columns would race.
    maskV = (
        similar(ϕ.Vx, Bool, (size(ϕ.Vx, 1) - 2, size(ϕ.Vx, 2))),
        similar(ϕ.Vy, Bool, (size(ϕ.Vy, 1), size(ϕ.Vy, 2) - 2)),
    )
    maskP = similar(ϕ.center, Bool)
    @parallel (@idx ni) update_valid_c_mask!(maskP, ϕ)
    @parallel (@idx ni) update_valid_v_masks!(maskV..., ϕ)
    # velocity interiors, which is what maskV is shaped like; views alias the parent, so these
    # stay current for the whole solve
    Vi = ntuple(d -> @views(@velocity(stokes)[d][2:(end - 1), 2:(end - 1)]), dim)
    # Momentum-residual norms run over the interior only, so that a boundary-condition row cannot
    # set the residual scale; the continuity residual is not trimmed. Trimming mask, residual and
    # preconditioner diagonal identically keeps them index-aligned.
    maskRi = ntuple(d -> @views(maskV[d][2:(end - 1), 2:(end - 1)]), dim)
    Ri = ntuple(d -> @views(residuals[d][2:(end - 1), 2:(end - 1)]), dim)
    R0i = ntuple(d -> @views(fields.R0[d][2:(end - 1), 2:(end - 1)]), dim)
    dVi = ntuple(d -> @views(fields.dV[d][2:(end - 1), 2:(end - 1)]), dim)
    Di = ntuple(d -> @views(fields.D[d][2:(end - 1), 2:(end - 1)]), dim)
    # Divisors that turn the masked L2 norms into RMS values: the number of entries actually
    # summed. The global grid DOF count would instead scale the residual by the rock fraction,
    # which — unlike the boundary trim — does not tend to 1 as the resolution grows, so ϵ would
    # mean something different for every sticky-air thickness. ϕ is fixed for a solve, so are these.
    nV = ntuple(d -> max(sum_mpi(maskRi[d]), 1), dim)
    nP = max(sum_mpi(maskP), 1)

    # errors
    err = 1.0
    iter = 0

    # The marker chain can change the reduced space between calls. Project primary unknowns out
    # of eliminated rows and discard all dynamic-relaxation history: dV/dτ, residual history and
    # modal damping coefficients belong to the old operator and are not valid after a topology
    # change. Resetting every call is cheap, deterministic, and equivalent when the mask is fixed.
    @parallel (@idx ni) project_reduced_state!(
        stokes.P, stokes.P0, stokes.ΔPψ, stokes.λ, @velocity(stokes)..., ϕ
    )
    foreach(A -> fill!(A, zero(eltype(A))), (fields.dVdτ..., fields.dV..., fields.R0..., fields.cV...))
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

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
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ
    converged = false

    errV0 = ntuple(_ -> 1.0, dim)
    errPt0 = 1.0
    err_evo_tot = Float64[]
    err_evo_V = Float64[]
    err_evo_P = Float64[]
    err_evo_it = Float64[]
    itg = 0
    # small pressure correction θc = γ_eff·RP + ΔPψ, assembled each iteration and read (alongside
    # the separately-differenced P) by the momentum kernel. Reuses the dyrel.P_num scratch.
    θc = dyrel.P_num

    # recompute all the DYREL variables
    compute_viscosity!(stokes, phase_ratios, ϕ, args, rheology, viscosity_cutoff; air_phase = air_phase)
    compute_ρg!(ρg[end], phase_ratios, rheology, args; air_phase)
    DYREL!(dyrel, stokes, rheology, phase_ratios, ϕ, grid.di, dt, iszero(free_surface) ? nothing : ρg[end])

    # Powell-Hestenes iterations
    for itPH in 1:Int(iterMax_PH)
        # update buoyancy forces
        update_ρg!(ρg, phase_ratios, rheology, args; air_phase)

        # compute divergence, deviatoric strain rate and pressure residual in one pass (masked)
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, args, true)

        # deviatoric stress, then a separate τII-viscosity refresh. The stress kernel derives the
        # vertex viscosity as harm_clamped(η) — the same convention as the APT variational stress
        # kernel; a stored ηv would disagree at the free-surface interface and under-move it.
        compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, λ_relaxation_PH, dt)
        if !linear_viscosity
            update_viscosity_τII!(stokes, phase_ratios, ϕ, args, rheology, viscosity_cutoff; relaxation = viscosity_relaxation, air_phase = air_phase)
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

        # Residual check, normalized as in the non-variational solver, but with the spans taken
        # over the masked (rock) cells only: void cells carry no meaningful V or P and would
        # otherwise set the scale. maskV[d] is shaped like the residuals, i.e. the interior of
        # the velocity arrays.
        Pspan = nonzero_span(masked_value_span(maskP, stokes.P))
        Vspan = nonzero_span(maximum(map(masked_value_scale, maskV, Vi)))
        errV = ntuple(d -> masked_norm_mpi(maskRi[d], Ri[d]) / Pspan * lx / √(nV[d]), dim)
        RP_rms = masked_norm_mpi(maskP, stokes.R.RP) / √(nP)
        errPt = RP_rms * lx / Vspan
        err = maximum((errV..., errPt))
        # Convergence additionally accepts a continuity residual that is negligible in absolute
        # terms: a field at rest has no velocity scale, so `Vspan` collapses to the residual-level
        # noise and `errPt` stops carrying information. `RP` is a divergence, so `RP·dt` is the
        # volumetric strain the step would accumulate — dimensionless and solution-independent.
        # Only the convergence test uses it; `err` continues to drive the tolerance schedule below,
        # which is tuned against the relative form.
        err_converged = max(maximum(errV), min(errPt, RP_rms * dt))

        if itPH ≤ 2
            errV0 = map(x -> x + eps(), errV)
            errPt0 = errPt + eps()
        end

        if verbose_PH && igg.me == 0
            errV_msg = join(
                ntuple(d -> @sprintf("R%d=%1.3e %1.3e", d, errV[d], errV[d] / errV0[d]), dim),
                ", ",
            )
            @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[%s, Rp=%1.3e %1.3e] \n", itPH, iter, iter / ni[1], err, errV_msg, errPt, errPt / errPt0)
        end
        igg.me == 0 && isnan(err) && error("NaN detected in outer loop")
        igg.me == 0 && err > 1.0e10 && itPH > 1 && error("Kaboom! Error > 1e10 in outer loop")
        if err_converged < ϵ && itPH > 1
            converged = true
            break
        end

        # Set tolerance of velocity solve proportional to residual
        if err > err_min * 1.05
            rel_drop = max(rel_drop * 0.1, 1.0e-3)
        end
        if err_min > err
            err_min = err
        end

        # Target a drop of `errV`, the residual the loop below measures — `err` mixes in `errPt`,
        # which is normalized by a different span. Both guards are load-bearing: an identically
        # zero momentum residual (boundary-driven flow) would otherwise set `ϵ_vel = 0` and burn
        # `iterMax_DR`, and `Inf` makes the loop always reach its first residual check.
        ϵ_vel = max(maximum(errV) * rel_drop, ϵ)
        err_vel = Inf
        itPT = 0
        # Initialize dτ for the FSSA-stabilized operator (mirrors solver.jl). The in-loop
        # dτ refresh only fires every `nout` iterations; without this the first window of
        # velocity updates would drive the free-surface-stabilization residual term against
        # a dτ tuned for the plain viscous operator and diverge.
        if !iszero(free_surface)
            Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, ϕ, rheology, grid.di, dt, ρg[end])
            update_dτV_α_β!(dyrel)
        end
        while (err_vel > ϵ_vel && itPT ≤ iterMax_DR)
            itPT += 1
            itg += 1
            iter += 1

            # Pseudo-old dudes (only needed by compute_λminV! on residual-check iterations)
            iszero(iter % nout) && foreach(copyto!, residuals0, residuals)

            # compute divergence, deviatoric strain rate and pressure residual in one pass (masked)
            compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, args, true)

            # deviatoric stress (vertex viscosity via harm_clamped(η)) + separate τII-viscosity
            # refresh, then assemble the small pressure correction θc = γ_eff·RP + ΔPψ
            compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, λ_relaxation_DR, dt)
            if !linear_viscosity
                update_viscosity_τII!(stokes, phase_ratios, ϕ, args, rheology, viscosity_cutoff; relaxation = viscosity_relaxation, air_phase = air_phase)
            end
            # exchange vertex-stress halos (+ vertex viscosity, refreshed above) before the momentum
            # kernel reads them, matching the non-variational solver
            if linear_viscosity
                update_halo!(stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)
            else
                update_halo!(stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy, stokes.viscosity.ηv)
            end
            @. θc = dyrel.γ_eff * stokes.R.RP + stokes.ΔPψ

            # Velocity residual + damped pseudo-transient velocity update (fused, masked). The face
            # fraction enters only through `variational_face_mass` inside `D`; the damping
            # recurrence itself carries no ϕ factor.
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

                # D·(stored residual) is the raw momentum residual; normalize it exactly
                # like the outer check so ϵ_vel = err_vel·rel_drop compares like with like.
                # P is fixed within a pass, so the outer Pspan is still current here.
                errV = ntuple(d -> masked_norm_mpi(maskRi[d], Di[d], Ri[d]) / Pspan * lx / √(nV[d]), dim)
                err_vel = maximum(errV)
                isnan(err_vel) && igg.me == 0 && error("NaN detected in inner loop")

                push!(err_evo_tot, err_vel)
                push!(err_evo_V, err_vel)
                push!(err_evo_P, errPt)
                push!(err_evo_it, iter)

                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err_vel)
                end
                # Estimate the smallest eigenvalue on exactly the same reduced, boundary-trimmed
                # velocity space used by the residual norm. Eliminated cut-cell rows otherwise
                # contaminate the Rayleigh quotient even though they are not part of the solve.
                @parallel (@idx ni) compute_dV!(fields.dV, fields.dVdτ, fields.βV, fields.dτV)
                λminV = masked_λminV(dVi, Ri, R0i, maskRi)
                @parallel (@idx ni) update_cV!(fields.cV, 2 * √(λminV) * dyrel.c_fact)

                # Optimal pseudo-time steps - can be replaced by AD
                Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, ϕ, rheology, grid.di, dt, iszero(free_surface) ? nothing : ρg[end])

                # Select dτ
                update_dτV_α_β!(dyrel)
            end
        end
        if itPT > iterMax_DR && igg.me == 0
            @warn "DYREL velocity solve exhausted iterMax_DR before reaching ϵ_vel" itPH iter itPT iterMax_DR err_vel ϵ_vel maxlog = 10
        end

        # update pressure — refresh RP from the final velocity first (do_strain_rate = false leaves
        # the strain-rate arrays untouched), otherwise the pressure correction lags one velocity update
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, args, false)
        @. stokes.P += pressure_relaxation * dyrel.γ_eff * stokes.R.RP
        # The uniform volumetric mode is fitted to what the local update above left behind, so RP
        # has to be refreshed in between; reusing the pre-update residual corrects the mean twice.
        # Both the refresh and the relaxation are skipped where the mode carries no correction.
        compliance = volumetric_compliance_total(dyrel.ηb, maskP)
        if !iszero(compliance)
            compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, args, false)
            relax_volumetric_mode!(stokes.P, stokes.R.RP, dyrel.ηb, maskP, pressure_relaxation, compliance)
        end

        iter > total_iterMax && break
    end
    if !converged && igg.me == 0
        @warn "DYREL returned without meeting ϵ — the velocity/pressure fields are not converged" err ϵ iter total_iterMax
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

    return (; err_evo_it, err_evo_V, err_evo_P, err_evo_tot, err, iter, converged)

end

# legacy uniform-grid wrapper (di as a spacing tuple / named tuple)
function _solve_VariationalDYREL!(
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
    return _solve_VariationalDYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg; kwargs...)
end
