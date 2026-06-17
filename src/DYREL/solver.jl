## VISCO-ELASTIC STOKES SOLVER
"""
    solve_DYREL!(
        stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
        kwargs...,
    )

Solve the Stokes system with the self-tuned dynamic relaxation (DYREL) method.

# Arguments (in the following order)
- `stokes`: `JustRelax.StokesArrays` containing the simulation fields.
- `ρg`: buoyancy forces arrays.
- `dyrel`: DYREL-specific parameters and fields.
- `flow_bcs`: `AbstractFlowBoundaryConditions` defining velocity boundary conditions.
- `phase_ratios`: `JustPIC.PhaseRatios` for material phase tracking.
- `rheology`: Material properties and rheological laws.
- `args`: Tuple of additional arguments needed to update viscosity, stress, and buoyancy forces.
- `grid`: `Geometry` object carrying grid spacing and staggered-grid coordinates. A legacy
  2D spacing tuple or named tuple is also accepted and converted to a uniform `Geometry`.
- `dt`: Time step.
- `igg`: `IGG` object for global grid information (MPI).

# Keyword Arguments
- `viscosity_cutoff`: Limits for viscosity `(min, max)`. Default: `(-Inf, Inf)`.
- `viscosity_relaxation`: Relaxation factor for viscosity updates. Default: `1.0e-2`.
- `λ_relaxation_DR`: Relaxation factor for dynamic relaxation. Default: `1`.
- `λ_relaxation_PH`: Relaxation factor for Powell-Hestenes iterations. Default: `1`.
- `iterMax`: Maximum number of iterations for each dynamic-relaxation solve. Default: `50.0e3`.
- `total_iterMax`: Maximum number of total dynamic-relaxation iterations. Default: `50.0e3`.
- `nout`: Output frequency for residuals. Default: `100`.
- `rel_drop`: Relative residual drop tolerance. Default: `1.0e-2`.
- `verbose_PH`: Print Powell-Hestenes iteration info. Default: `true`.
- `verbose_DR`: Print Dynamic Relaxation iteration info. Default: `true`.
- `linear_viscosity`: Whether to use linear viscosity. Default: `false`.
- `use_gershgorin_ad`: Use the AD-based Gershgorin entries from `GershgorinAD.jl`
  instead of the analytic linear-viscoelastic estimate from `Gershgorin.jl`. Default: `false`.
"""
function solve_DYREL!(stokes::JustRelax.StokesArrays, args...; kwargs)
    out = solve_DYREL!(backend(stokes), stokes, args...; kwargs)
    return out
end

# entry point for extensions
solve_DYREL!(::CPUBackendTrait, stokes, args...; kwargs) = _solve_DYREL!(stokes, args...; kwargs...)

function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        ρg,
        dyrel,
        flow_bcs::AbstractFlowBoundaryConditions,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        grid::Geometry{N},
        dt,
        igg::IGG;
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
        use_gershgorin_ad = false,
        kwargs...,
    ) where {N}

    dim = Val(N)
    v_dofs = velocity_dofs(dim)
    p_dof = pressure_dof(dim)
    di = grid.di
    _di = grid._di
    di_center = di.center
    ni = size(stokes.P)

    residuals = @residuals(stokes.R)
    fields = dyrel_fields(dyrel, dim)

    # errors
    err = 1.0
    iter = 0

    # solver loop
    @copy stokes.P0 stokes.P
    residuals0 = map(similar, residuals)

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
    P_num = similar(stokes.P)

    # recompute all the DYREL variables
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    if use_gershgorin_ad
        compute_∇V_strain_rate!(stokes, _di, ni, dim)
        compute_stress_DRYEL!(stokes, dyrel, rheology, phase_ratios, λ_relaxation_PH, dt, Val(true))
        DYREL_AD!(dyrel, stokes, rheology, phase_ratios, grid, dt; CFL = dyrel.CFL)
    else
        DYREL!(dyrel, stokes, rheology, phase_ratios, grid.di, dt)
    end

    # Powell-Hestenes iterations
    for itPH in 1:1000
        # update buoyancy forces
        update_ρg!(ρg, phase_ratios, rheology, args)

        # compute divergence and deviatoric strain rate in one pass
        compute_∇V_strain_rate!(stokes, _di, ni, dim)

        # compute deviatoric stress
        do_partials = Val(false)
        compute_stress_DRYEL!(stokes, dyrel, rheology, phase_ratios, λ_relaxation_PH, dt, do_partials)
        # update_halo!(stokes.λv)
        # update_halo!(stokes.τ.xx_v)
        # update_halo!(stokes.τ.yy_v)
        # update_halo!(stokes.τ.xy)

        if !linear_viscosity
            update_viscosity_τII!(
                stokes,
                phase_ratios,
                args,
                rheology,
                viscosity_cutoff;
                relaxation = viscosity_relaxation,
                do_partials = do_partials,
                ∂η_∂ε = (dyrel.∂ηc_∂ε, dyrel.∂ηv_∂ε),
            )
        end

        # compute velocity residuals
        @parallel (@idx ni) compute_PH_residual_V!(
            residuals...,
            stokes.P,
            stokes.ΔPψ,
            @stress(stokes)...,
            ρg...,
            _di.center,
            _di.vertex,
        )

        # compute pressure residual
        compute_residual_P!(
            stokes.R.RP,
            stokes.P,
            stokes.P0,
            stokes.∇V,
            stokes.Q, # volumetric source/sink term
            dyrel.ηb,
            rheology,
            phase_ratios,
            dt,
            args,
        )

        # Residual check
        errV = ntuple(d -> norm_mpi(residuals[d]) / √(v_dofs[d]), dim)
        errPt = norm_mpi(stokes.R.RP) / √(p_dof)
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
            # rel_drop = max(rel_drop * 0.1, ϵ)
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

            # Pseudo-old dudes
            foreach(copyto!, residuals0, residuals)

            # compute divergence and deviatoric strain rate in one pass
            compute_∇V_strain_rate!(stokes, _di, ni, dim)

            do_partials = use_gershgorin_ad && iszero(iter % nout)

            # Deviatoric stress
            compute_stress_DRYEL!(stokes, dyrel, rheology, phase_ratios, λ_relaxation_DR, dt, do_partials)
            # update_halo!(stokes.λv)
            # update_halo!(stokes.τ.xx_v)
            # update_halo!(stokes.τ.yy_v)
            # update_halo!(stokes.τ.xy)

            compute_residual_P!(
                stokes.R.RP,
                stokes.P,
                stokes.P0,
                stokes.∇V,
                stokes.Q, # volumetric source/sink term
                dyrel.ηb,
                rheology,
                phase_ratios,
                dt,
                args,
            )

            if !linear_viscosity
                update_viscosity_τII!(
                    stokes,
                    phase_ratios,
                    args,
                    rheology,
                    viscosity_cutoff;
                    relaxation = viscosity_relaxation,
                    do_partials = do_partials,
                    ∂η_∂ε = (dyrel.∂ηc_∂ε, dyrel.∂ηv_∂ε),
                )
            end

            # Residuals
            @. P_num = dyrel.γ_eff * stokes.R.RP
            @parallel (@idx ni) compute_DR_residual_V!(
                residuals...,
                stokes.P,
                P_num,
                stokes.ΔPψ,
                @stress(stokes)...,
                ρg...,
                fields.D...,
                _di.center,
                _di.vertex,
            )

            # Damping and pseudo-transient velocity updates
            @parallel (@idx ni) update_V_damping_DR_V!(
                @velocity(stokes),
                fields.dVdτ,
                residuals,
                fields.αV,
                fields.βV,
                fields.dτV,
            )
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)

            # Residual check
            if iszero(iter % nout)

                errV = ntuple(d -> norm_mpi(fields.D[d] .* residuals[d]) / √(v_dofs[d]), dim)

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

                # @printf("it = %d, iter = %d, ϵ_vel = %1.3e, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e] \n", itPT, iter, ϵ_vel, err, errVx, errVy)
                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
                end
                λminV = compute_λminV!(fields, residuals, residuals0, ni, dim)
                @parallel (@idx ni) update_cV!(fields.cV, 2 * √(λminV) * dyrel.c_fact)

                # Optimal pseudo-time steps
                if use_gershgorin_ad
                    Gershgorin_Stokes2D_SchurComplementAD(dyrel, _di.center, _di.vertex, _di.velocity[1], _di.velocity[2])
                else
                    Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, grid.di, dt)
                end

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

function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        ρg,
        dyrel,
        flow_bcs::AbstractFlowBoundaryConditions,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        di::Union{NTuple{2, <:Real}, NamedTuple},
        dt,
        igg::IGG;
        kwargs...,
    )
    grid = JustRelax.legacy_uniform_grid(size(stokes.P), di)
    return _solve_DYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg; kwargs...)
end

# Dimension-agnostic helpers for DYREL

@inline function dyrel_fields(dyrel::JustRelax.DYREL, ::Val{2})
    return (
        D = (dyrel.Dx, dyrel.Dy),
        λmaxV = (dyrel.λmaxVx, dyrel.λmaxVy),
        dVdτ = (dyrel.dVxdτ, dyrel.dVydτ),
        dτV = (dyrel.dτVx, dyrel.dτVy),
        dV = (dyrel.dVx, dyrel.dVy),
        βV = (dyrel.βVx, dyrel.βVy),
        cV = (dyrel.cVx, dyrel.cVy),
        αV = (dyrel.αVx, dyrel.αVy),
    )
end

@inline function dyrel_fields(dyrel::JustRelax.DYREL, ::Val{3})
    return (
        D = (dyrel.Dx, dyrel.Dy, dyrel.Dz),
        λmaxV = (dyrel.λmaxVx, dyrel.λmaxVy, dyrel.λmaxVz),
        dVdτ = (dyrel.dVxdτ, dyrel.dVydτ, dyrel.dVzdτ),
        dτV = (dyrel.dτVx, dyrel.dτVy, dyrel.dτVz),
        dV = (dyrel.dVx, dyrel.dVy, dyrel.dVz),
        βV = (dyrel.βVx, dyrel.βVy, dyrel.βVz),
        cV = (dyrel.cVx, dyrel.cVy, dyrel.cVz),
        αV = (dyrel.αVx, dyrel.αVy, dyrel.αVz),
    )
end

@inline dyrel_fields(::JustRelax.DYREL, ::Val{N}) where {N} = error("Unsupported dimension $N")

@inline global_grid_size(::Val{2}) = nx_g(), ny_g()
@inline global_grid_size(::Val{3}) = nx_g(), ny_g(), nz_g()
@inline global_grid_size(::Val{N}) where {N} = error("Unsupported dimension $N")

@inline pressure_dof(N) = prod(global_grid_size(N))

function velocity_dofs(::Val{N}) where {N}
    global_size = global_grid_size(Val(N))
    return ntuple(Val(N)) do d
        @inline
        prod(i -> i == d ? global_size[i] - 2 : global_size[i] - 1, 1:N)
    end
end

function compute_λminV!(fields, residuals, residuals0, ni, ::Val{N}) where {N}
    @parallel (@idx ni) compute_dV!(fields.dV, fields.dVdτ, fields.βV, fields.dτV)

    numerator = sum(ntuple(d -> sum_mpi(fields.dV[d] .* (residuals[d] .- residuals0[d])), Val(N)))
    denominator = sum(ntuple(d -> sum_mpi(fields.dV[d] .^ 2), Val(N)))
    return abs(numerator) / denominator
end

function copy_stress_vertices!(stokes::JustRelax.StokesArrays, ::Val{2})
    stokes.τ_o.xx_v .= stokes.τ.xx_v
    return stokes.τ_o.yy_v .= stokes.τ.yy_v
end

function copy_stress_vertices!(stokes::JustRelax.StokesArrays, ::Val{3})
    stokes.τ_o.xx_v .= stokes.τ.xx_v
    stokes.τ_o.yy_v .= stokes.τ.yy_v
    return stokes.τ_o.zz_v .= stokes.τ.zz_v
end
