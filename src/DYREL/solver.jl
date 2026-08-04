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
        free_surface = false,
        kwargs...,
    ) where {N}

    dim = Val(N)
    v_dofs = velocity_dofs(dim)
    p_dof = pressure_dof(dim)
    di = grid.di
    _di = grid._di
    di_center = di.center
    lx = grid.max_li
    ni = size(stokes.P)

    residuals = @residuals(stokes.R)
    fields = dyrel_fields(dyrel, dim)

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
    converged = false
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ

    errV0 = ntuple(_ -> 1.0, dim)
    errPt0 = 1.0
    err_evo_tot = Float64[]
    err_evo_V = Float64[]
    err_evo_P = Float64[]
    err_evo_it = Float64[]
    itg = 0
    # small pressure correction θc = P_num + ΔPψ = γ_eff·RP + ΔPψ, assembled by the stress kernel and
    # read (alongside the separately-differenced P) by the momentum kernel. Reuses the dyrel.P_num
    # scratch — P_num is no longer materialized separately.
    θc = dyrel.P_num

    # recompute all the DYREL variables
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    DYREL!(dyrel, stokes, rheology, phase_ratios, grid.di, dt, iszero(free_surface) ? nothing : ρg[end])

    # Powell-Hestenes iterations
    for itPH in 1:1000
        # update buoyancy forces
        update_ρg!(ρg, phase_ratios, rheology, args)

        # compute divergence, deviatoric strain rate and pressure residual in one pass
        # isone(itPH) &&
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, true)

        # compute deviatoric stress, refresh τII viscosity, and assemble θc = γ_eff·RP + ΔPψ in one pass
        compute_stress_viscosity_DRYEL!(stokes, θc, dyrel.γ_eff, rheology, phase_ratios, λ_relaxation_PH, dt, viscosity_relaxation, args, viscosity_cutoff, linear_viscosity)
        # update_halo!(stokes.λv)
        # update_halo!(stokes.τ.xx_v)
        # update_halo!(stokes.τ.yy_v)
        # update_halo!(stokes.τ.xy)

        # compute velocity residuals (free-surface stabilization via dt * free_surface)
        @parallel (@idx ni) compute_PH_residual_V!(
            residuals...,
            @velocity(stokes)...,
            stokes.P,
            stokes.ΔPψ,
            @stress(stokes)...,
            ρg...,
            _di.center,
            _di.vertex,
            dt * free_surface,
        )

        # pressure residual stokes.R.RP already computed in compute_∇V_strain_rate_RP! above

        # Residual check. Same normalization as the APT solver (solve!): momentum
        # residuals measured against the pressure span, continuity against the velocity
        # span, so ϵ means the same thing in both solvers, for every component, at every
        # resolution.
        Pspan = nonzero_span(value_span(stokes.P))
        Vspan = nonzero_span(maximum(map(value_span, @velocity(stokes))))
        errV = ntuple(d -> norm_mpi(@views residuals[d][2:(end - 1), 2:(end - 1)]) / Pspan * lx / √(v_dofs[d]), dim)
        errPt = norm_mpi(stokes.R.RP) / Vspan * lx / √(p_dof)
        err = maximum((errV..., errPt))

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
        if err < ϵ && itPH > 1
            converged = true
            break
        end

        # Set tolerance of velocity solve proportional to residual
        if err > err_min * 1.05
            # rel_drop = max(rel_drop * 0.1, ϵ)
            rel_drop = max(rel_drop * 0.1, 1.0e-3)
        end
        if err_min > err
            err_min = err
        end

        # `err_vel` tracks the velocity solve; the outer `err` stays fixed for the whole pass so
        # the convergence test, the return value and `converged` keep reporting the criterion
        # that actually decides convergence. Note the target is set from the mixed-norm `err`
        # while the loop measures the momentum residual alone: `errPt` is normalized by the
        # velocity span and `errV` by the pressure span, so in dimensional problems `ϵ_vel` sits
        # far above `err_vel` from the start and the pass ends after one `nout` window.
        ϵ_vel = err * rel_drop
        err_vel = err
        itPT = 0
        # Initialize dτ for the FSSA-stabilized operator. The in-loop dτ refresh only
        # fires every `nout` iterations; without this the first window of velocity
        # updates would drive the free-surface-stabilization residual term against a
        # dτ tuned for the plain viscous operator (Gershgorin without the FSSA
        # diagonal) and diverge.
        if !iszero(free_surface)
            Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, grid.di, dt, ρg[end])
            update_dτV_α_β!(dyrel)
        end
        while (err_vel > ϵ_vel && itPT ≤ iterMax)
            itPT += 1
            itg += 1
            iter += 1

            # Pseudo-old dudes (only needed by compute_λminV! on residual-check iterations)
            iszero(iter % nout) && foreach(copyto!, residuals0, residuals)

            # compute divergence, deviatoric strain rate and pressure residual in one pass
            compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, true)

            # Deviatoric stress, τII viscosity refresh, and θc = γ_eff·RP + ΔPψ assembly in one pass
            compute_stress_viscosity_DRYEL!(stokes, θc, dyrel.γ_eff, rheology, phase_ratios, λ_relaxation_DR, dt, viscosity_relaxation, args, viscosity_cutoff, linear_viscosity)
            # update_halo!(stokes.λv)
            # batch the vertex-stress halos (+ vertex viscosity, refreshed above in the fused
            # kernel from pre-halo stress) into a single MPI exchange, so shared boundary vertices
            # stay consistent across ranks — matching the original stress→halo→viscosity ordering.
            if linear_viscosity
                update_halo!(stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)
            else
                update_halo!(stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy, stokes.viscosity.ηv)
            end

            # Velocity residuals + damped pseudo-transient velocity update (fused; the small pressure
            # correction θc = γ_eff·RP + ΔPψ was assembled by the stress kernel above; P stays separate)
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
                errV = ntuple(d -> norm_mpi((@views fields.D[d][2:(end - 1), 2:(end - 1)]), (@views residuals[d][2:(end - 1), 2:(end - 1)])) / Pspan * lx / √(v_dofs[d]), dim)
                err_vel = maximum(errV)
                isnan(err_vel) && igg.me == 0 && error("NaN detected in inner loop")

                push!(err_evo_tot, err_vel)
                push!(err_evo_V, err_vel)
                push!(err_evo_P, errPt)
                push!(err_evo_it, iter)

                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err_vel)
                end
                λminV = compute_λminV!(fields, residuals, residuals0, ni, dim)
                @parallel (@idx ni) update_cV!(fields.cV, 2 * √(λminV) * dyrel.c_fact)

                # Optimal pseudo-time steps - can be replaced by AD
                Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, grid.di, dt, iszero(free_surface) ? nothing : ρg[end])

                # Select dτ
                update_dτV_α_β!(dyrel)
            end
        end
        if itPT > iterMax && igg.me == 0
            @warn "DYREL velocity solve exhausted iterMax before reaching ϵ_vel" itPH iter itPT iterMax err_vel ϵ_vel maxlog = 10
        end

        # update pressure
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, args, false)
        @. stokes.P += dyrel.γ_eff .* stokes.R.RP

        iter > total_iterMax && break
    end
    if !converged && igg.me == 0
        @warn "DYREL returned without meeting ϵ — the velocity/pressure fields are not converged" err ϵ iter total_iterMax
    end

    # absorb plastic pressure correction into P (mirrors APT: stokes.P .= θ = P + ΔPψ)
    @. stokes.P += stokes.ΔPψ

    # refresh the ∇V diagnostic from the converged velocity field (it is not stored inside the
    # DYREL/PH loop — see compute_∇V_strain_rate_RP!)
    @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di.vertex)

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
        R0 = (dyrel.Rx0, dyrel.Ry0),
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
        R0 = (dyrel.Rx0, dyrel.Ry0, dyrel.Rz0),
    )
end

# Amplitude scales for the residual normalization. `value_span` is the range of a field's
# values; the velocity scale is the largest component span, because in buoyancy-driven flow the
# cross-axis component can be orders smaller than the dominant one and normalizing the
# continuity residual by it would make the criterion arbitrarily strict. A zero span (cold
# start, V ≡ 0 or P ≡ 0) leaves the residual unnormalized — large, so the solve continues
# rather than exiting spuriously.
@inline function value_span(A)
    mn, mx = extrema(A)
    return mx - mn
end
@inline nonzero_span(s) = iszero(s) ? one(s) : s

# Span over the entries `mask` marks valid, without materializing the selection. An empty mask
# gives zero, which `nonzero_span` then turns into the unnormalized fallback.
@inline function masked_value_span(mask, A)
    T = eltype(A)
    lo = mapreduce((m, a) -> m ? a : typemax(T), min, mask, A)
    hi = mapreduce((m, a) -> m ? a : typemin(T), max, mask, A)
    return hi > lo ? hi - lo : zero(T)
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

    numerator = sum(ntuple(d -> sum_mpi((dv, r, r0) -> dv * (r - r0), fields.dV[d], residuals[d], residuals0[d]), Val(N)))
    denominator = sum(ntuple(d -> sum_mpi(abs2, fields.dV[d]), Val(N)))
    return abs(numerator) / denominator
end

# Reduced-space Rayleigh quotient used by variational DYREL. The masks and arrays are
# deliberately passed as equally-shaped views: eliminated cut-cell and boundary rows must not
# influence the damping estimate for the active velocity operator.
@inline rayleigh_quotient(numerator, denominator) = iszero(denominator) ? zero(denominator) : abs(numerator) / denominator

function masked_λminV(dV::NTuple{N}, residuals::NTuple{N}, residuals0::NTuple{N}, masks::NTuple{N}) where {N}
    numerator = sum(ntuple(d -> sum_mpi((m, dv, r, r0) -> m ? dv * (r - r0) : zero(dv), masks[d], dV[d], residuals[d], residuals0[d]), Val(N)))
    denominator = sum(ntuple(d -> sum_mpi((m, dv) -> m ? abs2(dv) : zero(abs2(dv)), masks[d], dV[d]), Val(N)))
    return rayleigh_quotient(numerator, denominator)
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
