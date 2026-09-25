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
- `pressure_relaxation`: Relaxation factor for the Powell-Hestenes pressure update. Default: `1`.
- `iterMax_PH`: Maximum number of Powell-Hestenes passes. Default: `1.0e3`.
- `iterMax_DR`: Maximum number of iterations for each dynamic-relaxation solve. Default: `50.0e3`.
- `iterMax`: Compatibility alias for `iterMax_DR`; used when `iterMax_DR` is not given.
- `total_iterMax`: Maximum number of total dynamic-relaxation iterations. Default: `50.0e3`.
- `nout`: Output frequency for residuals. Default: `100`.
- `rel_drop`: Relative residual drop tolerance. Default: `1.0e-2`.
- `verbose_PH`: Print Powell-Hestenes iteration info. Default: `true`.
- `verbose_DR`: Print Dynamic Relaxation iteration info. Default: `true`.
- `linear_viscosity`: Whether to use linear viscosity. Default: `false`.
- `free_surface`: Include the density-gradient free-surface stabilization term. Default: `false`.
- `adjoint`: Run `solve_DYREL_adjoint!` after convergence and before updating
  history-dependent state. Default: `false`.
- `observation`: Objective `J` of the adjoint solve, required with `adjoint = true`: a box
  `(; field, center, half_width)` with `J = Σ field` over the nodes inside it, or a weighted
  field `(; field, weights)` with `J = Σ weights · field` and `weights` of the size of the
  observed field (see `observation_mask`). `field` is `:Vx`, `:Vy` or `:P`. Default: `nothing`.
- `gradients`: Optional buffers from `material_controls`, filled by the adjoint
  solve with derivatives with respect to the actual material parameters. `G` is returned
  on the center grid; phase-specific density-parameter gradients have size `(nphases, ni...)`. Default:
  `nothing`.
- `η_multiplier`: Optional cell-wise viscosity scaling, given as a named tuple
  `(; center, vertex)` of arrays matching `stokes.viscosity.η` and `.ηv`. Applied right after
  the rheology-driven `compute_viscosity!` and before the DYREL coefficients are built, so
  the preconditioner stays consistent with the scaled viscosity. Used for a gradient-test to verify
  adjoint gradients. Pair it with `linear_viscosity = true`, otherwise the in-loop τII viscosity
  refresh overwrites it.
  Default: `nothing`.
- `update_material`: Recompute viscosity and buoyancy from `rheology`. Set to `false` when
  those fields are prescribed by the caller. Default: `true`.

Options may be passed either as plain keywords or bundled as a single
`kwargs = (; ...)` NamedTuple.
"""
function solve_DYREL!(stokes::JustRelax.StokesArrays, args...; kwargs...)
    return solve_DYREL!(backend(stokes), stokes, args...; kwargs = flatten_solver_kwargs(kwargs))
end

# entry point for extensions
solve_DYREL!(::CPUBackendTrait, stokes, args...; kwargs) = _solve_DYREL!(stokes, args...; kwargs...)

"""
    solve_VariationalDYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ,
        rheology, args, grid, dt, igg; kwargs...)
    solve_VariationalDYREL!(stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ,
        rheology, args, grid, dt, igg; adjoint = true, observation, kwargs...)

Solve the 2D variational Stokes problem with DYREL relaxation and the
`RockRatio` volume weights. This is a separate entry point from
`solve_DYREL!`; the latter remains the standard, unweighted DYREL solver.

Center fractions weight pressure and normal stress, vertex fractions weight
shear stress, and face fractions weight momentum rows. Rows whose volume
fraction vanishes are eliminated rather than solved with air properties.

# Arguments (in the following order)
- `stokes`: `JustRelax.StokesArrays` containing the simulation fields.
- `stokes_ad`: `JustRelax.AdjointStokesArrays` for the adjoint solve; only needed with
  `adjoint = true`.
- `ρg`: buoyancy forces arrays.
- `dyrel`: DYREL-specific parameters and fields, built with the same `ϕ`.
- `flow_bcs`: `AbstractFlowBoundaryConditions` defining velocity boundary conditions.
- `phase_ratios`: `JustPIC.PhaseRatios` for material phase tracking.
- `ϕ`: `JustRelax.RockRatio` carrying the cell, vertex and face volume fractions.
- `rheology`: Material properties and rheological laws.
- `args`: Tuple of additional arguments needed to update viscosity, stress, and buoyancy forces.
- `grid`: `Geometry{2}` object carrying grid spacing and staggered-grid coordinates. A legacy
  2D spacing tuple or named tuple is also accepted and converted to a uniform `Geometry`.
- `dt`: Time step.
- `igg`: `IGG` object for global grid information (MPI).

# Keyword Arguments
- `air_phase`: Phase index excluded from material averages; `0` disables the correction. Default: `0`.
- `viscosity_cutoff`: Limits for viscosity `(min, max)`. Default: `(-Inf, Inf)`.
- `viscosity_relaxation`: Relaxation factor for viscosity updates. Default: `1.0e-2`.
- `λ_relaxation_DR`: Relaxation factor for dynamic relaxation. Default: `1`.
- `λ_relaxation_PH`: Relaxation factor for Powell-Hestenes iterations. Default: `1`.
- `pressure_relaxation`: Relaxation factor for the Powell-Hestenes pressure update. Default: `1`.
- `iterMax_PH`: Maximum number of Powell-Hestenes passes. Default: `1.0e3`.
- `iterMax_DR`: Maximum number of iterations for each dynamic-relaxation solve. Default: `50.0e3`.
- `iterMax`: Alias for `iterMax_DR`; used when `iterMax_DR` is not given.
- `total_iterMax`: Maximum number of total dynamic-relaxation iterations. Default: `50.0e3`.
- `nout`: Output frequency for residuals. Default: `100`.
- `rel_drop`: Relative residual drop tolerance. Default: `1.0e-2`.
- `verbose_PH`: Print Powell-Hestenes iteration info. Default: `true`.
- `verbose_DR`: Print Dynamic Relaxation iteration info. Default: `true`.
- `linear_viscosity`: Whether to use linear viscosity. Default: `false`.
- `free_surface`: Include the density-gradient free-surface stabilization term. Default: `false`.
- `adjoint`: Run `solve_VariationalDYREL_adjoint!` after convergence and before updating
  history-dependent state. `ϕ` and the validity masks are frozen for the adjoint. Default: `false`.
- `observation`: Objective of the adjoint solve, a box or a weighted field, as for `solve_DYREL!`. Default: `nothing`.
- `gradients`: Optional buffers from `material_controls`, filled by the adjoint solve. Default: `(;)`.
- `η_multiplier`: Optional cell-wise viscosity scaling `(; center, vertex)`, applied right after
  the rheology-driven viscosity update, for gradient tests. Pair it with `linear_viscosity = true`.
  Default: `nothing`.

Options may be passed either as plain keywords or bundled as a single
`kwargs = (; ...)` NamedTuple.
"""
function solve_VariationalDYREL!(stokes::JustRelax.StokesArrays, args...; kwargs...)
    return solve_VariationalDYREL!(
        backend(stokes), stokes, args...; kwargs = flatten_solver_kwargs(kwargs)
    )
end

solve_VariationalDYREL!(::CPUBackendTrait, stokes, args...; kwargs) =
    _solve_VariationalDYREL!(stokes, args...; kwargs...)

function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        stokes_ad::Union{Nothing, JustRelax.AdjointStokesArrays},
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
        pressure_relaxation = 1,
        iterMax = nothing,
        iterMax_PH = 1.0e3,
        iterMax_DR = isnothing(iterMax) ? 50.0e3 : iterMax,
        total_iterMax = 50.0e3,
        nout = 100,
        rel_drop = 1.0e-2,
        verbose_PH = true,
        verbose_DR = true,
        linear_viscosity = false,
        free_surface = false,
        update_material = true,
        adjoint = false,
        observation = nothing,
        gradients = (;),
        η_multiplier = nothing,
        kwargs...,
    ) where {N}

    adjoint && isnothing(stokes_ad) &&
        throw(ArgumentError("adjoint = true requires AdjointStokesArrays as the second argument"))

    check_periodic_bcs(stokes, flow_bcs, igg, grid.di.center)

    @copy stokes.P0 stokes.P

    dim = Val(N)
    v_dofs = velocity_dofs(dim, periodic_dims(stokes))
    p_dof = pressure_dof(dim)
    di = grid.di
    _di = grid._di
    di_center = di.center
    ni = size(stokes.P)
    lx = grid.max_li

    igg.me == 0 && @printf("\n######## Running forward Stokes solver (DYREL) ########\n")

    residuals = @residuals(stokes.R)
    fields = dyrel_fields(dyrel, dim)

    # errors
    err = 1.0
    iter = 0

    # solver loop
    residuals0 = fields.R0

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # reset plastic multiplier at the beginning of the time step
    stokes.λ .= 0.0
    reset_dyrel_vertex_λ!(dyrel_vertex_λ(stokes, dim))

    # Iteration loop
    err_min = Inf
    err = 1.0
    errV0 = ntuple(_ -> 1.0, dim)
    errPt0 = 1.0
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ
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
    if update_material
        compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
        # impose a cell-wise viscosity scaling the per-phase rheology cannot express. This has to
    # happen before `DYREL!`, so that the Gershgorin bounds and the preconditioner are built
    # from the viscosity the solve actually uses.
    if !isnothing(η_multiplier)
        stokes.viscosity.η .*= η_multiplier.center
        stokes.viscosity.ηv .*= η_multiplier.vertex
    end
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    end
    DYREL!(dyrel, stokes, rheology, phase_ratios, grid.di, dt; CFL = dyrel.CFL)
    if free_surface
        apply_free_surface_diagonal!(fields.D[N], fields.λmaxV[N], ρg[end], grid.di.center, dt)
        update_dτV_α_β!(dyrel)
    end

    # Powell-Hestenes iterations
    for itPH in 1:Int(iterMax_PH)
        # update buoyancy forces
        update_material && update_ρg!(ρg, phase_ratios, rheology, args)

        # compute divergence, deviatoric strain rate and pressure residual in one pass
        # isone(itPH) &&
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, true; args...)

        # compute deviatoric stress, refresh τII viscosity, and assemble θc = γ_eff·RP + ΔPψ in one pass
        compute_stress_viscosity_DRYEL!(stokes, θc, dyrel.γ_eff, rheology, phase_ratios, λ_relaxation_PH, dt, viscosity_relaxation, args, viscosity_cutoff, linear_viscosity)
        update_stress_halo!(stokes, dim, linear_viscosity)
        free_surface_stress_bcs!(stokes, flow_bcs, dim)
        # update_halo!(stokes.λv)
        # update_halo!(stokes.τ.xx_v)
        # update_halo!(stokes.τ.yy_v)
        # update_halo!(stokes.τ.xy)

        # compute velocity residuals
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

        # Residual check
        # Scale-free, as in the variational solver: momentum residuals against the pressure span
        # over the domain length, continuity against the velocity scale. (The relative-or-absolute
        # form used before accepted any residual that is small in the problem's units, e.g. a
        # continuity residual of ~1e-16 1/s for velocities of ~1e-9 m/s.)
        Pspan = nonzero_span(maximum_mpi(stokes.P) - minimum_mpi(stokes.P))
        Vscale = maximum(V -> max(maximum_mpi(V), -minimum_mpi(V)), @velocity(stokes))
        Vspan = continuity_velocity_scale(Vscale, Pspan, lx, maximum_mpi(stokes.viscosity.η), ϵ)
        errV = ntuple(d -> norm_mpi(residuals[d]) / √(v_dofs[d]) / Pspan * lx, dim)
        RP_rms = norm_mpi(stokes.R.RP) / √(p_dof)
        errPt = RP_rms * lx / Vspan
        if itPH ≤ 2
            errV0 = map(x -> x + eps(), errV)
            errPt0 = errPt + eps()
        end
        err = maximum((errV..., errPt))

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

        # Target a drop of `errV`, the residual the loop below measures, as in the variational
        # solver; `max(…, ϵ)` guards a zero momentum residual, `Inf` forces a first check.
        ϵ_vel = max(maximum(errV) * rel_drop, ϵ)
        err_vel = Inf
        itPT = 0
        while (err_vel > ϵ_vel && itPT ≤ iterMax_DR)
            itPT += 1
            itg += 1
            iter += 1

            # Pseudo-old dudes (only needed by compute_λminV! on residual-check iterations)
            iszero(iter % nout) && foreach(copyto!, residuals0, residuals)

            # compute divergence, deviatoric strain rate and pressure residual in one pass
            compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, true; args...)

            # Deviatoric stress, τII viscosity refresh, and θc = γ_eff·RP + ΔPψ assembly in one pass
            compute_stress_viscosity_DRYEL!(stokes, θc, dyrel.γ_eff, rheology, phase_ratios, λ_relaxation_DR, dt, viscosity_relaxation, args, viscosity_cutoff, linear_viscosity)
            update_stress_halo!(stokes, dim, linear_viscosity)
            free_surface_stress_bcs!(stokes, flow_bcs, dim)

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
            free_surface_bcs!(
                stokes, flow_bcs, stokes.viscosity.η_vep, grid.di.velocity..., dim
            )
            update_halo!(@velocity(stokes)...)

            # Residual check
            if iszero(iter % nout)

                # D·(stored residual) is the raw momentum residual; normalized exactly like the
                # outer check, so ϵ_vel compares like with like. P is fixed within a pass, so the
                # outer Pspan is still current here.
                errV_in = ntuple(d -> norm_mpi(fields.D[d] .* residuals[d]) / √(v_dofs[d]) / Pspan * lx, dim)
                err_vel = maximum(errV_in)
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
                Gershgorin_Stokes_SchurComplement!(dim, fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, grid.di, dt)
                free_surface && apply_free_surface_diagonal!(fields.D[N], fields.λmaxV[N], ρg[end], grid.di.center, dt)

                # Select dτ
                update_dτV_α_β!(dyrel)
            end
        end

        # update pressure
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, _di, ni, dt, false; args...)
        @. stokes.P += pressure_relaxation * dyrel.γ_eff * stokes.R.RP

        iter > total_iterMax && break
    end

    adjoint_out = if adjoint
        solve_DYREL_adjoint!(
            stokes,
            stokes_ad,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            λ_relaxation_DR,
            λ_relaxation_PH,
            pressure_relaxation,
            free_surface,
            iterMax_PH,
            iterMax_DR,
            total_iterMax,
            nout,
            rel_drop,
            verbose_PH,
            verbose_DR,
            observation,
            gradients,
            viscosity_cutoff,
            viscosity_relaxation,
            linear_viscosity,
            kwargs...,
        )
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

    out = (; iter, err_evo_it, err_evo_V, err_evo_P, err_evo_tot)
    return adjoint ? (; out..., adjoint = adjoint_out) : out

end

# forward-only entry point: no adjoint arrays
function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        ρg,
        dyrel,
        flow_bcs::AbstractFlowBoundaryConditions,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        grid::Geometry,
        dt,
        igg::IGG;
        kwargs...,
    )
    return _solve_DYREL!(stokes, nothing, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg; kwargs...)
end

function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        ρg,
        dyrel,
        flow_bcs::AbstractFlowBoundaryConditions,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        di::Union{NTuple{2, <:Real}, NTuple{3, <:Real}, NamedTuple},
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

@inline dyrel_fields(::JustRelax.DYREL, ::Val{N}) where {N} = error("Unsupported dimension $N")

function update_stress_halo!(stokes::JustRelax.StokesArrays, ::Val{2}, linear_viscosity)
    if linear_viscosity
        update_halo!(stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)
    else
        update_halo!(stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy, stokes.viscosity.ηv)
    end
    return nothing
end

function update_stress_halo!(stokes::JustRelax.StokesArrays, ::Val{3}, _linear_viscosity)
    # The 3D momentum kernels use center viscosity directly; only edge shear stresses need halos.
    update_halo!(stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)
    return nothing
end

# Substitutes a unit scale for a field that is identically zero up to accumulated roundoff, so the
# relative residual norms it normalizes stay finite: the pressure field of a boundary-driven shear
# problem and the velocity field of a hydrostatic one are both of this kind.
#
# The threshold is absolute while the span is dimensional, so a problem nondimensionalized such
# that its true span falls below `sqrt(eps)` has its error deflated instead. Replacing it needs an
# absolute residual scale to fall back on, and no single one covers every problem: the buoyancy
# norm is exactly zero for boundary-driven flow, the pressure span for pure shear. `errPt` has such
# a fallback (see `_solve_VariationalDYREL!`); `errV` does not.
@inline nonzero_span(s) = abs(s) ≤ sqrt(eps(typeof(s))) ? one(s) : s

# Velocity scale of the continuity check `errPt = RP·lx/V`: the velocity magnitude, floored at a
# small fraction of the velocity the pressure span could drive through the stiffest material,
# `Vref = Pspan·lx/η_max`. A field at rest carries round-off noise only, `RP ≈ c·eps·Vref/lx` with
# c ≲ 1, where `errPt` would otherwise compare noise with noise and never converge. Flooring at
# `10·eps/ϵ·Vref` makes that noise read as `≈ 0.1·c·ϵ`, for any tolerance. A moving field is far
# above the floor (2e-9·Vref at ϵ = 1e-6), and even one below it has its relative continuity
# error off by only `floor/V`. The floor does not depend on `dt`, which in a viscous solve can be
# an arbitrary placeholder.
@inline function continuity_velocity_scale(Vscale, Pspan, lx, η_max, ϵ)
    floor_fraction = min(one(Vscale), 10 * eps(typeof(Vscale)) / ϵ)
    V = max(Vscale, floor_fraction * Pspan * lx / η_max)
    return iszero(V) || !isfinite(V) ? one(V) : V
end
@inline volumetric_compliance(ηb) = ηb > 0 ? inv(ηb) : zero(ηb)

@inline function masked_extrema(mask, A)
    lo = mapreduce((m, a) -> m ? a : typemax(a), min, mask, A)
    hi = mapreduce((m, a) -> m ? a : typemin(a), max, mask, A)
    return lo, hi
end

@inline function masked_value_span(mask, A)
    lo, hi = masked_extrema(mask, A)
    return hi > lo ? hi - lo : zero(eltype(A))
end

@inline function masked_value_scale(mask, A)
    lo, hi = masked_extrema(mask, A)
    return hi > lo ? max(hi - lo, abs(hi), abs(lo)) : zero(eltype(A))
end

# Total compliance of the valid pressure rows. Zero for an incompressible rheology, where the
# uniform volumetric mode carries no pressure correction and `relax_volumetric_mode!` is a no-op.
function volumetric_compliance_total(ηb, mask)
    return sum_mpi((ηbᵢ, valid) -> valid ? volumetric_compliance(ηbᵢ) : zero(ηbᵢ), ηb, mask)
end

# Variational counterpart: a uniform shift of the retained pressures moves each residual by
# `ϕ.center / ηb`, because the continuity residual is weighted by the rock fraction.
function volumetric_compliance_total(ηb, ϕ::JustRelax.RockRatio, mask)
    return sum_mpi(
        (ηbᵢ, ϕᵢ, valid) -> valid ? ϕᵢ * volumetric_compliance(ηbᵢ) : zero(ηbᵢ),
        ηb, ϕ.center, mask,
    )
end

function relax_volumetric_mode!(P, RP, ηb, mask, relaxation = 1, compliance = volumetric_compliance_total(ηb, mask))
    iszero(compliance) && return nothing
    δ = sum_mpi((RPᵢ, valid) -> valid ? RPᵢ : zero(RPᵢ), RP, mask) / compliance
    @. P += ifelse(mask, relaxation * δ, zero(δ))
    return nothing
end

@inline rayleigh_quotient(numerator, denominator) = iszero(denominator) ? zero(denominator) : abs(numerator) / denominator

function masked_λminV(dV::NTuple{N}, residuals::NTuple{N}, residuals0::NTuple{N}, masks::NTuple{N}) where {N}
    numerator = sum(ntuple(d -> sum_mpi((m, dv, r, r0) -> m ? dv * (r - r0) : zero(dv), masks[d], dV[d], residuals[d], residuals0[d]), Val(N)))
    denominator = sum(ntuple(d -> sum_mpi((m, dv) -> m ? abs2(dv) : zero(abs2(dv)), masks[d], dV[d]), Val(N)))
    return rayleigh_quotient(numerator, denominator)
end

@inline pressure_dof(N) = prod(global_grid_size(N))

# Number of momentum unknowns per direction, used to turn the residual norms into per-degree-of-
# freedom quantities. `global_grid_size` counts vertices, so a direction of `n` cells contributes
# its `n - 1` interior faces, or all `n` of them when it is periodic and the two boundary faces
# collapse onto a single unknown.
function velocity_dofs(::Val{N}, periodic::NTuple{N, Bool}) where {N}
    global_size = global_grid_size(Val(N))
    return ntuple(Val(N)) do d
        @inline
        prod(i -> i == d ? global_size[i] - 2 + periodic[d] : global_size[i] - 1, 1:N)
    end
end

velocity_dofs(::Val{N}) where {N} = velocity_dofs(Val(N), ntuple(_ -> false, Val(N)))

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
