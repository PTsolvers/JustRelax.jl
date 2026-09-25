"""
    observation_mask(stokes_ad, grid, observation)

Resolve the objective `J` of the adjoint solve. `observation` is either

  - a box, `(; field, center, half_width)`: `J = Σ field` over the nodes of `field` whose
    coordinates lie within `half_width` of `center`, or
  - a weighted field, `(; field, weights)`: `J = Σ weights · field`, with `weights` an array of
    the size of the observed field, i.e. `∂J/∂field`. Weights can follow the material, e.g. a
    phase fraction, to track a body as it moves.

`field` is `:Vx`, `:Vy` or `:P`. Returns `(; field, target, i, j, weights)`, where `target` is the
adjoint array the objective seeds; `i, j` are the box indices (`nothing` for weights) and
`weights` is `nothing` for a box.
"""
function observation_mask(stokes_ad, grid, observation)
    field = observation.field

    target, x, y, offset = if field === :Vx
        stokes_ad.V.Vx, grid.xvi[1], grid.xci[2], (0, 1)
    elseif field === :Vy
        stokes_ad.V.Vy, grid.xci[1], grid.xvi[2], (1, 0)
    elseif field === :P
        stokes_ad.P, grid.xci[1], grid.xci[2], (0, 0)
    else
        throw(ArgumentError("observation field must be :Vx, :Vy, or :P"))
    end

    if haskey(observation, :weights)
        weights = observation.weights
        size(weights) == size(target) || throw(
            DimensionMismatch("observation weights must have the size of the $field field, $(size(target))")
        )
        any(!iszero, weights) || throw(ArgumentError("the observation weights are zero everywhere"))
        return (; field, target, i = nothing, j = nothing, weights)
    end

    (; center, half_width) = observation
    i = findall(xi -> abs(xi - center[1]) ≤ half_width[1], x) .+ offset[1]
    j = findall(yj -> abs(yj - center[2]) ≤ half_width[2], y) .+ offset[2]
    (isempty(i) || isempty(j)) && throw(ArgumentError("the observation region does not contain any $field nodes"))
    return (; field, target, i, j, weights = nothing)
end

# Seed the adjoint of the observed field with -∂J/∂field: -1 on every node of a box, -weights
# for a weighted objective. `target` has just been zeroed by `initialize_adjoint_iteration!`.
function seed_observation!(observation)
    (; target, i, j, weights) = observation
    if isnothing(weights)
        target[i, j] .= -1.0
    else
        target .= .-weights
    end
    return nothing
end

"""
    solve_DYREL_adjoint!(stokes, stokes_ad, args...; kwargs...)

Adjoint entry point called by `solve_DYREL!(...; adjoint=true)` after the
forward iterations converge and before history-dependent state is updated.
"""
function solve_DYREL_adjoint!(
        stokes,
        stokes_ad,
        ρg,
        dyrel,
        flow_bcs,
        phase_ratios,
        rheology,
        args,
        grid::Geometry{N},
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
        gradients = (;),
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        linear_viscosity = false,
        kwargs...,
    ) where {N}
    dim = Val(N)
    v_dofs = velocity_dofs(dim, periodic_dims(stokes))
    p_dof = pressure_dof(dim)
    _di = grid._di
    ni = size(stokes.P)
    lx = grid.max_li
    # adjoint velocity unknowns, for the scale of the continuity check
    λVi = (
        @view(stokes_ad.λV.Vx[2:(size(stokes.R.Rx, 1) + 1), 2:(size(stokes.R.Rx, 2) + 1)]),
        @view(stokes_ad.λV.Vy[2:(size(stokes.R.Ry, 1) + 1), 2:(size(stokes.R.Ry, 2) + 1)]),
    )
    adjoint_velocity_residuals = (
        @view(stokes_ad.V.Vx[2:(size(stokes.R.Rx, 1) + 1), 2:(size(stokes.R.Rx, 2) + 1)]),
        @view(stokes_ad.V.Vy[2:(size(stokes.R.Ry, 1) + 1), 2:(size(stokes.R.Ry, 2) + 1)]),
    )

    size(stokes_ad.R.Rx) == size(stokes.R.Rx) &&
        size(stokes_ad.R.Ry) == size(stokes.R.Ry) || throw(
        ArgumentError("stokes and stokes_ad must use the same periodic boundary allocation")
    )

    get(args, :P, nothing) === stokes.P ||
        throw(ArgumentError("the adjoint solve requires args.P === stokes.P"))

    igg.me == 0 && @printf("\n######## Running adjoint Stokes solver (DYREL) ########\n")

    # Reuse the forward DYREL parameters, but not its iteration history.
    dyrel.dVxdτ .= 0
    dyrel.dVydτ .= 0
    stokes_ad.viscosity.η .= 0
    stokes_ad.viscosity.ηv .= 0

    # Iteration loop
    err_min = Inf
    errV0 = ntuple(_ -> 1.0, dim)
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ

    isnothing(observation) && throw(ArgumentError("an observation region is required for the adjoint solve"))
    observation = observation_mask(stokes_ad, grid, observation)

    converged = false
    for itPH in 1:Int(iterMax_PH)

        initialize_adjoint_iteration!(stokes_ad, ni)

        # Init observation points
        seed_observation!(observation)

        adjoint_residual!(
            stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, _di, ni, dt,
            λ_relaxation_PH, free_surface, viscosity_relaxation, viscosity_cutoff, linear_viscosity,
        )

        # Residual check
        errV = ntuple(d -> norm_mpi(adjoint_velocity_residuals[d]) / √(v_dofs[d]), dim)
        # Scale-free criteria, as in the variational adjoint: the momentum adjoints relative to
        # the largest initial one (the objective seeds only some components), the continuity
        # adjoint, the divergence of λV, like the forward `RP·lx/Vspan`. An absolute floor would
        # accept an unconverged pressure, since the adjoint carries the units of the problem.
        if isone(itPH)
            errV0 = ntuple(_ -> maximum(errV) + eps(), dim)
        end
        # λV is exactly zero only on the first pass
        λVspan = maximum(λ -> max(maximum_mpi(λ), -minimum_mpi(λ)), λVi)
        λVspan = iszero(λVspan) ? one(λVspan) : λVspan
        errPt = norm_mpi(stokes_ad.P) / √(p_dof) * lx / λVspan
        err = maximum((ntuple(d -> errV[d] / errV0[d], dim)..., errPt))

        if verbose_PH && igg.me == 0
            errV_msg = join(
                ntuple(d -> @sprintf("R%d=%1.3e %1.3e", d, errV[d], errV[d] / errV0[d]), dim),
                ", ",
            )
            @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[%s, Rp=%1.3e] \n", itPH, iter, iter / ni[1], err, errV_msg, errPt)
        end
        igg.me == 0 && isnan(err) && error("NaN detected in outer loop")
        igg.me == 0 && err > 1.0e10 && error("Kaboom! Error > 1e10 in outer loop")
        if err < ϵ
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

        ϵ_vel = err * rel_drop
        itPT = 0
        while (err > ϵ_vel && itPT ≤ iterMax_DR)
            itPT += 1
            iter += 1

            initialize_adjoint_iteration!(stokes_ad, ni)

            # Init observation points
            observation.field !== :P && seed_observation!(observation)

            adjoint_residual!(
                stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, _di, ni, dt,
                λ_relaxation_DR, free_surface, viscosity_relaxation, viscosity_cutoff, linear_viscosity,
            )

            @parallel (@idx ni) update_adjoint_V_damping_DR!(
                stokes_ad.λV.Vx,
                stokes_ad.λV.Vy,
                dyrel.dVxdτ,
                dyrel.dVydτ,
                stokes_ad.V.Vx,
                stokes_ad.V.Vy,
                stokes_ad.P,
                dyrel.γ_eff,
                dyrel.Dx,
                dyrel.Dy,
                dyrel.αVx,
                dyrel.αVy,
                dyrel.βVx,
                dyrel.βVy,
                dyrel.dτVx,
                dyrel.dτVy,
                _di.center,
            )

            if iszero(iter % nout)
                errV = ntuple(
                    d -> norm_mpi(adjoint_velocity_residuals[d]) / √(v_dofs[d]), dim
                )
            end

            # Residual check
            if iszero(iter % nout)
                # same reference as the outer check: relative to the value after the first `nout`
                # iterations, a residual already at round-off by then can never drop by `rel_drop`
                err = maximum(ntuple(d -> errV[d] / errV0[d], dim))
                isnan(err) && igg.me == 0 && error("NaN detected in inner loop")

                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
                end
            end

        end

        @. stokes_ad.λP += pressure_relaxation * dyrel.γ_eff * stokes_ad.P

        iter > total_iterMax && break
    end
    if !converged && igg.me == 0
        @warn "adjoint DYREL returned without meeting ϵ — the sensitivities are not converged" err ϵ iter total_iterMax
    end

    # sensitivity evaluation
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, _di, ni, λ_relaxation_PH, dt, igg,
        gradients, args; viscosity_cutoff, free_surface,
    )

    # Do not carry adjoint iteration history into the next forward solve.
    dyrel.dVxdτ .= 0
    dyrel.dVydτ .= 0

    return stokes_ad
end

# One reverse sweep through the forward kernels, in the reverse order of the forward update:
# momentum residual, fused stress and viscosity update, strain rate and continuity residual,
# boundary conditions. Seeds are set by the caller.
function adjoint_residual!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, _di, ni, dt,
        λ_relaxation, free_surface, viscosity_relaxation, viscosity_cutoff, linear_viscosity,
    )
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, _di, ni, rheology, phase_ratios, args;
        free_surface_dt = dt * free_surface,
    )
    enzyme_compute_stress_viscosity_DRYEL!(
        stokes, stokes_ad, dyrel.P_num, dyrel.γ_eff, rheology, phase_ratios,
        λ_relaxation, dt, viscosity_relaxation, args, viscosity_cutoff, linear_viscosity,
    )
    enzyme_compute_∇V_strain_rate_RP!(stokes, stokes_ad, dyrel, rheology, phase_ratios, _di, ni, dt, args)
    enzyme_flow_bcs!(stokes, stokes_ad, flow_bcs)
    return nothing
end
