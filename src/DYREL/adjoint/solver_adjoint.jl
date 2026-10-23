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
        linear_viscosity = false,
        kwargs...,
    ) where {N}
    dim = Val(N)
    # Differentiate the unrelaxed plastic multiplier: at convergence λ equals its trial value,
    # and holding the relaxed history constant would scale the plastic tangent by the relaxation.
    λ_relaxation = one(eltype(stokes.P))
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
    constitutive_scratch = linear_viscosity ? nothing : (;
            η_target = copy(stokes.viscosity.η),
            ηv_target = copy(stokes.viscosity.ηv),
            η = similar(stokes.viscosity.η),
            ηv = similar(stokes.viscosity.ηv),
            feedback = similar(stokes.viscosity.η),
            feedback_v = similar(stokes.viscosity.ηv),
            feedback_λ = Ref{Any}(nothing),
            τ = map(similar, (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)),
            τ_v = map(similar, (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)),
            τII = similar(stokes.τ.II),
            θ = similar(stokes.ΔPψ),
        )
    if !isnothing(constitutive_scratch)
        compute_constitutive_tangent_amplification!(
            constitutive_scratch, stokes, rheology, phase_ratios,
            λ_relaxation, dt, args, viscosity_cutoff,
        )
        update_adjoint_dτV_α_β!(
            dyrel, constitutive_scratch.η, constitutive_scratch.ηv,
        )
    end

    # Iteration loop
    err_min = Inf
    errV0 = ntuple(_ -> 0.0, dim)
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
            λ_relaxation, free_surface, viscosity_cutoff, linear_viscosity,
            constitutive_scratch,
        )

        # Residual check
        errV = ntuple(d -> norm_mpi(adjoint_velocity_residuals[d]) / √(v_dofs[d]), dim)
        # Scale-free criteria, as in the variational adjoint: the momentum adjoints relative to
        # the largest initial one (the objective seeds only some components), the continuity
        # adjoint, the divergence of λV, like the forward `RP·lx/Vspan`. An absolute floor would
        # accept an unconverged pressure, since the adjoint carries the units of the problem.
        errV0 = momentum_residual_reference(errV0, errV)
        λVspan = maximum(λ -> max(maximum_mpi(λ), -minimum_mpi(λ)), λVi)
        # λV is zero before its first update and gives the continuity check no scale yet
        has_λV_scale = !iszero(λVspan)
        errPt = has_λV_scale ? norm_mpi(stokes_ad.P) / √(p_dof) * lx / λVspan : zero(λVspan)
        errV_relative = relative_momentum_residual(errV, errV0)
        err = has_λV_scale ? maximum((errV_relative..., errPt)) : maximum(errV_relative)

        if verbose_PH && igg.me == 0
            errV_msg = join(
                ntuple(d -> @sprintf("R%d=%1.3e %1.3e", d, errV[d], errV[d] / errV0[d]), dim),
                ", ",
            )
            @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[%s, Rp=%1.3e] \n", itPH, iter, iter / ni[1], err, errV_msg, errPt)
        end
        isnan(err) && error("NaN detected in outer loop")
        err > 1.0e10 && error("Kaboom! Error > 1e10 in outer loop")
        # A pressure objective has no initial momentum residual: its first outer pass transfers
        # the pressure seed into λP. Do not accept that pre-update state as converged.
        pending_pressure_seed = isone(itPH) && observation.field === :P
        if err < ϵ && !pending_pressure_seed
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

            # A pressure seed enters the Schur correction and the λP update below
            seed_observation!(observation)

            adjoint_residual!(
                stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, _di, ni, dt,
                λ_relaxation, free_surface, viscosity_cutoff, linear_viscosity,
                constitutive_scratch,
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

            # Residual check
            if iszero(iter % nout)
                errV = ntuple(
                    d -> norm_mpi(adjoint_velocity_residuals[d]) / √(v_dofs[d]), dim
                )
                # same reference as the outer check: relative to the value after the first `nout`
                # iterations, a residual already at round-off by then can never drop by `rel_drop`
                errV0 = momentum_residual_reference(errV0, errV)
                err = maximum(relative_momentum_residual(errV, errV0))
                isnan(err) && error("NaN detected in inner loop")

                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
                end
            end

        end

        @. stokes_ad.λP += pressure_relaxation * dyrel.γ_eff * stokes_ad.P

        iter > total_iterMax && break
    end
    if !converged
        # Leave the shared DYREL state ready for a subsequent solve, but do not calculate or
        # return sensitivities from an unconverged adjoint state.
        dyrel.dVxdτ .= 0
        dyrel.dVydτ .= 0
        !isnothing(constitutive_scratch) && update_dτV_α_β!(dyrel)
        error(
            "DYREL adjoint did not converge " *
                "(err=$err, ϵ=$ϵ, iterations=$iter, total_iterMax=$total_iterMax)"
        )
    end

    # sensitivity evaluation
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, rheology, _di, ni, λ_relaxation, dt, igg,
        gradients, args; viscosity_cutoff, free_surface, constitutive_scratch,
    )

    # Do not carry adjoint iteration history into the next forward solve.
    dyrel.dVxdτ .= 0
    dyrel.dVydτ .= 0
    !isnothing(constitutive_scratch) && update_dτV_α_β!(dyrel)

    return stokes_ad
end

# One reverse sweep through the final forward equations: momentum residual, the separate stress
# kernel with its local implicit viscosity correction, strain rate and continuity residual, then
# boundary conditions. The relaxed forward viscosity update is not differentiated.
function adjoint_residual!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, _di, ni, dt,
        λ_relaxation, free_surface, viscosity_cutoff, linear_viscosity,
        constitutive_scratch,
    )
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, _di, ni, rheology, phase_ratios, args;
        free_surface_dt = dt * free_surface,
    )
    enzyme_compute_stress_DRYEL!(
        stokes, stokes_ad, rheology, phase_ratios, λ_relaxation, dt, args,
        viscosity_cutoff, linear_viscosity, constitutive_scratch,
    )
    enzyme_compute_∇V_strain_rate_RP!(stokes, stokes_ad, dyrel, rheology, phase_ratios, _di, ni, dt, args)
    enzyme_flow_bcs!(stokes, stokes_ad, flow_bcs)
    return nothing
end
