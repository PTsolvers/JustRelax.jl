"""
    solve_VariationalDYREL_adjoint!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
        maskV, maskP, air_phase, kwargs...,
    )

Adjoint entry point called by `solve_VariationalDYREL!(...; adjoint=true)` after the forward
iterations converge and before history-dependent state is updated. Mirrors
[`solve_DYREL_adjoint!`](@ref) on the reduced (rock) space of the variational solver:
`ϕ`, the velocity masks `maskV` and the pressure mask `maskP` are those of the converged
forward solve and stay frozen. Adjoint velocities and pressures of eliminated rows and cells
are held at zero, like their forward counterparts.

Nonlinear viscosity is differentiated as the converged stress-based constitutive relation
`η = viscosity_τII(stress(η))`; viscosity relaxation is not differentiated.
"""
function solve_VariationalDYREL_adjoint!(
        stokes,
        stokes_ad,
        ρg,
        dyrel,
        flow_bcs,
        phase_ratios,
        ϕ::JustRelax.RockRatio,
        rheology,
        args,
        grid::Geometry{N},
        dt,
        igg;
        maskV,
        maskP,
        air_phase::Integer = 0,
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
    _di = grid._di
    ni = size(stokes.P)
    periodic = periodic_dims(stokes)
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

    igg.me == 0 && @printf("\n######## Running adjoint Stokes solver (variational DYREL) ########\n")

    # Residual norms over the same reduced, boundary-trimmed space as the forward solver, so
    # that ϵ means the same thing. A periodic axis has no boundary row to trim.
    norm_trim(A) = @views A[ntuple(k -> periodic[k] ? (1:size(A, k)) : (2:(size(A, k) - 1)), dim)...]
    maskRi = ntuple(d -> norm_trim(maskV[d]), dim)
    adjoint_Ri = ntuple(d -> norm_trim(adjoint_velocity_residuals[d]), dim)
    # adjoint velocity unknowns on the same reduced space, for the scale of the continuity check
    λV = (stokes_ad.λV.Vx, stokes_ad.λV.Vy)
    λVi = ntuple(d -> @views(λV[d][2:(size(maskV[d], 1) + 1), 2:(size(maskV[d], 2) + 1)]), dim)
    nV = ntuple(d -> max(sum_mpi(maskRi[d]), 1), dim)
    nP = max(sum_mpi(maskP), 1)
    lx = grid.max_li

    # Reuse the forward DYREL parameters, but not its iteration history.
    dyrel.dVxdτ .= 0
    dyrel.dVydτ .= 0
    stokes_ad.viscosity.η .= 0
    stokes_ad.viscosity.ηv .= 0
    constitutive_scratch = linear_viscosity ? nothing : (;
            η_target = copy(stokes.viscosity.η),
            η = similar(stokes.viscosity.η),
            feedback = similar(stokes.viscosity.η),
            feedback_λ = Ref{Any}(nothing),
            τ = map(similar, (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c)),
            τ_v = map(similar, (stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy)),
            τII = similar(stokes.τ.II),
            θ = similar(stokes.ΔPψ),
        )
    if !isnothing(constitutive_scratch)
        compute_constitutive_tangent_amplification!(
            constitutive_scratch, stokes, rheology, phase_ratios, ϕ,
            λ_relaxation, dt, args, viscosity_cutoff; air_phase,
        )
        update_variational_adjoint_dτV_α_β!(dyrel, constitutive_scratch.η)
    end
    # λmin is estimated on the same reduced, boundary-trimmed rows as the forward estimate
    damping_masks = ntuple(dim) do d
        mask = fill!(similar(maskV[d]), false)
        norm_trim(mask) .= maskRi[d]
        mask
    end
    damping = AdjointDamping(dyrel, damping_masks)
    update_steps! = if isnothing(constitutive_scratch)
        () -> update_dτV_α_β!(dyrel)
    else
        # `constitutive_scratch.η` is reused as scratch during the iterations, so keep the
        # tangent amplification it holds now
        amplification = copy(constitutive_scratch.η)
        () -> update_variational_adjoint_dτV_α_β!(dyrel, amplification)
    end

    # Iteration loop
    err_min = Inf
    errV0 = ntuple(_ -> 0.0, dim)
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ

    isnothing(observation) && throw(ArgumentError("an observation region is required for the adjoint solve"))
    observation = observation_mask(stokes_ad, grid, observation)
    check_observation_mask(observation, maskV, maskP)

    converged = false
    for itPH in 1:Int(iterMax_PH)

        initialize_adjoint_iteration!(stokes_ad, ni)

        # Init observation points
        seed_observation!(observation)

        variational_adjoint_residual!(
            stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, _di, ni,
            dt, λ_relaxation, free_surface, air_phase, maskP,
            constitutive_scratch, viscosity_cutoff,
        )

        # Residual check
        errV = ntuple(d -> masked_norm_mpi(maskRi[d], adjoint_Ri[d]) / √(nV[d]), dim)
        # Scale-free criteria, so that ϵ does not depend on the units of the problem. The
        # objective seeds only some velocity components, so the momentum adjoints are measured
        # against the largest initial one. The continuity adjoint is the divergence of λV and,
        # like the forward `RP·lx/Vspan`, is measured against the λV scale over the domain length.
        # (Relative to its own initial value instead, it would have to drop by ϵ, which the
        # Powell–Hestenes iteration only reaches after many passes.)
        errV0 = momentum_residual_reference(errV0, errV)
        λVspan = maximum(map(masked_value_scale, maskV, λVi))
        has_λV_scale = !iszero(λVspan)
        errPt = has_λV_scale ?
            masked_norm_mpi(maskP, stokes_ad.P) / √(nP) * lx / λVspan : zero(λVspan)
        errV_relative = relative_momentum_residual(errV, errV0)
        # λV is exactly zero before the first adjoint update, so no physically meaningful
        # continuity normalization exists yet. Omit that term until λV provides a scale;
        # the momentum residual remains one for a nontrivial objective on the first pass.
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
        # A pressure-only objective has no initial momentum residual: its first outer pass
        # transfers the pressure seed into λP. Do not accept that pre-update state as converged.
        pending_pressure_seed = isone(itPH) && observation.field === :P
        if err < ϵ && !pending_pressure_seed
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

        ϵ_vel = err * rel_drop
        itPT = 0
        while (err > ϵ_vel && itPT ≤ iterMax_DR)
            itPT += 1
            iter += 1

            initialize_adjoint_iteration!(stokes_ad, ni)

            # A pressure seed enters the Schur correction and the λP update below
            seed_observation!(observation)

            variational_adjoint_residual!(
                stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, _di,
                ni, dt, λ_relaxation, free_surface, air_phase, maskP,
                constitutive_scratch, viscosity_cutoff,
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
                ϕ,
                _di.center,
            )
            update_adjoint_damping!(
                damping, dyrel, adjoint_velocity_residuals, iter, nout, ni, update_steps!
            )

            # Residual check
            if iszero(iter % nout)
                errV = ntuple(d -> masked_norm_mpi(maskRi[d], adjoint_Ri[d]) / √(nV[d]), dim)
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

        # stokes_ad.P is masked, so eliminated cells keep λP = 0
        @. stokes_ad.λP += pressure_relaxation * dyrel.γ_eff * stokes_ad.P

        iter > total_iterMax && break
    end
    if !converged
        # Leave the shared DYREL state ready for a subsequent solve, but do not calculate or
        # return sensitivities from an unconverged adjoint state.
        restore_forward_damping!(damping, dyrel)
        error(
            "Variational DYREL adjoint did not converge " *
                "(err=$err, ϵ=$ϵ, iterations=$iter, total_iterMax=$total_iterMax)"
        )
    end

    # sensitivity evaluation
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, ϕ, rheology, _di, ni, λ_relaxation, dt, igg,
        gradients, args; viscosity_cutoff, free_surface, air_phase, constitutive_scratch,
    )

    # Do not carry adjoint damping or iteration history into the next forward solve.
    restore_forward_damping!(damping, dyrel)

    return stokes_ad
end

# One reverse sweep through the masked forward kernels, in the reverse order of the forward
# update: momentum residual, the converged stress/viscosity relation, strain rate and
# continuity residual, boundary conditions.
# Seeds are set by the caller. The adjoint pressure residual of eliminated cells is dropped.
function variational_adjoint_residual!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, _di, ni, dt,
        λ_relaxation, free_surface, air_phase, maskP, constitutive_scratch,
        viscosity_cutoff,
    )
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, ϕ, _di, ni, rheology, phase_ratios, args;
        free_surface_dt = dt * free_surface, air_phase,
    )
    enzyme_compute_stress_DRYEL!(
        stokes, stokes_ad, rheology, phase_ratios, ϕ, λ_relaxation, dt, args,
        viscosity_cutoff, constitutive_scratch; air_phase,
    )
    enzyme_compute_∇V_strain_rate_RP!(stokes, stokes_ad, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, args)
    enzyme_flow_bcs!(stokes, stokes_ad, flow_bcs)
    # Keep the pressure adjoint zero in cells eliminated from the variational system.
    stokes_ad.P .*= maskP
    return nothing
end
