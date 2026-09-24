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

Only `linear_viscosity = true` is supported: the separate τII-viscosity refresh of the
variational solver is not differentiated.
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
        viscosity_relaxation = 1.0e-2,
        linear_viscosity = false,
        kwargs...,
    ) where {N}
    dim = Val(N)
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

    linear_viscosity || throw(
        ArgumentError("the variational DYREL adjoint requires linear_viscosity = true")
    )

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

    # Iteration loop
    err_min = Inf
    errV0 = ntuple(_ -> 1.0, dim)
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ

    isnothing(observation) && throw(ArgumentError("an observation region is required for the adjoint solve"))
    observation = observation_mask(stokes_ad, grid, observation)
    check_observation_mask(observation, maskV, maskP)

    for itPH in 1:Int(iterMax_PH)

        initialize_adjoint_iteration!(stokes_ad, ni)

        # Init observation points
        observation.target[observation.i, observation.j] .= -1.0

        variational_adjoint_residual!(
            stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, _di, ni,
            dt, λ_relaxation_PH, free_surface, air_phase, maskP,
        )

        # Residual check
        errV = ntuple(d -> masked_norm_mpi(maskRi[d], adjoint_Ri[d]) / √(nV[d]), dim)
        # Scale-free criteria, so that ϵ does not depend on the units of the problem. The
        # objective seeds only some velocity components, so the momentum adjoints are measured
        # against the largest initial one. The continuity adjoint is the divergence of λV and,
        # like the forward `RP·lx/Vspan`, is measured against the λV scale over the domain length.
        # (Relative to its own initial value instead, it would have to drop by ϵ, which the
        # Powell–Hestenes iteration only reaches after many passes.)
        if isone(itPH)
            errV0 = ntuple(_ -> maximum(errV) + eps(), dim)
        end
        # not `nonzero_span`: its absolute threshold is far above a dimensional λV (~dx²/η);
        # λV is exactly zero only on the first pass
        λVspan = maximum(map(masked_value_scale, maskV, λVi))
        λVspan = iszero(λVspan) ? one(λVspan) : λVspan
        errPt = masked_norm_mpi(maskP, stokes_ad.P) / √(nP) * lx / λVspan
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
        while (err > ϵ_vel && itPT ≤ iterMax_DR)
            itPT += 1
            iter += 1

            initialize_adjoint_iteration!(stokes_ad, ni)

            # Init observation points
            observation.field !== :P && (observation.target[observation.i, observation.j] .= -1.0)

            variational_adjoint_residual!(
                stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, _di,
                ni, dt, λ_relaxation_DR, free_surface, air_phase, maskP,
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

            # Residual check
            if iszero(iter % nout)
                errV = ntuple(d -> masked_norm_mpi(maskRi[d], adjoint_Ri[d]) / √(nV[d]), dim)
                # same reference as the outer check: relative to the value after the first `nout`
                # iterations, a residual already at round-off by then can never drop by `rel_drop`
                err = maximum(ntuple(d -> errV[d] / errV0[d], dim))
                isnan(err) && igg.me == 0 && error("NaN detected in inner loop")

                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
                end
            end

        end

        # stokes_ad.P is masked, so eliminated cells keep λP = 0
        @. stokes_ad.λP += pressure_relaxation * dyrel.γ_eff * stokes_ad.P

        iter > total_iterMax && break
    end

    # sensitivity evaluation
    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, ϕ, rheology, _di, ni, λ_relaxation_PH, dt, igg,
        gradients, args; viscosity_cutoff, free_surface, air_phase,
    )

    # Do not carry adjoint iteration history into the next forward solve.
    dyrel.dVxdτ .= 0
    dyrel.dVydτ .= 0

    return stokes_ad
end

# One reverse sweep through the masked forward kernels, in the reverse order of the forward
# update: momentum residual, stress, strain rate and continuity residual, boundary conditions.
# Seeds are set by the caller. The adjoint pressure residual of eliminated cells is dropped.
function variational_adjoint_residual!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, _di, ni, dt,
        λ_relaxation, free_surface, air_phase, maskP,
    )
    enzyme_compute_PH_residual_V!(
        stokes, stokes_ad, ρg, ϕ, _di, ni, rheology, phase_ratios, args;
        free_surface_dt = dt * free_surface, air_phase,
    )
    enzyme_compute_stress_DRYEL!(stokes, stokes_ad, rheology, phase_ratios, ϕ, λ_relaxation, dt)
    enzyme_compute_∇V_strain_rate_RP!(stokes, stokes_ad, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, args)
    enzyme_flow_bcs!(stokes, stokes_ad, flow_bcs)
    stokes_ad.P .*= maskP
    return nothing
end

# The objective must only read unknowns of the reduced system: an observed velocity in an
# eliminated (air) row is not a degree of freedom of the forward solve.
function check_observation_mask(observation, maskV, maskP)
    (; field, i, j) = observation
    mask, offset = if field === :Vx
        Array(maskV[1]), 1
    elseif field === :Vy
        Array(maskV[2]), 1
    else
        Array(maskP), 0
    end
    valid(a, b) = checkbounds(Bool, mask, a - offset, b - offset) && mask[a - offset, b - offset]
    all(valid(a, b) for a in i, b in j) || throw(
        ArgumentError("the observation region must lie in the rock part of the domain; some $field nodes are masked out")
    )
    return nothing
end

# Masked counterpart of `update_adjoint_V_damping_DR!`. The Schur-complement correction is the
# transpose of the forward penalty, whose continuity residual carries the rock fraction; rows
# eliminated by `ϕ` are held at zero, as in the forward `compute_DR_residual_update_V!`.
@parallel_indices (i, j) function update_adjoint_V_damping_DR!(
        Vx,
        Vy,
        dVxdτ,
        dVydτ,
        Rx,
        Ry,
        P,
        γ_eff,
        Dx,
        Dy,
        αVx,
        αVy,
        βVx,
        βVy,
        dτVx,
        dτVy,
        ϕ::JustRelax.RockRatio,
        _di_center,
    )
    @inbounds begin
        if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
            if isvalid_vx(ϕ, i + 1, j)
                iE = wrap_next(i, size(γ_eff, 1))
                Rx_ij = Rx[i + 1, j + 1] - (
                    ϕ.center[i, j] * γ_eff[i, j] * P[i, j] -
                        ϕ.center[iE, j] * γ_eff[iE, j] * P[iE, j]
                ) * @dx(_di_center, i)
                Rx[i + 1, j + 1] = Rx_ij
                dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx_ij / Dx[i, j], αVx[i, j], βVx[i, j], dτVx[i, j])
                dVxdτ[i, j] = dVx_new
                Vx[i + 1, j + 1] += ΔVx
            else
                Rx[i + 1, j + 1] = 0.0
                dVxdτ[i, j] = 0.0
                Vx[i + 1, j + 1] = 0.0
            end
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            if isvalid_vy(ϕ, i, j + 1)
                jN = wrap_next(j, size(γ_eff, 2))
                Ry_ij = Ry[i + 1, j + 1] - (
                    ϕ.center[i, j] * γ_eff[i, j] * P[i, j] -
                        ϕ.center[i, jN] * γ_eff[i, jN] * P[i, jN]
                ) * @dy(_di_center, j)
                Ry[i + 1, j + 1] = Ry_ij
                dVy_new, ΔVy = damped_update_V(dVydτ[i, j], Ry_ij / Dy[i, j], αVy[i, j], βVy[i, j], dτVy[i, j])
                dVydτ[i, j] = dVy_new
                Vy[i + 1, j + 1] += ΔVy
            else
                Ry[i + 1, j + 1] = 0.0
                dVydτ[i, j] = 0.0
                Vy[i + 1, j + 1] = 0.0
            end
        end
    end
    return nothing
end
