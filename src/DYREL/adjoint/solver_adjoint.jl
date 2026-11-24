"""
    solve_VariationalDYREL_adjoint!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
        maskV, maskP, air_phase, kwargs...,
    )

Adjoint entry point called by `solve_VariationalDYREL!(...; adjoint=true)` after the forward
iterations converge and before history-dependent state is updated. It works on the reduced
(rock) space of the variational solver: `ϕ`, the velocity masks `maskV` and the pressure mask
`maskP` are those of the converged forward solve and stay frozen. Adjoint velocities and
pressures of eliminated rows and cells are held at zero, like their forward counterparts.

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
        ϵ = nothing,
        penalty_scaling = :local,
        penalty_factor = nothing,
        inner_solver = :bicgstab,
        deflation = 0,
        kwargs...,
    ) where {N}
    factor = adjoint_penalty_factor(penalty_scaling, penalty_factor)
    dim = Val(N)
    # Differentiate the unrelaxed plastic multiplier: at convergence λ equals its trial value,
    # and holding the relaxed history constant would scale the plastic tangent by the relaxation.
    λ_relaxation = one(eltype(stokes.P))
    _di = grid._di
    ni = size(stokes.P)
    lx = grid.max_li
    periodic = periodic_dims(stokes)

    size(stokes_ad.R.Rx) == size(stokes.R.Rx) &&
        size(stokes_ad.R.Ry) == size(stokes.R.Ry) || throw(
        ArgumentError("stokes and stokes_ad must use the same periodic boundary allocation")
    )
    get(args, :P, nothing) === stokes.P ||
        throw(ArgumentError("the adjoint solve requires args.P === stokes.P"))
    isnothing(observation) &&
        throw(ArgumentError("an observation region is required for the adjoint solve"))

    igg.me == 0 && @printf("\n######## Running adjoint Stokes solver (variational DYREL) ########\n")

    ## 1. Setup

    # the unknowns λV and the momentum residuals live on the momentum rows, the interior of the
    # velocity arrays
    momentum_rows(A, R) = @view A[2:(size(R, 1) + 1), 2:(size(R, 2) + 1)]
    R = (stokes.R.Rx, stokes.R.Ry)
    λV = (stokes_ad.λV.Vx, stokes_ad.λV.Vy)
    λVi = map(momentum_rows, λV, R)
    rV = map(momentum_rows, (stokes_ad.V.Vx, stokes_ad.V.Vy), R)

    # the adjoint iterates with its own DYREL state; the forward one is left untouched
    dyrel = adjoint_dyrel!(stokes_ad, dyrel)
    configure_adjoint_penalty!(
        dyrel, stokes, phase_ratios, ϕ, rheology, grid, dt, ρg, free_surface, factor,
    )
    stokes_ad.viscosity.η .= 0
    stokes_ad.viscosity.ηv .= 0
    constitutive_scratch = linear_viscosity ? nothing : implicit_viscosity_scratch(stokes, ϕ)
    isnothing(constitutive_scratch) || check_viscosity_feedback!(
        constitutive_scratch, stokes, rheology, phase_ratios, ϕ, λ_relaxation, dt, args,
        viscosity_cutoff; air_phase,
    )

    observation = observation_mask(stokes_ad, grid, observation)
    check_observation_mask(observation, maskV, maskP)

    ## 2. Adjoint residual

    sweep!() = variational_adjoint_residual!(
        stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, _di, ni, dt,
        λ_relaxation, free_surface, air_phase, maskP, constitutive_scratch, viscosity_cutoff,
    )
    # Adjoint residuals r_V (in `stokes_ad.V`) and r_P (in `stokes_ad.P`) at (λV_at, λP_at).
    # `seed` adds the objective as −∂J/∂y; `schur` adds the penalty term A_pvᵀ γ r_P to r_V, the
    # residual the inner solvers iterate on.
    function residual!(λV_at = λV, λP_at = stokes_ad.λP; seed = true, schur = true)
        reset_adjoint_residuals!(stokes_ad, ni, λV_at, λP_at)
        seed && seed_observation!(observation)
        sweep!()
        schur && @parallel (@idx ni) adjoint_schur_correction!(
            stokes_ad.V.Vx, stokes_ad.V.Vy, stokes_ad.P, dyrel.γ_eff, dyrel.Dx, dyrel.Dy,
            ϕ, _di.center,
        )
        return nothing
    end

    ## 3. Convergence measures

    # the rows of the forward residual norms, so that ϵ means the same thing
    maskRi = map(A -> norm_rows(A, periodic), maskV)
    rVi = map(A -> norm_rows(A, periodic), rV)
    nV = map(mask -> max(sum_mpi(mask), 1), maskRi)
    nP = max(sum_mpi(maskP), 1)
    momentum_norms() = ntuple(d -> masked_norm_mpi(maskRi[d], rVi[d]) / √(nV[d]), dim)

    # Every solve starts from λ = 0: the previous time step's λ did not shorten the solve (and
    # lengthened it on fine grids).
    foreach(A -> fill!(A, 0), (λV..., stokes_ad.λP))
    initialize_adjoint_iteration!(stokes_ad, ni)
    seed_observation!(observation)

    # Scale-free measures, so that ϵ does not depend on the units of the problem. The momentum
    # residual is measured against its largest component at λ = 0, the objective's velocity seed.
    # A pressure objective seeds only the continuity adjoint, so its reference is the momentum
    # residual after the first λP update. A reference that is still zero is replaced by the first
    # nonzero residual.
    momentum_reference = Ref(maximum(momentum_norms()))
    if observation.field === :P
        momentum_reference[] = maximum(
            pressure_objective_reference!(stokes_ad, dyrel, pressure_relaxation) do
                initialize_adjoint_iteration!(stokes_ad, ni)
                seed_observation!(observation)
                sweep!()
                momentum_norms()
            end
        )
    end
    function momentum_error(norms = momentum_norms())
        e = maximum(norms)
        iszero(momentum_reference[]) && (momentum_reference[] = e)
        return iszero(momentum_reference[]) ? zero(e) : e / momentum_reference[]
    end
    # The continuity adjoint is essentially the divergence of λV, so, like the forward
    # `RP·lx/Vspan`, it is measured against the λV scale over the domain length. It has no
    # reference of its own: for a velocity objective it is zero at λ = 0. Before the first
    # velocity update λV = 0 gives no scale, and the check is skipped.
    function continuity_error()
        λV_scale = maximum(map(masked_value_scale, maskV, λVi))
        iszero(λV_scale) && return zero(λV_scale)
        return masked_norm_mpi(maskP, stokes_ad.P) / √(nP) * lx / λV_scale
    end

    ## 4. Inner velocity solver

    inner = adjoint_inner_solver(
        inner_solver, deflation, stokes_ad, dyrel, maskV, maskRi, periodic,
        constitutive_scratch, ϕ, ni,
    )
    inner_problem = (;
        residual!, momentum_error, λVi, R = rV, D = (dyrel.Dx, dyrel.Dy),
        iterMax = iterMax_DR, total_iterMax, nout, verbose = verbose_DR && igg.me == 0,
    )

    ## 5. Powell–Hestenes iteration

    ϵ = something(ϵ, dyrel.ϵ)
    err = 2 * ϵ
    err_min = Inf
    iter = 0
    converged = false
    for itPH in 1:Int(iterMax_PH)
        residual!(; schur = false)
        norms = momentum_norms()
        errP = continuity_error()
        err = max(momentum_error(norms), errP)

        if verbose_PH && igg.me == 0
            errV_msg = join(
                ntuple(d -> @sprintf("R%d=%1.3e %1.3e", d, norms[d], norms[d] / momentum_reference[]), dim),
                ", ",
            )
            @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[%s, Rp=%1.3e] \n", itPH, iter, iter / ni[1], err, errV_msg, errP)
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
        # the iteration budget is checked here, after the full criterion of the last pass
        iter ≥ total_iterMax && break

        # tighten the inner tolerance when the outer error grows
        err > err_min * 1.05 && (rel_drop = max(rel_drop * 0.1, 1.0e-3))
        err_min = min(err_min, err)

        err, iter = inner_solve!(inner, inner_problem, err, err * rel_drop, iter)

        # stokes_ad.P is masked, so eliminated cells keep λP = 0
        @. stokes_ad.λP += pressure_relaxation * dyrel.γ_eff * stokes_ad.P
        after_pressure_update!(inner)
    end
    # do not calculate or return sensitivities from an unconverged adjoint state
    converged || error(
        "Variational DYREL adjoint did not converge " *
            "(err=$err, ϵ=$ϵ, iterations=$iter, total_iterMax=$total_iterMax)"
    )

    ## 6. Sensitivities

    compute_sensitivities!(
        stokes, stokes_ad, ρg, phase_ratios, ϕ, rheology, _di, ni, λ_relaxation, dt, igg,
        gradients, args; viscosity_cutoff, free_surface, air_phase, constitutive_scratch,
    )

    return stokes_ad
end

# The reduced, boundary-trimmed momentum rows of the residual norms, as in the forward solver. A
# periodic axis has no boundary row to trim.
norm_rows(A, periodic) =
    @views A[ntuple(k -> periodic[k] ? (1:size(A, k)) : (2:(size(A, k) - 1)), ndims(A))...]

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
