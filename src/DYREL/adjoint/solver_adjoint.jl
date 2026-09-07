function observation_mask(stokes_ad, grid, observation)
    (; field, center, half_width) = observation

    target, x, y, offset = if field === :Vx
        stokes_ad.V.Vx, grid.xvi[1], grid.xci[2], (0, 1)
    elseif field === :Vy
        stokes_ad.V.Vy, grid.xci[1], grid.xvi[2], (1, 0)
    elseif field === :P
        stokes_ad.P, grid.xci[1], grid.xci[2], (0, 0)
    else
        throw(ArgumentError("observation field must be :Vx, :Vy, or :P"))
    end

    i = findall(xi -> abs(xi - center[1]) ≤ half_width[1], x) .+ offset[1]
    j = findall(yj -> abs(yj - center[2]) ≤ half_width[2], y) .+ offset[2]
    (isempty(i) || isempty(j)) && throw(ArgumentError("the observation region does not contain any $field nodes"))
    return (; field, target, i, j)
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
        viscosity_cutoff,
        viscosity_relaxation,
        λ_relaxation_DR,
        λ_relaxation_PH,
        iterMax,
        total_iterMax,
        nout,
        rel_drop,
        b_width,
        verbose_PH,
        verbose_DR,
        linear_viscosity,
        free_surface,
        observation,
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

    # Reuse the forward DYREL parameters, but not its iteration history.
    dyrel.dVxdτ .= 0
    dyrel.dVydτ .= 0

    # errors
    err = 1.0
    iter = 0

    residuals0 = fields.R0

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

    nx, ny = ni
    x_pen = @zeros(nx - 1, ny)
    y_pen = @zeros(nx, ny - 1)

    isnothing(observation) && throw(ArgumentError("an observation region is required for the adjoint solve"))
    observation = observation_mask(stokes_ad, grid, observation)

    for itPH in 1:1000

        stokes_ad.R.Rx .= 0.0
        stokes_ad.R.Ry .= 0.0
        stokes_ad.R.RP .= 0.0
        stokes_ad.P .= 0.0
        stokes_ad.V.Vx .= 0.0
        stokes_ad.V.Vy .= 0.0
        stokes_ad.ε.xx .= 0.0
        stokes_ad.ε.yy .= 0.0
        stokes_ad.ε.xy .= 0.0
        stokes_ad.dτ.xx .= 0.0
        stokes_ad.dτ.yy .= 0.0
        stokes_ad.dτ.xy .= 0.0

        # Init seeds for reverse accumulation
        stokes_ad.R.Rx .= stokes_ad.λV.Vx[2:(end - 1), 2:(end - 1)]
        stokes_ad.R.Ry .= stokes_ad.λV.Vy[2:(end - 1), 2:(end - 1)]
        stokes_ad.R.RP .= stokes_ad.λP

        # Init observation points
        observation.target[observation.i, observation.j] .= -1.0

        enzyme_compute_PH_residual_V!(stokes, stokes_ad, ρg, _di, ni)
        enzyme_compute_stress_DRYEL!(stokes, stokes_ad, rheology, phase_ratios, λ_relaxation_PH, dt)
        enzyme_compute_∇V_strain_rate_RP!(stokes, stokes_ad, dyrel, rheology, phase_ratios, _di, ni, dt, args)
        enzyme_flow_bcs!(stokes, stokes_ad, flow_bcs)

        # Residual check
        errV = (
            norm_mpi(@view(stokes_ad.V.Vx[2:(end - 1), 2:(end - 1)])) / √(v_dofs[1]),
            norm_mpi(@view(stokes_ad.V.Vy[2:(end - 1), 2:(end - 1)])) / √(v_dofs[2]),
        )
        errPt = norm_mpi(stokes_ad.P) / √(p_dof)
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

            stokes_ad.R.Rx .= 0.0
            stokes_ad.R.Ry .= 0.0
            stokes_ad.R.RP .= 0.0
            stokes_ad.P .= 0.0
            stokes_ad.V.Vx .= 0.0
            stokes_ad.V.Vy .= 0.0
            stokes_ad.ε.xx .= 0.0
            stokes_ad.ε.yy .= 0.0
            stokes_ad.ε.xy .= 0.0
            stokes_ad.dτ.xx .= 0.0
            stokes_ad.dτ.yy .= 0.0
            stokes_ad.dτ.xy .= 0.0

            # Init seeds for reverse accumulation
            stokes_ad.R.Rx .= stokes_ad.λV.Vx[2:(end - 1), 2:(end - 1)]
            stokes_ad.R.Ry .= stokes_ad.λV.Vy[2:(end - 1), 2:(end - 1)]
            stokes_ad.R.RP .= stokes_ad.λP

            # Init observation points
            observation.field !== :P && (observation.target[observation.i, observation.j] .= -1.0)

            enzyme_compute_PH_residual_V!(stokes, stokes_ad, ρg, _di, ni)
            enzyme_compute_stress_DRYEL!(stokes, stokes_ad, rheology, phase_ratios, λ_relaxation_DR, dt)
            enzyme_compute_∇V_strain_rate_RP!(stokes, stokes_ad, dyrel, rheology, phase_ratios, _di, ni, dt, args)
            enzyme_flow_bcs!(stokes, stokes_ad, flow_bcs)

            # calculate Schur complement contribution
            x_pen .= ((dyrel.γ_eff[1:(end - 1), :] .* stokes_ad.P[1:(end - 1), :]) .- (dyrel.γ_eff[2:end, :] .* stokes_ad.P[2:end, :])) .* _di.center[1]
            y_pen .= ((dyrel.γ_eff[:, 1:(end - 1)] .* stokes_ad.P[:, 1:(end - 1)]) .- (dyrel.γ_eff[:, 2:end] .* stokes_ad.P[:, 2:end])) .* _di.center[2]

            stokes_ad.V.Vx[2:(end - 1), 2:(end - 1)] .-= x_pen
            stokes_ad.V.Vy[2:(end - 1), 2:(end - 1)] .-= y_pen

            if iszero(iter % nout)
                errV = (
                    norm_mpi(@view(stokes_ad.V.Vx[2:(end - 1), 2:(end - 1)])) / √(v_dofs[1]),
                    norm_mpi(@view(stokes_ad.V.Vy[2:(end - 1), 2:(end - 1)])) / √(v_dofs[2]),
                )
            end

            # preconditioning
            stokes_ad.V.Vx[2:(end - 1), 2:(end - 1)] ./= dyrel.Dx
            stokes_ad.V.Vy[2:(end - 1), 2:(end - 1)] ./= dyrel.Dy

            @parallel (@idx ni) update_V_damping_DR_V!(
                (stokes_ad.λV.Vx, stokes_ad.λV.Vy),
                (dyrel.dVxdτ, dyrel.dVydτ),
                (@view(stokes_ad.V.Vx[2:(end - 1), 2:(end - 1)]), @view(stokes_ad.V.Vy[2:(end - 1), 2:(end - 1)])),
                (dyrel.αVx, dyrel.αVy),
                (dyrel.βVx, dyrel.βVy),
                (dyrel.dτVx, dyrel.dτVy),
            )

            # Residual check
            if iszero(iter % nout)
                if iter == nout
                    errV_scale = maximum(errV) + eps()
                    errV00 = ntuple(_ -> errV_scale, dim)
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
                # λminV = compute_λminV!(fields, residuals, residuals0, ni, dim)
                # @parallel (@idx ni) update_cV!(fields.cV, 2 * √(λminV) * dyrel.c_fact)

                # # Optimal pseudo-time steps - can be replaced by AD
                # Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, stokes.viscosity.ηv, dyrel.γ_eff, phase_ratios, rheology, grid.di, dt)
                # free_surface && apply_free_surface_diagonal!(fields.D[2], fields.λmaxV[2], ρg[end], grid.di.center, dt)

                # # Select dτ
                # update_dτV_α_β!(dyrel)
            end

        end

        @. stokes_ad.λP += dyrel.γ_eff * stokes_ad.P

        iter > total_iterMax && break
    end

    # Do not carry adjoint iteration history into the next forward solve.
    dyrel.dVxdτ .= 0
    dyrel.dVydτ .= 0

    return stokes_ad
end
