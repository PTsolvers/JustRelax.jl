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
        verbose_PH = true,
        verbose_DR = true,
        linear_viscosity = false,
        free_surface = false,
        strict_convergence = false,
        penalty_viscosity::Symbol = :local,
        penalty_floor_fraction = 0.1,
        penalty_ratio = 100,
        inner_tolerance_floor = 1,
        adaptive_inner_tolerance = false,
        momentum_restart_every::Integer = 0,
        residual_check_every::Integer = nout,
        stagnation_window::Integer = 0,
        inner_solver::Symbol = :dr,
        gcr_depth::Integer = 20,
        viscosity_relaxation_PH = viscosity_relaxation,
        kwargs...,
    ) where {N}

    check_periodic_bcs(stokes, flow_bcs, igg, grid.di.center)
    residual_check_every > 0 || throw(ArgumentError("residual_check_every must be positive, got $residual_check_every"))
    inner_solver in (:dr, :pcg, :gcr) || throw(ArgumentError("inner_solver must be :dr, :pcg or :gcr, got :$inner_solver"))
    penalty_viscosity in (:local, :mean, :geomean, :floored, :capped) || throw(ArgumentError("penalty_viscosity must be :local, :mean, :geomean, :floored or :capped, got :$penalty_viscosity"))

    dim = Val(N)
    _di = grid._di
    lx = grid.max_li
    ni = size(stokes.P)
    periodic = periodic_dims(stokes)

    residuals = @residuals(stokes.R)
    fields = dyrel_fields(dyrel, dim)

    # Masks: only count residuals over the valid (rock) part of the domain. `similar` keeps the
    # element type a `Bool`, not a `Bit`: the kernels below write single entries from concurrent
    # threads, and `BitArray` `setindex!` is a non-atomic read-modify-write of a whole 64-bit
    # chunk, so neighbouring columns would race.
    # Shaped like the momentum residual, so a periodic direction carries the seam row here too.
    maskV = (
        similar(ϕ.Vx, Bool, momentum_rows(ni, periodic, 1)),
        similar(ϕ.Vy, Bool, momentum_rows(ni, periodic, 2)),
    )
    maskP = similar(ϕ.center, Bool)
    @parallel (@idx ni) update_valid_c_mask!(maskP, ϕ)
    @parallel (@idx ni) update_valid_v_masks!(maskV..., ϕ)
    # velocity interiors, which is what maskV is shaped like; views alias the parent, so these
    # stay current for the whole solve. Momentum row `i` of direction `d` drives `V[d][i + 1]`
    # along `d`, so that axis runs to `1 + size(maskV[d], d)` -- one further when `d` is periodic
    # and the seam face is an unknown. The transverse axes are always the ghosted interior.
    Vi = ntuple(
        d -> @views(
            @velocity(stokes)[d][
                ntuple(k -> k == d ? (2:(1 + size(maskV[d], k))) : (2:(size(@velocity(stokes)[d], k) - 1)), dim)...,
            ]
        ), dim
    )
    # Momentum-residual norms run over the interior only, so that a boundary-condition row cannot
    # set the residual scale; the continuity residual is not trimmed. Trimming mask, residual and
    # preconditioner diagonal identically keeps them index-aligned. A periodic axis has no
    # boundary row to exclude -- every row of it, the seam included, is a genuine unknown -- so it
    # is left whole.
    norm_trim(A) = @views A[ntuple(k -> periodic[k] ? (1:size(A, k)) : (2:(size(A, k) - 1)), dim)...]
    maskRi = ntuple(d -> norm_trim(maskV[d]), dim)
    Ri = ntuple(d -> norm_trim(residuals[d]), dim)
    R0i = ntuple(d -> norm_trim(fields.R0[d]), dim)
    dVi = ntuple(d -> norm_trim(fields.dV[d]), dim)
    Di = ntuple(d -> norm_trim(fields.D[d]), dim)
    dVdτi = ntuple(d -> norm_trim(fields.dVdτ[d]), dim)
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
        stokes.P, stokes.P0, stokes.ΔPψ, stokes.λ, @velocity(stokes)..., ϕ, maskV...
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
    # Lower bound of the velocity-solve tolerance, as a multiple of `ϵ`. With
    # `adaptive_inner_tolerance`, every pass whose momentum error starts between `ϵ` and `2ϵ`
    # tightens it by 0.7, down to 0.05, so a solve that hovers just above `ϵ` is resolved instead
    # of stopping at the first check of each pass.
    tol_floor = float(inner_tolerance_floor)

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

    # Work arrays of the Picard–PCG inner solve (`inner_solver = :pcg`): secant stiffness at
    # centers and vertices, the stress and strain rate it linearizes about, and the CG vectors.
    if inner_solver in (:pcg, :gcr)
        k_c, k_v = similar(stokes.P), similar(stokes.viscosity.ηv)
        τ0 = (similar(stokes.τ.xx), similar(stokes.τ.yy), similar(stokes.τ.xy))
        ε0 = (similar(stokes.ε.xx), similar(stokes.ε.yy), similar(stokes.ε.xy))
        cg_r, cg_z, cg_p, cg_Ap, cg_R = ntuple(_ -> map(similar, residuals), 5)
        cg_ri = map(norm_trim, cg_r)
        mdot(a, b) = sum(ntuple(d -> sum_mpi((m, x, y) -> m ? x * y : zero(x * y), maskV[d], a[d], b[d]), dim))
        # GCR keeps `gcr_depth` search directions and their images; residuals are compared in the
        # norm weighted by the inverse preconditioner diagonal, so rows of very different stiffness
        # count alike.
        wdot(a, b) = sum(ntuple(d -> sum_mpi((m, x, y, D) -> m ? x * y / D : zero(x * y), maskV[d], a[d], b[d], fields.D[d]), dim))
        gcr_Z = [map(similar, residuals) for _ in 1:(inner_solver === :gcr ? gcr_depth : 0)]
        gcr_Q = [map(similar, residuals) for _ in 1:(inner_solver === :gcr ? gcr_depth : 0)]
        gcr_QQ = zeros(inner_solver === :gcr ? gcr_depth : 0)
    end

    # recompute all the DYREL variables
    compute_viscosity!(stokes, phase_ratios, ϕ, args, rheology, viscosity_cutoff; air_phase = air_phase)
    compute_ρg!(ρg[end], phase_ratios, rheology, args; air_phase)
    DYREL!(dyrel, stokes, rheology, phase_ratios, ϕ, grid.di, dt, iszero(free_surface) ? nothing : ρg[end])
    # `penalty_viscosity = :mean` / `:geomean` sizes `γ_eff` from one viscosity for the whole
    # domain, the arithmetic or geometric mean of the finite rock viscosities at entry, instead of
    # the local `η`; `:floored` keeps the local `η` but bounds it below by
    # `penalty_floor_fraction` times the arithmetic mean; `:capped` uses the arithmetic mean but at
    # most `penalty_ratio` times the cell's effective viscosity `min(η, η_vep)`, so that the penalty
    # in a weak (yielding) cell exceeds its own viscosity by a bounded factor. Inside the velocity iterations
    # `γ_eff·RP` is the only resistance to volume change in yielding cells, where the local `η`
    # can be orders of magnitude below the rest.
    if penalty_viscosity !== :local
        η = stokes.viscosity.η
        η_rock = η[maskP .& .!isinf.(η)]
        η_pen = if penalty_viscosity === :mean
            fill!(similar(η), mean(η_rock))
        elseif penalty_viscosity === :geomean
            fill!(similar(η), exp(mean(log.(η_rock))))
        elseif penalty_viscosity === :capped
            η_vep = stokes.viscosity.η_vep
            min.(mean(η_rock), penalty_ratio .* ifelse.(η_vep .> 0, min.(η, η_vep), η))
        else
            max.(η, penalty_floor_fraction * mean(η_rock))
        end
        compute_bulk_viscosity_and_penalty!(dyrel, η_pen, rheology, phase_ratios, ϕ, dyrel.γfact, dt)
        Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., η, dyrel.γ_eff, phase_ratios, ϕ, rheology, grid.di, dt, iszero(free_surface) ? nothing : ρg[end])
        update_dτV_α_β!(dyrel)
    end

    # Powell-Hestenes iterations
    for itPH in 1:Int(iterMax_PH)
        # update buoyancy forces
        update_ρg!(ρg, phase_ratios, rheology, args; air_phase)

        # compute divergence, deviatoric strain rate and pressure residual in one pass (masked)
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, true; args...)

        # deviatoric stress, then a separate τII-viscosity refresh. The stress kernel derives the
        # vertex viscosity as harm_clamped(η) — the same convention as the APT variational stress
        # kernel; a stored ηv would disagree at the free-surface interface and under-move it.
        compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, λ_relaxation_PH, dt; Pf = fluid_pressure(args, stokes.P))
        if !linear_viscosity
            update_viscosity_τII!(stokes, phase_ratios, ϕ, args, rheology, viscosity_cutoff; relaxation = viscosity_relaxation_PH, air_phase = air_phase)
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
        # which is tuned against the relative form. `strict_convergence = true` drops the absolute
        # form and requires the relative continuity residual itself to fall below `ϵ`.
        err_converged = max(maximum(errV), strict_convergence ? errPt : min(errPt, RP_rms * dt))

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
        # `inner_tolerance_floor` scales the lower bound: at 1, a pass that starts with `errV` just
        # above `ϵ` ends its velocity solve at the first check, and the outer loop can cycle there.
        if adaptive_inner_tolerance && ϵ < maximum(errV) < 2ϵ
            tol_floor = max(0.7 * tol_floor, 0.05)
        end
        ϵ_vel = max(maximum(errV) * rel_drop, tol_floor * ϵ)
        err_vel = Inf
        itPT = 0
        # Initialize dτ for the FSSA-stabilized operator (mirrors solver.jl). The in-loop
        # dτ refresh only fires every `nout` iterations; without this the first window of
        # velocity updates would drive the free-surface-stabilization residual term against
        # a dτ tuned for the plain viscous operator and diverge.
        if !iszero(free_surface)
            Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, dyrel.γ_eff, phase_ratios, ϕ, rheology, grid.di, dt, ρg[end])
            update_dτV_α_β!(dyrel)
        end
        if inner_solver in (:pcg, :gcr)
            # Picard–PCG: the plastic state is frozen at the start of the pass. With the secant
            # stiffness `k = τII / 2εII_eff` and the pass-start stress and strain rate (τ₀, ε₀), the
            # stress is τ = τ₀ + 2k(ε − ε₀) and the momentum residual is affine in V, equal to the full
            # nonlinear residual at the start of the pass. CG with the diagonal `D` as preconditioner
            # solves it; `A p = R(V) − R(V + p)` costs one residual evaluation per iteration.
            @parallel (@idx ni .+ 1) secant_stiffness!(
                k_c, k_v,
                stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c, stokes.τ.xx_v, stokes.τ.yy_v, stokes.τ.xy,
                stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c, stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy,
                stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, stokes.viscosity.η,
                ϕ, rheology, phase_ratios.center, phase_ratios.vertex, dt, periodic,
            )
            foreach(copyto!, τ0, (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy))
            foreach(copyto!, ε0, (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy))
            function linear_residual!(R)
                compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, true; args...)
                @parallel (@idx ni .+ 1) secant_stress!(
                    stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, τ0..., stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, ε0..., k_c, k_v, ϕ
                )
                update_halo!(stokes.τ.xy)
                @. θc = dyrel.γ_eff * stokes.R.RP + stokes.ΔPψ
                @parallel (@idx ni) compute_PH_residual_V!(
                    R..., @velocity(stokes)..., stokes.P, θc, stokes.τ.xx, stokes.τ.yy, stokes.τ.xy,
                    ρg..., ϕ, _di.center, _di.vertex, dt * free_surface,
                )
                return nothing
            end
            function move_V!(p, α)
                foreach((v, q) -> v .+= α .* q, Vi, p)
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
                return nothing
            end
            precondition!(z, r) = foreach((zd, rd, md, Dd) -> (@. zd = ifelse(md, rd / Dd, zero(rd))), z, r, maskV, fields.D)

            linear_residual!(cg_r)
            if inner_solver === :gcr
                # GCR(m) on the affine Picard system, restarted every `gcr_depth` directions: each new
                # direction z = D⁻¹r is made W-orthogonal (W = D⁻¹) in image space to the stored ones,
                # and the step minimizes the W-norm of the residual, which therefore never grows.
                nstored = 0
                while itPT < iterMax_DR
                    itPT += 1
                    itg += 1
                    iter += 1
                    precondition!(cg_z, cg_r)
                    move_V!(cg_z, 1)
                    linear_residual!(cg_R)
                    move_V!(cg_z, -1)
                    foreach((q, r, R) -> (@. q = r - R), cg_Ap, cg_r, cg_R)
                    for j in 1:nstored
                        βj = wdot(cg_Ap, gcr_Q[j]) / gcr_QQ[j]
                        foreach((z, Zj) -> (@. z -= βj * Zj), cg_z, gcr_Z[j])
                        foreach((q, Qj) -> (@. q -= βj * Qj), cg_Ap, gcr_Q[j])
                    end
                    qq = wdot(cg_Ap, cg_Ap)
                    qq > 0 || break
                    α = wdot(cg_r, cg_Ap) / qq
                    move_V!(cg_z, α)
                    foreach((r, q) -> (@. r -= α * q), cg_r, cg_Ap)
                    if nstored == gcr_depth
                        nstored = 0
                    end
                    nstored += 1
                    foreach(copyto!, gcr_Z[nstored], cg_z)
                    foreach(copyto!, gcr_Q[nstored], cg_Ap)
                    gcr_QQ[nstored] = qq
                    errV = ntuple(d -> masked_norm_mpi(maskRi[d], cg_ri[d]) / Pspan * lx / √(nV[d]), dim)
                    err_vel = maximum(errV)
                    isnan(err_vel) && igg.me == 0 && error("NaN detected in GCR inner loop")
                    if verbose_DR && igg.me == 0 && iszero(itPT % nout)
                        @printf("gcr it = %d, iter = %d, err = %1.3e \n", itPT, iter, err_vel)
                    end
                    err_vel ≤ ϵ_vel && break
                end
                rz = 0.0
            end
            precondition!(cg_z, cg_r)
            foreach(copyto!, cg_p, cg_z)
            rz = mdot(cg_r, cg_z)
            while inner_solver === :pcg && itPT < iterMax_DR
                itPT += 1
                itg += 1
                iter += 1
                move_V!(cg_p, 1)
                linear_residual!(cg_R)
                move_V!(cg_p, -1)
                foreach((Ap, r, R) -> (@. Ap = r - R), cg_Ap, cg_r, cg_R)
                pAp = mdot(cg_p, cg_Ap)
                if !(pAp > 0)
                    verbose_DR && igg.me == 0 && @printf("PCG: p⋅Ap = %1.3e at it = %d, operator not positive definite along p\n", pAp, itPT)
                    break
                end
                α = rz / pAp
                move_V!(cg_p, α)
                foreach((r, Ap) -> (@. r -= α * Ap), cg_r, cg_Ap)
                errV = ntuple(d -> masked_norm_mpi(maskRi[d], cg_ri[d]) / Pspan * lx / √(nV[d]), dim)
                err_vel = maximum(errV)
                isnan(err_vel) && igg.me == 0 && error("NaN detected in PCG inner loop")
                if verbose_DR && igg.me == 0 && iszero(itPT % nout)
                    @printf("pcg it = %d, iter = %d, err = %1.3e \n", itPT, iter, err_vel)
                end
                err_vel ≤ ϵ_vel && break
                precondition!(cg_z, cg_r)
                rz_new = mdot(cg_r, cg_z)
                β = rz_new / rz
                rz = rz_new
                foreach((p, z) -> (@. p = z + β * p), cg_p, cg_z)
            end
        else
            # `stagnation_window = w` ends the pass when the velocity error has dropped by less than 1%
            # over the last `w` iterations of it.
            err_ref, it_ref = Inf, 0
            while (err_vel > ϵ_vel && itPT ≤ iterMax_DR)
                itPT += 1
                itg += 1
                iter += 1

                # Pseudo-old dudes (only needed by compute_λminV! on residual-check iterations)
                iszero(iter % nout) && foreach(copyto!, residuals0, residuals)

                # compute divergence, deviatoric strain rate and pressure residual in one pass (masked)
                compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, true; args...)

                # deviatoric stress (vertex viscosity via harm_clamped(η)) + separate τII-viscosity
                # refresh, then assemble the small pressure correction θc = γ_eff·RP + ΔPψ
                compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, λ_relaxation_DR, dt; Pf = fluid_pressure(args, stokes.P))
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

                # Adaptive restart: when the accumulated update `dVdτ` points against the (preconditioned)
                # residual, the momentum term is carrying the iterate uphill, so drop it. Never on a
                # damping-refresh iteration: the λmin estimate below is a Rayleigh quotient of this
                # iteration's update and degenerates when that update is zeroed.
                if momentum_restart_every > 0 && iszero(itPT % momentum_restart_every) && !iszero(iter % nout)
                    uphill = sum(ntuple(d -> sum_mpi((m, r, v) -> m ? r * v : zero(r * v), maskRi[d], Ri[d], dVdτi[d]), dim))
                    uphill < 0 && foreach(A -> fill!(A, zero(eltype(A))), fields.dVdτ)
                end

                # Residual check every `residual_check_every` iterations; the damping and pseudo-time steps
                # are refreshed every `nout`, which needs the residual history saved at the top of the loop.
                refresh_now = iszero(iter % nout)
                if refresh_now || iszero(iter % residual_check_every)

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
                    if stagnation_window > 0 && itPT - it_ref ≥ stagnation_window
                        err_vel > 0.99 * err_ref && break
                        err_ref, it_ref = err_vel, itPT
                    end
                end

                if refresh_now
                    # Estimate the smallest eigenvalue on exactly the same reduced, boundary-trimmed
                    # velocity space used by the residual norm. Eliminated cut-cell rows otherwise
                    # contaminate the Rayleigh quotient even though they are not part of the solve.
                    @parallel (@idx ni) compute_dV!(fields.dV, fields.dVdτ, fields.βV, fields.dτV)
                    λminV = masked_λminV(dVi, Ri, R0i, maskRi)
                    @parallel (@idx ni) update_cV!(fields.cV, 2 * √(λminV) * dyrel.c_fact)

                    # Optimal pseudo-time steps - can be replaced by AD
                    Gershgorin_Stokes2D_SchurComplement!(fields.D..., fields.λmaxV..., stokes.viscosity.η, dyrel.γ_eff, phase_ratios, ϕ, rheology, grid.di, dt, iszero(free_surface) ? nothing : ρg[end])

                    # Select dτ
                    update_dτV_α_β!(dyrel)
                end
            end        end
        if itPT > iterMax_DR && igg.me == 0
            @warn "DYREL velocity solve exhausted iterMax_DR before reaching ϵ_vel" itPH iter itPT iterMax_DR err_vel ϵ_vel maxlog = 10
        end

        # update pressure — refresh RP from the final velocity first (do_strain_rate = false leaves
        # the strain-rate arrays untouched), otherwise the pressure correction lags one velocity update
        compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, false; args...)
        @. stokes.P += pressure_relaxation * dyrel.γ_eff * stokes.R.RP
        # The uniform volumetric mode is fitted to what the local update above left behind, so RP
        # has to be refreshed in between; reusing the pre-update residual corrects the mean twice.
        # Both the refresh and the relaxation are skipped where the mode carries no correction.
        compliance = volumetric_compliance_total(dyrel.ηb, ϕ, maskP)
        if !iszero(compliance)
            compute_∇V_strain_rate_RP!(stokes, dyrel, rheology, phase_ratios, ϕ, _di, ni, dt, false; args...)
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

## Picard–PCG kernels

# Secant stiffness `k = τII / 2εII_eff` of the stress the last stress update produced, at cell
# centers and vertices, with `εII_eff` the invariant of `ε + τ_o / 2Gdt`. The return mapping is
# radial in deviatoric space, so `τ = 2k·ε_eff` holds componentwise. Where the effective strain rate
# vanishes the viscoelastic viscosity is used instead.
@parallel_indices (I...) function secant_stiffness!(
        k_c, k_v, τxx, τyy, τxy_c, τxx_v, τyy_v, τxy_v, τoxx, τoyy, τoxy_c, τoxx_v, τoyy_v, τoxy_v,
        εxx, εyy, εxy, η, ϕ::JustRelax.RockRatio, rheology, phase_center, phase_vertex, dt, periodic,
    )
    ni = size(phase_center)
    @inline secant(τII, εII, ηve) = εII > 0 ? τII / (2 * εII) : ηve

    if isvalid_v(ϕ, I...)
        Ic = clamped_indices(ni, periodic, I...)
        G = fn_ratio(get_shear_modulus, rheology, phase_vertex[I...])
        ηv = harm_clamped(η, Ic...)
        _2Gdt = inv(2 * G * dt)
        ε_eff = (
            av_clamped(εxx, Ic...) + τoxx_v[I...] * _2Gdt,
            av_clamped(εyy, Ic...) + τoyy_v[I...] * _2Gdt,
            εxy[I...] + τoxy_v[I...] * _2Gdt,
        )
        τII = second_invariant(τxx_v[I...], τyy_v[I...], τxy_v[I...])
        k_v[I...] = secant(τII, second_invariant(ε_eff...), ηv / (1 + ηv * inv(G * dt)))
    else
        k_v[I...] = zero(eltype(k_v))
    end

    if all(I .≤ ni)
        if isvalid_c(ϕ, I...)
            G = fn_ratio(get_shear_modulus, rheology, phase_center[I...])
            ηc = η[I...]
            _2Gdt = inv(2 * G * dt)
            εxy_c = sum(_gather(εxy, I...)) / 4
            ε_eff = (εxx[I...] + τoxx[I...] * _2Gdt, εyy[I...] + τoyy[I...] * _2Gdt, εxy_c + τoxy_c[I...] * _2Gdt)
            τII = second_invariant(τxx[I...], τyy[I...], τxy_c[I...])
            k_c[I...] = secant(τII, second_invariant(ε_eff...), ηc / (1 + ηc * inv(G * dt)))
        else
            k_c[I...] = zero(eltype(k_c))
        end
    end
    return nothing
end

# Stress linearized about the start of the pass: τ = τ₀ + 2k(ε − ε₀). The momentum residual reads the
# normal components at centers and the shear component at vertices.
@parallel_indices (I...) function secant_stress!(
        τxx, τyy, τxy, τ0xx, τ0yy, τ0xy, εxx, εyy, εxy, ε0xx, ε0yy, ε0xy, k_c, k_v, ϕ::JustRelax.RockRatio
    )
    if isvalid_v(ϕ, I...)
        τxy[I...] = τ0xy[I...] + 2 * k_v[I...] * (εxy[I...] - ε0xy[I...])
    end
    if all(I .≤ size(τxx)) && isvalid_c(ϕ, I...)
        τxx[I...] = τ0xx[I...] + 2 * k_c[I...] * (εxx[I...] - ε0xx[I...])
        τyy[I...] = τ0yy[I...] + 2 * k_c[I...] * (εyy[I...] - ε0yy[I...])
    end
    return nothing
end
