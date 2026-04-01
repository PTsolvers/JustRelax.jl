## 2D VISCO-ELASTIC STOKES SOLVER
"""
    solve_DYREL!(stokes::JustRelax.StokesArrays, args...; kwargs)

Internal solver implementation for the 2D DYREL method on the CPU.

# Arguments (in the following order)
- `stokes`: `JustRelax.StokesArrays` containing the simulation fields.
- `ρg`: buoyancy forces arrays.
- `dyrel`: DYREL-specific parameters and fields.
- `flow_bcs`: `AbstractFlowBoundaryConditions` defining velocity boundary conditions.
- `phase_ratios`: `JustPIC.PhaseRatios` for material phase tracking.
- `rheology`: Material properties and rheological laws.
- `args`: Tuple of additional arguments needed to update viscosity, stress, and buoyancy forces.
- `di`: Grid spacing tuple `(dx, dy)`.
- `dt`: Time step.
- `igg`: `IGG` object for global grid information (MPI).

# Keyword Arguments
- `viscosity_cutoff`: Limits for viscosity `(min, max)`. Default: `(-Inf, Inf)`.
- `viscosity_relaxation`: Relaxation factor for viscosity updates. Default: `1.0e-2`.
- `λ_relaxation_DR`: Relaxation factor for dynamic relaxation. Default: `1`.
- `λ_relaxation_PH`: Relaxation factor for Powell-Hestenes iterations. Default: `1`.
- `iterMax`: Maximum number of iterations. Default: `50.0e3`.
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
        grid::Geometry{2},
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
        kwargs...,
    )

    # unpack
    (;
        γ_eff,
        Dx,
        Dy,
        λmaxVx,
        λmaxVy,
        dVxdτ,
        dVydτ,
        dτVx,
        dτVy,
        dVx,
        dVy,
        βVx,
        βVy,
        cVx,
        cVy,
        αVx,
        αVy,
        c_fact,
        ηb,
    ) = dyrel

    di = grid.di
    _di = grid._di
    di_center = di.center
    ni = size(stokes.P)

    # errors
    err = 1.0
    iter = 0

    # solver loop
    @copy stokes.P0 stokes.P
    Rx0 = similar(stokes.R.Rx)
    Ry0 = similar(stokes.R.Ry)

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # reset plastic multiplier at the beginning of the time step
    stokes.λ .= 0.0
    stokes.λv .= 0.0

    # Iteration loop
    err_min = Inf
    err = 1.0
    errVx0 = 1.0
    errVy0 = 1.0
    errPt0 = 1.0
    errVx00 = 1.0
    errVy00 = 1.0
    iter = 0
    ϵ = dyrel.ϵ
    err = 2 * ϵ
    err_evo_V = Float64[]
    err_evo_P = Float64[]
    err_evo_it = Float64[]
    itg = 0
    P_num = similar(stokes.P)

    # recompute all the DYREL variables
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    DYREL!(dyrel, stokes, rheology, phase_ratios, grid.di, dt)

    # Powell-Hestenes iterations
    for itPH in 1:1000
        # update buoyancy forces
        update_ρg!(ρg, phase_ratios, rheology, args)

        # compute divergence and deviatoric strain rate in one pass
        @parallel (@idx ni .+ 1) compute_∇V_strain_rate!(
            stokes.∇V,
            @strain(stokes)...,
            @velocity(stokes)...,
            _di.vertex,
            _di.velocity[1],
            _di.velocity[2],
        )
        vertex2center!(stokes.ε.xy_c, stokes.ε.xy)

        # compute deviatoric stress
        compute_stress_DRYEL!(stokes, rheology, phase_ratios, λ_relaxation_PH, dt) # not resetting λ in every PH iteration seems to work better
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
            )
        end

        # compute velocity residuals
        @parallel (@idx ni) compute_PH_residual_V!(
            stokes.R.Rx,
            stokes.R.Ry,
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
            ηb,
            rheology,
            phase_ratios,
            dt,
            args,
        )
        # Residual check
        errVx = norm_mpi(stokes.R.Rx) / √((nx_g() - 2) * (ny_g() - 1))
        errVy = norm_mpi(stokes.R.Ry) / √((nx_g() - 1) * (ny_g() - 2))
        errPt = norm_mpi(stokes.R.RP) / √(nx_g() * ny_g())
        if isone(itPH)
            errVx0 = errVx + eps()
            errVy0 = errVy + eps()
            errPt0 = errPt + eps()
        end
        if itPH == 2
            errPt0 = errPt + eps()
        end
        err = maximum(
            # (min(errVx/errVx0, errVx), min(errVy/errVy0, errVy))
            (min(errVx / errVx0, errVx), min(errVy / errVy0, errVy), min(errPt / errPt0, errPt))
        )

        if verbose_PH && igg.me == 0
            @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[Rx=%1.3e %1.3e, Ry=%1.3e %1.3e, Rp=%1.3e %1.3e] \n", itPH, iter, iter / ni[1], err, errVx, errVx / errVx0, errVy, errVy / errVy0, errPt, errPt / errPt0)
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
            copyto!(Rx0, stokes.R.Rx)
            copyto!(Ry0, stokes.R.Ry)

            # Deviatoric strain rate and divergence
            @parallel (@idx ni .+ 1) compute_∇V_strain_rate!(
                stokes.∇V,
                @strain(stokes)...,
                @velocity(stokes)...,
                _di.vertex,
                _di.velocity[1],
                _di.velocity[2],
            )
            vertex2center!(stokes.ε.xy_c, stokes.ε.xy)

            compute_residual_P!(
                stokes.R.RP,
                stokes.P,
                stokes.P0,
                stokes.∇V,
                stokes.Q, # volumetric source/sink term
                ηb,
                rheology,
                phase_ratios,
                dt,
                args,
            )

            # Deviatoric stress
            compute_stress_DRYEL!(stokes, rheology, phase_ratios, λ_relaxation_DR, dt)
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
                )
            end

            # Residuals
            @. P_num = γ_eff * stokes.R.RP
            @parallel (@idx ni) compute_DR_residual_V!(
                stokes.R.Rx,
                stokes.R.Ry,
                stokes.P,
                P_num,
                stokes.ΔPψ,
                @stress(stokes)...,
                ρg...,
                Dx,
                Dy,
                _di.center,
                _di.vertex,
            )

            # Damping-pong
            @parallel (@idx ni) update_V_damping!((dVxdτ, dVydτ), (stokes.R.Rx, stokes.R.Ry), (αVx, αVy))

            # PT updates
            @parallel (@idx ni .+ 1) update_DR_V!((stokes.V.Vx, stokes.V.Vy), (dVxdτ, dVydτ), (βVx, βVy), (dτVx, dτVy))
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)

            # Residual check
            if iszero(iter % nout)

                errVx = norm_mpi(Dx .* stokes.R.Rx) / √((nx_g() - 2) * (ny_g() - 1))
                errVy = norm_mpi(Dy .* stokes.R.Ry) / √((nx_g() - 1) * (ny_g() - 2))

                if iter == nout
                    errVx00 = errVx
                    errVy00 = errVy
                end

                err = max(
                    errVx / errVx00, errVy / errVy00
                )
                isnan(err) && igg.me == 0 && error("NaN detected in inner loop")

                push!(err_evo_V, errVx / errVx00)
                push!(err_evo_P, errPt / errPt0)
                push!(err_evo_it, iter)

                # @printf("it = %d, iter = %d, ϵ_vel = %1.3e, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e] \n", itPT, iter, ϵ_vel, err, errVx, errVy)
                if verbose_DR && igg.me == 0
                    @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
                end
                @. dVx = dVxdτ * βVx * dτVx
                @. dVy = dVydτ * βVy * dτVy

                λminV = abs(sum_mpi(dVx .* (stokes.R.Rx .- Rx0)) + sum_mpi(dVy .* (stokes.R.Ry .- Ry0))) /
                    (sum_mpi(dVx .^ 2) + sum_mpi(dVy .^ 2))
                @. cVx = 2 * √(λminV) * c_fact
                @. cVy = 2 * √(λminV) * c_fact

                # Optimal pseudo-time steps - can be replaced by AD
                Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, γ_eff, phase_ratios, rheology, grid.di, dt)

                # Select dτ
                update_dτV_α_β!(dyrel)
            end
        end

        # update pressure
        @. stokes.P += γ_eff .* stokes.R.RP

        iter > total_iterMax && break
    end

    # compute vorticity
    @parallel (@idx ni .+ 1) compute_vorticity!(
        stokes.ω.xy, @velocity(stokes)..., _di.velocity[1], _di.velocity[2]
    )

    # Interpolate shear components to cell center arrays
    shear2center!(stokes.ε)
    shear2center!(stokes.ε_pl)
    shear2center!(stokes.Δε)

    # accumulate plastic strain tensor
    accumulate_tensor!(stokes.EII_pl, stokes.ε_pl, dt)

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))
    stokes.τ_o.xx_v .= stokes.τ.xx_v
    stokes.τ_o.yy_v .= stokes.τ.yy_v


    return (; err_evo_it, err_evo_V, err_evo_P)

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

# TODO: will be addressed in following PRs
# ## variational version
# function _solve_DYREL!(
#         stokes::JustRelax.StokesArrays,
#         ρg,
#         dyrel,
#         flow_bcs::AbstractFlowBoundaryConditions,
#         phase_ratios::JustPIC.PhaseRatios,
#         ϕ::JustRelax.RockRatio,
#         rheology,
#         args,
#         di::NTuple{2, T},
#         dt,
#         igg::IGG;
#         viscosity_cutoff = (-Inf, Inf),
#         viscosity_relaxation = 1.0e-2,
#         λ_relaxation = 0.2,
#         iterMax = 50.0e3,
#         iterMin = 1.0e2,
#         free_surface = false,
#         nout = 100,
#         b_width = (4, 4, 0),
#         verbose = true,
#         kwargs...,
#     ) where {T}

#     (;
#         γ_eff,
#         Dx,
#         Dy,
#         λmaxVx,
#         λmaxVy,
#         dVxdτ,
#         dVydτ,
#         dτVx,
#         dτVy,
#         dVx,
#         dVy,
#         βVx,
#         βVy,
#         cVx,
#         cVy,
#         αVx,
#         αVy,
#         c_fact,
#         ηb,
#     ) = dyrel
#     # unpack

#     _di   = inv.(di)
#     ni    = size(stokes.P)
#     γfact = 20

#     # errors
#     err = 1.0
#     iter = 0
#     err_evo1 = Float64[]
#     err_evo2 = Float64[]
#     norm_Rx = Float64[]
#     norm_Ry = Float64[]
#     norm_∇V = Float64[]

#     # solver loop
#     @copy stokes.P0 stokes.P
#     Rx0 = similar(stokes.R.Rx)
#     Ry0 = similar(stokes.R.Ry)

#     for Aij in @tensor_center(stokes.ε_pl)
#         Aij .= 0.0
#     end

#     # compute buoyancy forces and viscosity
#     compute_ρg!(ρg, phase_ratios, rheology, args)
#     compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
#     center2vertex!(stokes.viscosity.ηv, stokes.viscosity.η)

#     DYREL!(dyrel, stokes, rheology, phase_ratios, ϕ, di, dt)

#     rel_drop   = 0.75 #1e-1         # relative drop of velocity residual per PH iteration
#     # Iteration loop
#     errVx0     = 1.0
#     errVy0     = 1.0
#     errPt0     = 1.0
#     errVx00    = 1.0
#     errVy00    = 1.0
#     iter       = 0
#     ϵ          = dyrel.ϵ
#     err        = 2 * ϵ
#     err_evo_V  = Float64[]
#     err_evo_P  = Float64[]
#     err_evo_it = Float64[]
#     itg        = 0

#     # Powell-Hestenes iterations
#     for itPH in 1:250

#         # compute divergence
#         @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)

#         # compute deviatoric strain rate
#         @parallel (@idx ni .+ 1) compute_strain_rate!(
#             @strain(stokes)..., stokes.∇V, @velocity(stokes)..., ϕ, _di...
#         )

#         # compute deviatoric stress
#         compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, 1, dt)
#         update_halo!(stokes.τ.xy)

#         # compute velocity residuals
#         @parallel (@idx ni) compute_PH_residual_V!(
#             stokes.R.Rx,
#             stokes.R.Ry,
#             stokes.P,
#             stokes.ΔPψ,
#             @stress(stokes)...,
#             ρg...,
#             ϕ,
#             _di...,
#         )

#         # compute pressure residual
#         compute_residual_P!(
#             stokes.R.RP,
#             stokes.P,
#             stokes.P0,
#             stokes.∇V,
#             stokes.Q, # volumetric source/sink term
#             ηb,
#             rheology,
#             phase_ratios,
#             dt,
#             args,
#         )

#         # Residual check
#         errVx = norm(stokes.R.Rx)
#         errVy = norm(stokes.R.Ry)
#         errPt = norm(stokes.R.RP)
#         if isone(itPH)
#             errVx0 = errVx
#             errVy0 = errVy
#             errPt0 = errPt
#         end
#         err = maximum(
#             (min(errVx/errVx0, errVx), min(errVy/errVy0, errVy)), min(errPt/errPt0, errPt)
#         )
#         isnan(err) && error("NaN detected in the errors")
#         @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, Rp=%1.3e] \n", itPH, iter, iter/ni[1], err, min(errVx/errVx0, errVx), min(errVy/errVy0, errVy), min(errPt/errPt0, errPt))
#         # @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, Rp=%1.3e] \n", itPH, iter, iter/ni[1], err, errVx, errVy, errPt)
#         err<ϵ && break

#         # Set tolerance of velocity solve proportional to residual
#         ϵ_vel = err * rel_drop
#         itPT  = 0
#         while (err>ϵ_vel && itPT ≤ iterMax)
#             itPT += 1
#             itg  += 1
#             iter += 1

#             # Pseudo-old dudes
#             copyto!(Rx0, stokes.R.Rx)
#             copyto!(Ry0, stokes.R.Ry)

#             # Divergence
#             @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di)

#             compute_residual_P!(
#                 stokes.R.RP,
#                 stokes.P,
#                 stokes.P0,
#                 stokes.∇V,
#                 stokes.Q, # volumetric source/sink term
#                 ηb,
#                 rheology,
#                 phase_ratios,
#                 dt,
#                 args,
#             )

#             # Deviatoric strain rate
#             @parallel (@idx ni .+ 1) compute_strain_rate!(
#                 @strain(stokes)..., stokes.∇V, @velocity(stokes)..., ϕ, _di...
#             )
#             # update viscosity
#             # update_viscosity_τII!(
#             #     stokes,
#             #     phase_ratios,
#             #     args,
#             #     rheology,
#             #     viscosity_cutoff;
#             #     relaxation = viscosity_relaxation,
#             # )
#             # compute_bulk_viscosity_and_penalty!(dyrel, stokes, rheology, phase_ratios, γfact, dt)

#             # Deviatoric stress
#             compute_stress_DRYEL!(stokes, rheology, phase_ratios, ϕ, λ_relaxation, dt)
#             update_halo!(stokes.τ.xy)

#             # Residuals
#             P_num = γ_eff .* stokes.R.RP
#             @parallel (@idx ni) compute_DR_residual_V!(
#                 stokes.R.Rx,
#                 stokes.R.Ry,
#                 stokes.P,
#                 P_num,
#                 stokes.ΔPψ,
#                 @stress(stokes)...,
#                 ρg...,
#                 ϕ,
#                 Dx,
#                 Dy,
#                 _di...,
#             )

#             # Damping-pong
#             @parallel (@idx ni) update_V_damping!( (dVxdτ, dVydτ), (stokes.R.Rx, stokes.R.Ry), (αVx, αVy) )

#             # PT updates
#             @parallel (@idx ni.+1) update_DR_V!( (stokes.V.Vx, stokes.V.Vy), (dVxdτ, dVydτ), (βVx, βVy), (dτVx, dτVy) )
#             flow_bcs!(stokes, flow_bcs)
#             update_halo!(@velocity(stokes)...)

#             # Residual check
#             if iszero(iter % nout)

#                 errVx = norm(Dx .* stokes.R.Rx)
#                 errVy = norm(Dy .* stokes.R.Ry)

#                 if iter == nout
#                     errVx00 = errVx
#                     errVy00 = errVy
#                 end

#                 err = maximum( (errVx / errVx00, errVy / errVy00) )
#                 push!(err_evo_V, errVx/errVx00)
#                 push!(err_evo_P, errPt/errPt0)
#                 push!(err_evo_it, iter)

#                 @. dVx = dVxdτ * βVx * dτVx
#                 @. dVy = dVydτ * βVy * dτVy

#                 @printf("it = %d, iter = %d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e] \n", it, iter, err, errVx, errVy)

#                 λminV  = abs(sum_mpi(dVx .* (stokes.R.Rx .- Rx0)) + sum_mpi(dVy.*(stokes.R.Ry .- Ry0))) /
#                     (sum_mpi(dVx.*dVx) .+ sum_mpi(dVy.*dVy))
#                 @. cVx = cVy = 2 * √(λminV) * c_fact
#                 # cVy .= 2 * √(λminV) * c_fact

#                 # Optimal pseudo-time steps - can be replaced by AD
#                 # Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, γ_eff, phase_ratios, rheology, di, dt)

#                 # Select dτ
#                 update_dτV_α_β!(dyrel)
#             end
#         end

#         # update pressure
#         stokes.P .+= γ_eff.*stokes.R.RP

#         # update buoyancy forces
#         update_ρg!(ρg, phase_ratios, rheology, args)

#         # update viscosity
#         update_viscosity_τII!(
#             stokes,
#             phase_ratios,
#             args,
#             rheology,
#             viscosity_cutoff;
#             relaxation = viscosity_relaxation,
#         )
#         center2vertex!(stokes.viscosity.ηv, stokes.viscosity.η)

#         # if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
#         #     println("Pseudo-transient iterations converged in $iter iterations")
#         # end
#     end

#     # compute vorticity
#     @parallel (@idx ni .+ 1) compute_vorticity!(
#         stokes.ω.xy, @velocity(stokes)..., inv.(di)...
#     )

#     # Interpolate shear components to cell center arrays
#     shear2center!(stokes.ε)
#     shear2center!(stokes.ε_pl)
#     shear2center!(stokes.Δε)

#     # accumulate plastic strain tensor
#     accumulate_tensor!(stokes.EII_pl, stokes.ε_pl, dt)

#     @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
#     @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))
#     stokes.τ_o.xx_v .= stokes.τ.xx_v
#     stokes.τ_o.yy_v .= stokes.τ.yy_v

#     return (; err_evo_it, err_evo_V, err_evo_P)

# end
