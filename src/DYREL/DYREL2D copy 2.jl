## 2D VISCO-ELASTIC STOKES SOLVER

# backend trait
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
        di::NTuple{2, T},
        dt,
        igg::IGG;
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        λ_relaxation_DR = 1,
        λ_relaxation_PH = 1,
        iterMax = 50.0e3,
        iterMin = 1.0e2,
        free_surface = false,
        nout = 100,
        rel_drop = 1e-2,
        b_width = (4, 4, 0),
        verbose = true,
        kwargs...,
    ) where {T}

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
    # unpack

    _di   = inv.(di)
    ni    = size(stokes.P)
    γfact = 20

    # errors
    err      = 1.0
    iter     = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx  = Float64[]
    norm_Ry  = Float64[]
    norm_∇V  = Float64[]

    # solver loop
    @copy stokes.P0 stokes.P
    Rx0 = similar(stokes.R.Rx)
    Ry0 = similar(stokes.R.Ry)

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # reset plastic multiplier at the beginning of the time step
    stokes.λ  .= 0.0
    stokes.λv .= 0.0

    ηc_true = similar(stokes.viscosity.η)

    # # compute buoyancy forces and viscosity
    # compute_ρg!(ρg, phase_ratios, rheology, args)
    # # viscosity guess based on strain rate
    # compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    # center2vertex!(stokes.viscosity.ηv, stokes.viscosity.η)

    # rel_drop   = 1e-2 # relative drop of velocity residual per PH iteration
    # Iteration loop
    errVx0     = 1.0
    errVy0     = 1.0
    errPt0     = 1.0 
    errVx00    = 1.0
    errVy00    = 1.0 
    iter       = 0
    ϵ          = dyrel.ϵ
    err        = 2 * ϵ
    err_evo_V  = Float64[]
    err_evo_P  = Float64[]
    err_evo_it = Float64[]
    itg        = 0
    P_num      = similar(stokes.P)
    # Powell-Hestenes iterations
    for itPH in 1:250

        # # reset plastic multiplier at the beginning of the time step
        # stokes.λ  .= 0.0
        # stokes.λv .= 0.0
        
        # # update buoyancy forces
        # update_ρg!(ρg, phase_ratios, rheology, args)

        # compute divergence
        @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)
        stokes.∇V    .= (stokes.V.Vx[2:end,2:end-1] .- stokes.V.Vx[1:end-1,2:end-1])./di[1] .+ (stokes.V.Vy[2:end-1,2:end] .- stokes.V.Vy[2:end-1,1:end-1])./di[2]

        # compute deviatoric strain rate
        @parallel (@idx ni .+ 1) compute_strain_rate!(
            @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
        )

        # Deviatoric strain rate
        # stokes.ε.xx   .= (stokes.V.Vx[2:end,2:end-1] .- stokes.V.Vx[1:end-1,2:end-1])./di[1] .- 1.0/3.0.*stokes.∇V
        # stokes.ε.yy   .= (stokes.V.Vy[2:end-1,2:end] .- stokes.V.Vy[2:end-1,1:end-1])./di[2] .- 1.0/3.0.*stokes.∇V
        # stokes.ε.xy   .= 0.5.*((stokes.V.Vx[:,2:end] .- stokes.V.Vx[:,1:end-1])./di[2] .+ (stokes.V.Vy[2:end,:] .- stokes.V.Vy[1:end-1,:])./di[1])
        vertex2center!(stokes.ε.xy_c, stokes.ε.xy)

        # compute deviatoric stress
        # compute_stress_DRYEL!(stokes, rheology, phase_ratios, λ_relaxation_PH, dt) # not resetting λ in every PH iteration seems to work better
        # compute_stress_DRYEL!(stokes, rheology, phase_ratios, 1, dt) # λ_relaxation = 1 to reset λ
        # update_halo!(stokes.τ.xy)

        # update_viscosity_τII!(
        #     stokes,
        #     phase_ratios,
        #     args,
        #     rheology,
        #     viscosity_cutoff;
        #     relaxation = viscosity_relaxation,
        # )

        stokes.τ.xx   .= 2.0.*stokes.viscosity.η  .* stokes.ε.xx 
        stokes.τ.yy   .= 2.0.*stokes.viscosity.η  .* stokes.ε.yy 
        stokes.τ.xy   .= 2.0.*stokes.viscosity.ηv .* stokes.ε.xy 
        stokes.τ.xy_c .= 2.0.*stokes.viscosity.η  .* stokes.ε.xy_c
        stokes.τ.II   .= sqrt.(0.5.*(stokes.τ.xx.^2  .+ stokes.τ.yy.^2  .+ (.-stokes.τ.xx.-stokes.τ.yy).^2)   .+ stokes.τ.xy_c.^2 )

        ηc_true .= @. 2^(3-1) * args.ηc0^3 * stokes.τ.II^(1 - 3)
        stokes.viscosity.η .= exp.(viscosity_relaxation*log.(ηc_true) .+ (1-viscosity_relaxation).*log.(stokes.viscosity.η))
        center2vertex_harm!(stokes.viscosity.ηv, stokes.viscosity.η)

        # compute velocity residuals
        @parallel (@idx ni) compute_PH_residual_V!(
            stokes.R.Rx,
            stokes.R.Ry,
            stokes.P,
            stokes.ΔPψ,
            @stress(stokes)...,
            ρg...,
            _di...,
        )
        
        # compute pressure residual
        # compute_residual_P!(
        #     stokes.R.RP,
        #     stokes.P,
        #     stokes.P0,
        #     stokes.∇V,
        #     stokes.Q, # volumetric source/sink term
        #     ηb,
        #     rheology,
        #     phase_ratios,
        #     dt,
        #     args,
        # )

        @. stokes.R.RP = -stokes.∇V - (stokes.P.-stokes.P0) / dyrel.ηb 

        # Residual check
        errVx = norm(stokes.R.Rx) / √(length(stokes.R.Rx))
        errVy = norm(stokes.R.Ry) / √(length(stokes.R.Ry))
        errPt = norm(stokes.R.RP) / √(length(stokes.R.RP))
        if isone(itPH)
            errVx0 = errVx
            errVy0 = errVy
            errPt0 = errPt
        end
        err = maximum(
            # (min(errVx/errVx0, errVx), min(errVy/errVy0, errVy))
            (min(errVx/errVx0, errVx), min(errVy/errVy0, errVy), min(errPt/(errPt0 + eps()), errPt))
        )
        isnan(err) && error("NaN detected in outer loop")
        # @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, Rp=%1.3e] \n", itPH, iter, iter/ni[1], err, min(errVx/errVx0, errVx), min(errVy/errVy0, errVy), min(errPt/errPt0, errPt))
        # @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, Rp=%1.3e] \n", itPH, iter, iter/ni[1], err, errVx, errVy, errPt)
        @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e - norm[Rx=%1.3e %1.3e, Ry=%1.3e %1.3e, Rp=%1.3e %1.3e] \n", itPH, iter, iter/ni[1], err, errVx, errVx/errVx0, errVy, errVy/errVy0, errPt, errPt/errPt0)

        err < ϵ && break

        # Set tolerance of velocity solve proportional to residual
        # ϵ_vel = 1.5e-6 #err * rel_drop
        ϵ_vel = err * rel_drop
        itPT  = 0
        # while (err > dyrel.ϵ_vel && itPT ≤ iterMax)
        while (err > ϵ_vel && itPT ≤ iterMax)
            itPT   += 1
            itg    += 1
            iter   += 1 

            # Pseudo-old dudes 
            copyto!(Rx0, stokes.R.Rx)
            copyto!(Ry0, stokes.R.Ry)
           
            # Divergence 
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)
            # stokes.∇V    .= (stokes.V.Vx[2:end,2:end-1] .- stokes.V.Vx[1:end-1,2:end-1])./di[1] .+ (stokes.V.Vy[2:end-1,2:end] .- stokes.V.Vy[2:end-1,1:end-1])./di[2]

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
         
            # @. stokes.R.RP = -stokes.∇V - (stokes.P.-stokes.P0) / dyrel.ηb 

            # Deviatoric strain rate
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            # # Deviatoric strain rate
            # stokes.ε.xx   .= (stokes.V.Vx[2:end,2:end-1] .- stokes.V.Vx[1:end-1,2:end-1])./di[1] .- 1.0/3.0.*stokes.∇V
            # stokes.ε.yy   .= (stokes.V.Vy[2:end-1,2:end] .- stokes.V.Vy[2:end-1,1:end-1])./di[2] .- 1.0/3.0.*stokes.∇V
            # stokes.ε.xy   .= 0.5.*((stokes.V.Vx[:,2:end] .- stokes.V.Vx[:,1:end-1])./di[2] .+ (stokes.V.Vy[2:end,:] .- stokes.V.Vy[1:end-1,:])./di[1])

            # Deviatoric stress
            # compute_stress_DRYEL!(stokes, rheology, phase_ratios, λ_relaxation_DR, dt)
            # update_halo!(stokes.τ.xy)

            vertex2center!(stokes.ε.xy_c, stokes.ε.xy)
            stokes.τ.xx   .= 2.0.*stokes.viscosity.η  .* stokes.ε.xx 
            stokes.τ.yy   .= 2.0.*stokes.viscosity.η  .* stokes.ε.yy 
            stokes.τ.xy   .= 2.0.*stokes.viscosity.ηv .* stokes.ε.xy 
            stokes.τ.xy_c .= 2.0.*stokes.viscosity.η  .* stokes.ε.xy_c
            stokes.τ.II   .= sqrt.(0.5.*(stokes.τ.xx.^2  .+ stokes.τ.yy.^2  .+ (.-stokes.τ.xx.-stokes.τ.yy).^2)   .+ stokes.τ.xy_c.^2 )

            # update_viscosity_τII!(
            #     stokes,
            #     phase_ratios,
            #     args,
            #     rheology,
            #     viscosity_cutoff;
            #     relaxation = viscosity_relaxation,
            # )

            ηc_true .= @. 2^(3-1) * args.ηc0^3 * stokes.τ.II^(1 - 3)
            stokes.viscosity.η .= exp.(viscosity_relaxation*log.(ηc_true) .+ (1-viscosity_relaxation).*log.(stokes.viscosity.η))
            center2vertex_harm!(stokes.viscosity.ηv, stokes.viscosity.η)
            
            # Residuals
            @. P_num = γ_eff * stokes.R.RP
            # @parallel (@idx ni) compute_DR_residual_V!(
            #     stokes.R.Rx,
            #     stokes.R.Ry,
            #     stokes.P,
            #     P_num,
            #     stokes.ΔPψ,
            #     @stress(stokes)...,
            #     ρg...,
            #     Dx, 
            #     Dy,
            #     _di...,
            # )  
            
            stokes.R.Rx .= (1.0./dyrel.Dx).*(.-(P_num[2:end,:] .- P_num[1:end-1,:])./di[1] .- (stokes.P[2:end,:] .- stokes.P[1:end-1,:])./di[1] .+ (stokes.τ.xx[2:end,:] .- stokes.τ.xx[1:end-1,:])./di[1] .+ (stokes.τ.xy[2:end-1,2:end] .- stokes.τ.xy[2:end-1,1:end-1])./di[2])
            stokes.R.Ry .= (1.0./dyrel.Dy).*(.-(P_num[:,2:end] .- P_num[:,1:end-1])./di[2] .- (stokes.P[:,2:end] .- stokes.P[:,1:end-1])./di[2] .+ (stokes.τ.yy[:,2:end] .- stokes.τ.yy[:,1:end-1])./di[2] .+ (stokes.τ.xy[2:end,2:end-1] .- stokes.τ.xy[1:end-1,2:end-1])./di[1])

            # Damping-pong
            # @parallel (@idx ni) update_V_damping!( (dVxdτ, dVydτ), (stokes.R.Rx, stokes.R.Ry), (αVx, αVy) )

            dyrel.dVxdτ  .= dyrel.αVx.*dyrel.dVxdτ .+ stokes.R.Rx
            dyrel.dVydτ  .= dyrel.αVy.*dyrel.dVydτ .+ stokes.R.Ry

            # # PT updates
            # @parallel (@idx ni.+1) update_DR_V!( (stokes.V.Vx, stokes.V.Vy), (dVxdτ, dVydτ), (βVx, βVy), (dτVx, dτVy) )

            # PT updates
            stokes.V.Vx[2:end-1,2:end-1] .+= dyrel.dVxdτ.*dyrel.βVx.*dyrel.dτVx 
            stokes.V.Vy[2:end-1,2:end-1] .+= dyrel.dVydτ.*dyrel.βVy.*dyrel.dτVy 
            
            flow_bcs!(stokes, flow_bcs)
            # update_halo!(@velocity(stokes)...)
                        
            # @show stokes.τ.II[20,20], stokes.viscosity.η[20,20]

            # Residual check
            if iszero(iter % 400)
                # error()

                errVx = norm(Dx .* stokes.R.Rx) / √(length(stokes.R.Rx))
                errVy = norm(Dy .* stokes.R.Ry) / √(length(stokes.R.Ry))
                
                if iter == nout 
                    errVx00 = errVx
                    errVy00 = errVy
                end
                
                err = maximum( 
                    # (errVx, errVy) 
                    (errVx / errVx00, errVy / errVy00) 
                )
                isnan(err) && error("NaN detected in inner loop")

                push!(err_evo_V, errVx/errVx00)
                push!(err_evo_P, errPt/errPt0)
                push!(err_evo_it, iter)

                @. dVx = dVxdτ * βVx * dτVx
                @. dVy = dVydτ * βVy * dτVy

                # @printf("it = %d, iter = %d, ϵ_vel = %1.3e, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e] \n", itPT, iter, ϵ_vel, err, errVx, errVy)
                @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)

                λminV  = abs(sum(dVx .* (stokes.R.Rx .- Rx0)) + sum(dVy.*(stokes.R.Ry .- Ry0))) / 
                    (sum(dVx.^2) + sum(dVy.^2))
                @. cVx = 2 * √(λminV) * c_fact
                @. cVy = 2 * √(λminV) * c_fact

                # Optimal pseudo-time steps - can be replaced by AD
                Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, γ_eff, phase_ratios, rheology, di, dt)
            
                # Select dτ
                update_dτV_α_β!(dyrel)

                # @show λminV
                # @show dyrel.αVy[20,20], dyrel.βVy[20,20], dyrel.dτVy[20,20], dyrel.cVy[20,20]
                # @show stokes.viscosity.η[20,20]
                # error()
            end
        end
        
        # update pressure
        @. stokes.P += γ_eff.*stokes.R.RP
        
        # Because pressure changed....
        # update viscosity based on the deviatoric stress tensor
        # update_viscosity_τII!(
        #     stokes,
        #     phase_ratios,
        #     args,
        #     rheology,
        #     viscosity_cutoff;
        #     relaxation = 1,
        #     # relaxation = viscosity_relaxation,
        # )
        # # center2vertex_harm!(stokes.viscosity.ηv, stokes.viscosity.η)
       
        # if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
        #     println("Pseudo-transient iterations converged in $iter iterations")
        # end

        iter > 200e3 && break
    end

    # compute vorticity
    @parallel (@idx ni .+ 1) compute_vorticity!(
        stokes.ω.xy, @velocity(stokes)..., inv.(di)...
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

    # recompute all the DYREL variables
    DYREL!(dyrel, stokes, rheology, phase_ratios, di, dt)

    return (; err_evo_it, err_evo_V, err_evo_P)
     
end

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

#                 λminV  = abs(sum(dVx .* (stokes.R.Rx .- Rx0)) + sum(dVy.*(stokes.R.Ry .- Ry0))) / 
#                     (sum(dVx.*dVx) .+ sum(dVy.*dVy))
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
