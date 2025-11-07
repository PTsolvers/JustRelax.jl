using StaticArrays, MuladdMacro, Printf
import JustRelax.JustRelax2D: get_shear_modulus, compute_∇V!, update_ρg!, compute_strain_rate!, cache_tensors, get_bulk_modulus, clamped_indices, av_clamped
using LinearAlgebra

include("pressure_kernels.jl")
include("stress_kernels.jl")
include("velocity_kernels.jl")
include("Gershgorin.jl")

function _solve_DYREL!(
        stokes::JustRelax.StokesArrays,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        dt,
        DYREL,
        igg::IGG;
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        λ_relaxation = 0.2,
        iterMax = 50.0e3,
        iterMin = 1.0e2,
        free_surface = false,
        nout = 500,
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
        CFL_v,
        c_fact,
        ηb,
    ) = DYREL
    # unpack

    _di = inv.(di)
    _dt = inv.(dt)
    # (; ϵ_rel, ϵ_abs, r, θ_dτ, ηdτ) = pt_stokes
    (; η, η_vep) = stokes.viscosity
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)

    # errors
    err_it1 = 1.0
    err = 1.0
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]
    # sizehint!(norm_Rx, Int(iterMax))
    # sizehint!(norm_Ry, Int(iterMax))
    # sizehint!(norm_∇V, Int(iterMax))
    # sizehint!(err_evo1, Int(iterMax))
    # sizehint!(err_evo2, Int(iterMax))

    # solver loop
    @copy stokes.P0 stokes.P
    Rx0 = similar(stokes.R.Rx)
    Ry0 = similar(stokes.R.Ry)

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg, phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # while iter ≤ iterMax
    #     iterMin < iter && ((err / err_it1) < ϵ_rel || err < ϵ_abs) && break

    rel_drop = 1e-1         # relative drop of velocity residual per PH iteration
    # Iteration loop
    errVx0     = 1.0
    errVy0     = 1.0
    errPt0     = 1.0 
    errVx00    = 1.0
    errVy00    = 1.0 
    iter       = 1
    ϵ          = 1e-9
    err        = 2 * ϵ
    err_evo_V  = Float64[]
    err_evo_P  = Float64[]
    err_evo_it = Float64[]
    itg        = 0

    # Powell-Hestenes iterations
    for itPH in 1:50

        # compute divergence
        @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)

        # compute deviatoric strain rate
        @parallel (@idx ni .+ 1) compute_strain_rate!(
            @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
        )

        # compute deviatoric stress
        compute_stress_DRYEL!(stokes, rheology, phase_ratios, 1, dt)
        update_halo!(stokes.τ.xy)

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
        errVx = norm(stokes.R.Rx)
        errVy = norm(stokes.R.Ry)
        errPt = norm(stokes.R.RP)
        if isone(itPH)
            errVx0 = errVx
            errVy0 = errVy
            errPt0 = errPt
        end
        err = maximum(
            (min(errVx/errVx0, errVx), min(errVy/errVy0, errVy))
        )
        @printf("itPH = %02d iter = %06d iter/nx = %03d, err = %1.3e norm[Rx=%1.3e, Ry=%1.3e, Rp=%1.3e] \n", itPH, iter, iter/ni[1], err, min(errVx/errVx0, errVx), min(errVy/errVy0, errVy), min(errPt/errPt0, errPt))
        err<ϵ && break

        # Set tolerance of velocity solve proportional to residual
        ϵ_vel = err * rel_drop
        itPT  = 0
        while (err>ϵ_vel && itPT<=iterMax)
            itPT   += 1
            itg    += 1

            # Pseudo-old dudes 
            copyto!(Rx0, stokes.R.Rx)
            copyto!(Ry0, stokes.R.Ry)
           
            # Divergence 
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)
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
         
            # Deviatoric strain rate
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
                  
            # Deviatoric stress
            compute_stress_DRYEL!(stokes, rheology, phase_ratios, λ_relaxation, dt)
            update_halo!(stokes.τ.xy)
            
            # Residuals
            P_num = γ_eff .* stokes.R.RP
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
                _di...,
            )  
            
            # Damping-pong
            @parallel (@idx ni) update_V_damping!( (dVxdτ, dVydτ), (stokes.R.Rx, stokes.R.Ry), (αVx, αVy) )

            # PT updates
            @parallel (@idx ni.+1) update_DR_V!( (stokes.V.Vx, stokes.V.Vy), (dVxdτ, dVydτ), (βVx, βVy), (dτVx, dτVy) )
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)
            
            # Residual check
            if iszero(iter % nout)
                
                errVx = norm(Dx.*stokes.R.Rx)
                errVy = norm(Dy.*stokes.R.Ry)
                
                if iter == nout 
                    errVx00 = errVx
                    errVy00 = errVy
                end
                
                err = maximum( (errVx / errVx00, errVy / errVy00) )
                push!(err_evo_V, errVx/errVx00)
                push!(err_evo_P, errPt/errPt0)
                push!(err_evo_it, iter)

                @. dVx = dVxdτ * βVx * dτVx
                @. dVy = dVydτ * βVy * dτVy

                λminV  = abs(sum(dVx .* (stokes.R.Rx .- Rx0)) + sum(dVy.*(stokes.R.Ry .- Ry0))) / 
                    (sum(dVx.*dVx) .+ sum(dVy.*dVy))
                cVx .= 2 * √(λminV) * c_fact
                cVy .= 2 * √(λminV) * c_fact

                # Optimal pseudo-time steps - can be replaced by AD
                Gershgorin_Stokes2D_SchurComplement!(Dx, Dy, λmaxVx, λmaxVy, stokes.viscosity.η, stokes.viscosity.ηv, γ_eff, phase_ratios, rheology, di, dt)
            
                # Select dτ
                update_dτV_α_β!(dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy, CFL_v)

            end
            iter += 1 
        end
        
        # update pressure
        stokes.P .+= γ_eff.*stokes.R.RP
        
        # update buoyancy forces
        JustRelax2D.update_ρg!(ρg, phase_ratios, rheology, args)
        
        # update buoyancy viscosity
        JustRelax2D.update_viscosity_τII!(
            stokes,
            phase_ratios,
            args,
            rheology,
            viscosity_cutoff;
            relaxation = viscosity_relaxation,
        )

        # if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
        #     println("Pseudo-transient iterations converged in $iter iterations")
        # end
    end

    # compute vorticity
    # @parallel (@idx ni .+ 1) JustRelax2D.compute_vorticity!(
    #     stokes.ω.xy, @velocity(stokes)..., inv.(di)...
    # )

    # # Interpolate shear components to cell center arrays
    # JustRelax2D.shear2center!(stokes.ε)
    # JustRelax2D.shear2center!(stokes.ε_pl)
    # JustRelax2D.shear2center!(stokes.Δε)

    # # accumulate plastic strain tensor
    # JustRelax2D.accumulate_tensor!(stokes.EII_pl, stokes.ε_pl, dt)

    @parallel (@idx ni .+ 1) JustRelax2D.multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) JustRelax2D.multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    # return (
    #     iter = iter,
    #     err_evo1 = err_evo1,
    #     err_evo2 = err_evo2,
    #     norm_Rx = norm_Rx,
    #     norm_Ry = norm_Ry,
    #     norm_∇V = norm_∇V,
    # )
end


