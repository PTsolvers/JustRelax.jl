function adjoint_2D!(
    stokes,
    stokesAD,
    pt_stokes,
    _di,
    flow_bcs,
    ρg,
    phase_ratios,
    rheology,
    grid,
    Vx_on_Vy,
    dt,
    igg::IGG;
    free_surface,
    θ,
    λ,
    λv,
    relλ,
    ητ,
    args,
    origin,
    iterMax,
    ni,
    li,
    SensInd,
    SensType,
    )

        free_surface = false    # deactivate free surface terms for adjoint solve

        print("############################################\n")
        print("Adjoint solve\n")
        print("############################################\n")

        if  isdefined(Main,:CUDA)
            mode = Enzyme.Reverse
        else
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse,true)
        end

        (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
        # errors
        ϵ        = 1e-3 * ϵ
        err      = 2*ϵ
        err_evo1 = Float64[]
        err_evo2 = Float64[]
        norm_Rx  = Float64[]
        norm_Ry  = Float64[]
        norm_∇V  = Float64[]
        sizehint!(norm_Rx, Int(iterMax))
        sizehint!(norm_Ry, Int(iterMax))
        sizehint!(norm_∇V, Int(iterMax))
        sizehint!(err_evo1, Int(iterMax))
        sizehint!(err_evo2, Int(iterMax))
        (; xvi, xci) = grid
        lx, ly = li
        nout   = 1e3
        iter   = 1
        iterMax = 2e4
        λtemp   = deepcopy(λ)
        λvtemp  = deepcopy(λv)

        while (iter ≤ iterMax && err > ϵ)

            # reset derivatives to zero
            stokesAD.V.Vx .= 0.0
            stokesAD.V.Vy .= 0.0
            stokesAD.P    .= 0.0
            
            # sensitivity reference points for adjoint solve
            #N_norm = length(SensInd[1])*length(SensInd[2])
            #di = inv.(_di)
            #norm = (di[1]*di[2])/N_norm
            if SensType == "Vx"
                stokesAD.V.Vx[SensInd[1],SensInd[2].+1] .= -1.0#*norm
            elseif SensType == "Vy"
                stokesAD.V.Vy[SensInd[1].+1,SensInd[2]] .= -1.0#*norm
            elseif SensType == "P"
                stokesAD.P[SensInd[1],SensInd[2]] .= -1.0#*norm
            end

            # initialize the residuals with the adjoint variables to act as a multiplier
            @views stokesAD.R.Rx .= stokesAD.VA.Vx[2:end-1,2:end-1]
            @views stokesAD.R.Ry .= stokesAD.VA.Vy[2:end-1,2:end-1]
            @views stokesAD.R.RP .= stokesAD.PA

            @parallel (@idx ni) configcall=compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                @velocity(stokes)...,
                Vx_on_Vy, 
                stokes.P,
                @stress(stokes)...,
                ρg...,
                _di...,
                dt * free_surface
                ) AD.autodiff_deferred!(
                    mode, 
                    Const(compute_Res!), 
                    Const{Nothing}, 
                    DuplicatedNoNeed(stokes.R.Rx, stokesAD.R.Rx),
                    DuplicatedNoNeed(stokes.R.Ry, stokesAD.R.Ry),
                    Const(stokes.V.Vx),
                    Const(stokes.V.Vy),
                    Const(Vx_on_Vy),
                    DuplicatedNoNeed(stokes.P,stokesAD.P),
                    DuplicatedNoNeed(stokes.τ.xx,stokesAD.τ.xx),
                    DuplicatedNoNeed(stokes.τ.yy,stokesAD.τ.yy),
                    DuplicatedNoNeed(stokes.τ.xy,stokesAD.τ.xy),
                    Const(ρg[1]),
                    Const(ρg[2]),
                    Const(_di[1]),
                    Const(_di[2]),
                    Const(dt * free_surface))
            
            @parallel (@idx ni) configcall=compute_P_kernelAD!(
                θ,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                ητ,
                rheology,
                phase_ratios.center,
                dt,
                r,
                θ_dτ,
                nothing,
                nothing
                ) AD.autodiff_deferred!(
                    mode,
                    Const(compute_P_kernelAD!),
                    Const{Nothing},
                    DuplicatedNoNeed(θ,stokesAD.P),
                    Const(stokes.P0),
                    DuplicatedNoNeed(stokes.R.RP,stokesAD.R.RP),
                    Const(stokes.∇V),
                    Const(ητ),
                    Const(rheology),
                    Const(phase_ratios.center),
                    Const(dt),
                    Const(r),
                    Const(θ_dτ),
                    Const(nothing),
                    Const(nothing)
                    )
            
            # apply free slip or no slip boundary conditions for adjoint solve
            if ((flow_bcs.free_slip[1]) && (xvi[1][1]   == origin[1]) ) stokesAD.τ.xy[1,:]       .= 0.0 end
            if ((flow_bcs.free_slip[2]) && (xvi[1][end] == origin[1] + lx)) stokesAD.τ.xy[end,:] .= 0.0 end
            if ((flow_bcs.free_slip[3]) && (xvi[2][end] == origin[2] + ly)) stokesAD.τ.xy[:,end] .= 0.0 end
            if ((flow_bcs.free_slip[4]) && (xvi[2][1]   == origin[2])) stokesAD.τ.xy[:,1]        .= 0.0 end

            # Copy stokes stress. If not, stokes.τ is changed during the Enzyme call
            stokesAD.dτ.xx   .= stokes.τ.xx
            stokesAD.dτ.yy   .= stokes.τ.yy
            stokesAD.dτ.xy_c .= stokes.τ.xy_c
            stokesAD.dτ.xy   .= stokes.τ.xy
            stokesAD.P0      .= stokes.P
            λtemp            .= λ
            λvtemp           .= λv
            @parallel (@idx ni.+1) configcall=update_stresses_center_vertex_psAD!(
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                @tensor_center(stokesAD.dτ),
                (stokesAD.dτ.xy,),
                @tensor_center(stokes.τ_o),
                (stokes.τ_o.xy,),
                θ,
                stokesAD.P0,
                stokes.viscosity.η,
                λtemp,
                λvtemp,
                stokes.τ.II,
                stokes.viscosity.η_vep,
                relλ,
                dt,
                θ_dτ,
                rheology,
                phase_ratios.center,
                phase_ratios.vertex,
                phase_ratios.xy,
                phase_ratios.yz,
                phase_ratios.xz
                ) AD.autodiff_deferred!(
                    mode,
                    Const(update_stresses_center_vertex_psAD!),
                    Const{Nothing},
                    DuplicatedNoNeed(@strain(stokes), @strain(stokesAD)),
                    Const(@tensor_center(stokes.ε_pl)),
                    Const(stokes.EII_pl),
                    DuplicatedNoNeed(@tensor_center(stokesAD.dτ), @tensor_center(stokesAD.τ)),
                    DuplicatedNoNeed((stokesAD.dτ.xy,),(stokesAD.τ.xy,)),
                    Const(@tensor_center(stokes.τ_o)),
                    Const((stokes.τ_o.xy,)),
                    #DuplicatedNoNeed(stokesAD.P0,stokesAD.P),
                    Const(θ),
                    Const(stokesAD.P0),
                    #DuplicatedNoNeed(stokesAD.P0,stokesAD.P),
                    Const(stokes.viscosity.η),
                    Const(λtemp),
                    Const(λvtemp),
                    Const(stokes.τ.II),
                    Const(stokes.viscosity.η_vep),
                    Const(relλ),
                    Const(dt),
                    Const(θ_dτ),
                    Const(rheology),
                    Const(phase_ratios.center),
                    Const(phase_ratios.vertex),
                    Const(phase_ratios.xy),
                    Const(phase_ratios.yz),
                    Const(phase_ratios.xz)
                    )
        
            @parallel (@idx ni) update_PAD!(
                stokesAD.PA,
                stokesAD.P,
                stokesAD.R.RP,
                stokes.∇V,
                ητ,
                rheology,
                phase_ratios.center,
                dt,
                r,
                θ_dτ,
                args)

            @parallel (@idx ni .+ 1) configcall=compute_strain_rateAD!(
                @strain(stokes)...,
                @velocity(stokes)...,
                _di...
                ) AD.autodiff_deferred!(
                    mode,
                    Const(compute_strain_rateAD!),
                    Const{Nothing},
                    DuplicatedNoNeed(stokes.ε.xx,stokesAD.ε.xx),
                    DuplicatedNoNeed(stokes.ε.yy,stokesAD.ε.yy),
                    DuplicatedNoNeed(stokes.ε.xy,stokesAD.ε.xy),
                    DuplicatedNoNeed(stokes.V.Vx,stokesAD.V.Vx),
                    DuplicatedNoNeed(stokes.V.Vy,stokesAD.V.Vy),
                    Const(_di[1]),
                    Const(_di[2]))

            # multiplier λP for ∂RP/∂V 
            stokesAD.∇V .= -stokesAD.PA
            # calculate ∂RP/∂V and multiply with λP
            @parallel (@idx ni) configcall=compute_∇V!(
                stokes.∇V,
                @velocity(stokes)...,
                _di...
                ) AD.autodiff_deferred!(
                    mode,
                    Const(compute_∇V!),
                    Const{Nothing},
                    DuplicatedNoNeed(stokes.∇V,stokesAD.∇V),
                    DuplicatedNoNeed(stokes.V.Vx,stokesAD.V.Vx),
                    DuplicatedNoNeed(stokes.V.Vy,stokesAD.V.Vy),
                    Const(_di[1]),
                    Const(_di[2]))
                    
            # update λV
            @parallel update_V!(
                stokesAD.VA.Vx,
                stokesAD.VA.Vy,
                stokesAD.V.Vx,
                stokesAD.V.Vy,
                Vx_on_Vy,
                ηdτ,
                ρg[2],
                ητ,
                _di...,
                dt* free_surface)

            iter += 1

            if iter % nout == 0 && iter > 1

                errs = (
                    norm_mpi(@views @velocity(stokesAD)[1][2:(end - 1), 2:(end - 1)]) /
                    length(@velocity(stokesAD)[1]),
                    norm_mpi(@views @velocity(stokesAD)[2][2:(end - 1), 2:(end - 1)]) /
                    length(@velocity(stokesAD)[2]),
                    norm_mpi(stokesAD.P) / length(stokesAD.P),
                )
                #global normVx,normVy,normP,it
                push!(norm_Rx,sqrt(sum((abs.(@velocity(stokesAD)[1]).^2)))); push!(norm_Ry,sqrt(sum((abs.(@velocity(stokesAD)[2]).^2)))); push!(norm_∇V,sum((abs.(stokesAD.P))))

                push!(norm_Rx, errs[1])
                push!(norm_Ry, errs[2])
                push!(norm_∇V, errs[3])
                err = maximum_mpi(errs)
                push!(err_evo1, err)
                push!(err_evo2, iter)

                if igg.me == 0 #&& ((verbose && err > ϵ) || iter == iterMax)
                    @printf(
                        "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                        iter,
                        err,
                        norm_Rx[end],
                        norm_Ry[end],
                        norm_∇V[end]
                    )
                end
                isnan(err) && error("NaN(s)")
            end
    end
end
