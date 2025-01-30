

function solve_adjoint_2D!(stokes,stokesAD,η,xci,xvi,Vx_on_Vy,ρg,_di,dt,free_surface,θ,λ,λv,relλ, ητ,rheology,phase_ratios,r,θ_dτ,args,flow_bcs,origin,ϵ,iterMax,ni,lx,ly,ηdτ,igg)


    print("############################################\n")
    print("Pseudo transient adjoint solver incoooooming\n")
    print("############################################\n")

    free_surface = false    # deactivate free surface terms for AD

    print("############################################\n")
    print("Enzyme START\n")
    print("############################################\n")

        #mode = Enzyme.set_runtime_activity(Enzyme.Reverse,true)
        #Enzyme.API.runtimeActivity!(true)


        if  isdefined(Main,:CUDA)
            mode = Enzyme.Reverse
        else
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse,true)
        end

        # errors
        ϵ  = 1e-2 * ϵ 
        #ϵ  = 1e0 * ϵ
        err = 2*ϵ
        err_evo1 = Float64[]
        err_evo2 = Float64[]
        norm_Rx = Float64[]
        norm_Ry = Float64[]
        norm_∇V = Float64[]
        sizehint!(norm_Rx, Int(iterMax))
        sizehint!(norm_Ry, Int(iterMax))
        sizehint!(norm_∇V, Int(iterMax))
        sizehint!(err_evo1, Int(iterMax))
        sizehint!(err_evo2, Int(iterMax))
        nout = 1e3
        iter = 1
        #iterMax = 20000

            indx = findall((xci[1] .> -0.5) .& (xci[1] .< 0.5))
            indy = findall((xvi[2] .> 0.19) .& (xvi[2] .< 0.21))
            #indx = findall((xvi[1] .> 1750.0*1e3) .& (xvi[1] .< 2250.0*1e3))
            #indy = findall((xci[2] .> -20.0*1e3) .& (xci[2] .< -6.0*1e3))

            
            while (iter ≤ iterMax && err > ϵ)

            stokesAD.V.Vx .= 0.0
            stokesAD.V.Vy .= 0.0
            stokesAD.P    .= 0.0

            stokesAD.V.Vy[indx.+1,indy] .= -1.0
            #stokesAD.V.Vx[indx,indy] .= -1.0
            #update_halo!(stokesAD.VA.Vx)
            #update_halo!(stokesAD.VA.Vy)
    
            stokesAD.R.Rx .= stokesAD.VA.Vx[2:end-1,2:end-1]
            stokesAD.R.Ry .= stokesAD.VA.Vy[2:end-1,2:end-1]
    
            @parallel (@idx ni) configcall=compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                @velocity(stokes)...,
                Vx_on_Vy, stokes.P, 
                @stress(stokes)...,
                ρg...,
                _di...,
                dt * free_surface) AD.autodiff_deferred!(mode, Const(compute_Res!), Const{Nothing}, DuplicatedNoNeed(stokes.R.Rx, stokesAD.R.Rx),DuplicatedNoNeed(stokes.R.Ry, stokesAD.R.Ry),Const(stokes.V.Vx),Const(stokes.V.Vy),Const(Vx_on_Vy),DuplicatedNoNeed(stokes.P,stokesAD.P),DuplicatedNoNeed(stokes.τ.xx,stokesAD.τ.xx),DuplicatedNoNeed(stokes.τ.yy,stokesAD.τ.yy),DuplicatedNoNeed(stokes.τ.xy,stokesAD.τ.xy),Const(ρg[1]),Const(ρg[2]),Const(_di[1]),Const(_di[2]),Const(dt * free_surface))
                #update_halo!(stokesAD.P)
                #update_halo!(stokesAD.τ.xx)
                #update_halo!(stokesAD.τ.yy)
                #update_halo!(stokesAD.τ.xy)

            compute_P!(
                stokesAD.PA,
                stokesAD.P0,
                stokesAD.R.RP,
                stokesAD.P,
                ητ,
                rheology,
                phase_ratios.center,
                dt,
                r,
                θ_dτ,
                args,
                )
    
            stokesAD.∇V .= stokesAD.PA
    
            # apply free slip boundary conditions for adjoint solve
            if ((flow_bcs.free_slip[1]) && (xvi[1][1]   == origin[1]) ) stokesAD.τ.xy[1,:]   .= 0.0 end
            if ((flow_bcs.free_slip[2]) && (xvi[1][end] == origin[1] + lx)) stokesAD.τ.xy[end,:] .= 0.0 end
            if ((flow_bcs.free_slip[3]) && (xvi[2][end] == origin[2] + ly)) stokesAD.τ.xy[:,end] .= 0.0 end
            if ((flow_bcs.free_slip[4]) && (xvi[2][1]   == origin[2])) stokesAD.τ.xy[:,1]   .= 0.0 end
            #update_halo!(stokesAD.τ.xy)
            
            @parallel (@idx ni.+1) configcall=update_stresses_center_vertex_psAD!(
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                @tensor_center(stokes.τ),
                (stokes.τ.xy,),
                @tensor_center(stokes.τ_o),
                (stokes.τ_o.xy,),
                θ,
                stokes.P,
                stokes.viscosity.η,
                λ,
                λv,
                stokes.τ.II,
                stokes.viscosity.η_vep, 
                relλ,
                dt,
                θ_dτ,
                rheology, 
                phase_ratios.center,
                phase_ratios.vertex,
            ) AD.autodiff_deferred!(mode,Const(update_stresses_center_vertex_psAD!),Const{Nothing},DuplicatedNoNeed(@strain(stokes),@strain(stokesAD)),Const(@tensor_center(stokes.ε_pl)),Const(stokes.EII_pl),DuplicatedNoNeed(@tensor_center(stokes.τ),@tensor_center(stokesAD.τ)),DuplicatedNoNeed((stokes.τ.xy,),(stokesAD.τ.xy,)),Const(@tensor_center(stokes.τ_o)),Const((stokes.τ_o.xy,)),Const(θ),Const(stokes.P),Const(stokes.viscosity.η),Const(λ),Const(λv),Const(stokes.τ.II),Const(stokes.viscosity.η_vep),Const(relλ),Const(dt),Const(θ_dτ),Const(rheology),Const(phase_ratios.center),Const(phase_ratios.vertex))
            #update_halo!(stokesAD.ε.xx)
            #update_halo!(stokesAD.ε.yy)
            #update_halo!(stokesAD.ε.xy)

            @parallel (@idx ni .+ 1) configcall=compute_strain_rate!(
                @strain(stokes)...,
                stokes.∇V,
                @velocity(stokes)...,
                _di...) AD.autodiff_deferred!(mode, Const(compute_strain_rate!),Const{Nothing},DuplicatedNoNeed(stokes.ε.xx,stokesAD.ε.xx),DuplicatedNoNeed(stokes.ε.yy,stokesAD.ε.yy),DuplicatedNoNeed(stokes.ε.xy,stokesAD.ε.xy),DuplicatedNoNeed(stokes.∇V,stokesAD.∇V),DuplicatedNoNeed(stokes.V.Vx,stokesAD.V.Vx),DuplicatedNoNeed(stokes.V.Vy,stokesAD.V.Vy),Const(_di[1]),Const(_di[2]))
            #update_halo!(stokesAD.VA.Vx)
            #update_halo!(stokesAD.VA.Vy)

            @parallel (@idx ni) configcall=compute_∇V!(stokes.∇V,@velocity(stokes)..., _di...) AD.autodiff_deferred!(mode, Const(compute_∇V!),Const{Nothing},DuplicatedNoNeed(stokes.∇V,stokesAD.∇V),DuplicatedNoNeed(stokes.V.Vx,stokesAD.V.Vx),DuplicatedNoNeed(stokes.V.Vy,stokesAD.V.Vy),Const(_di[1]),Const(_di[2]))
    
            @parallel update_V!(stokesAD.VA.Vx, stokesAD.VA.Vy, stokesAD.V.Vx, stokesAD.V.Vy, Vx_on_Vy, ηdτ, ρg[2], ητ, _di..., dt* free_surface)
            #update_halo!(stokesAD.VA.Vx)
            #update_halo!(stokesAD.VA.Vy)
    
            iter += 1
    
            if iter % nout == 0 && iter > 1
    
                #er_η = norm_mpi(@.(log10(η) - log10(η0)))
                #er_η < 1e-3 && (do_visc = false)
                #@parallel (@idx ni) compute_Res!(ResVx,ResVy,Vx,Vy,Vx_on_Vy, P, τxx, τyy, τxy, ρgx, ρgy, dx, dy, dt)
                # errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
                errs = (
                    norm_mpi(@views @velocity(stokesAD)[1][2:(end - 1), 2:(end - 1)]) /
                    length(@velocity(stokesAD)[1]),
                    norm_mpi(@views @velocity(stokesAD)[2][2:(end - 1), 2:(end - 1)]) /
                    length(@velocity(stokesAD)[2]),
                    norm_mpi(stokesAD.P) / length(stokesAD.P),
                )
                #global normVx,normVy,normP,it
                push!(norm_Rx,sqrt(sum((abs.(@velocity(stokesAD)[1]).^2)))); push!(norm_Ry,sqrt(sum((abs.(@velocity(stokesAD)[2]).^2)))); push!(norm_∇V,sqrt(sum((abs.(stokesAD.P).^2))))
             
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

return indx, indy

end


function calc_sensitivity_2D(stokes,stokesAD,η,Vx_on_Vy,ρg,_di,dt,free_surface,θ,λ,λv,relλ,rheology,phase_ratios,r,θ_dτ,ni)

if  isdefined(Main,:CUDA)
    mode = Enzyme.Reverse
else
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse,true)
end

stokesAD.τ.xx   .= 0.0    
stokesAD.τ.yy   .= 0.0
stokesAD.τ.xy_c .= 0.0
stokesAD.τ.xy   .= 0.0

stokesAD.Gv  .= 0.0
stokesAD.Gc  .= 0.0
stokesAD.G   .= 0.0
stokesAD.frv .= 0.0
stokesAD.frc .= 0.0
stokesAD.fr  .= 0.0
stokesAD.Cv  .= 0.0
stokesAD.Cc  .= 0.0
stokesAD.C   .= 0.0

ηb      = @zeros(size(η))
ρb      = @zeros(size(ρg[2]))
Gvb     = @zeros(size(stokesAD.Gv))
Gcb     = @zeros(size(stokesAD.Gc))
Gb      = @zeros(size(stokesAD.Gc))
frvb    = @zeros(size(stokesAD.frv))
frcb    = @zeros(size(stokesAD.frc))
frb     = @zeros(size(stokesAD.frc))
Cvb     = @zeros(size(stokesAD.Cv))
Ccb     = @zeros(size(stokesAD.Cc))
Cb      = @zeros(size(stokesAD.Cc))
stokesAD.R.Rx .= -stokesAD.VA.Vx[2:end-1,2:end-1]
stokesAD.R.Ry .= -stokesAD.VA.Vy[2:end-1,2:end-1]


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
     ) AD.autodiff_deferred!(mode, Const(compute_Res!),Const{Nothing},DuplicatedNoNeed(stokes.R.Rx, stokesAD.R.Rx),DuplicatedNoNeed(stokes.R.Ry, stokesAD.R.Ry),Const(stokes.V.Vx),Const(stokes.V.Vy),Const(Vx_on_Vy),Const(stokes.P),DuplicatedNoNeed(stokes.τ.xx,stokesAD.τ.xx),DuplicatedNoNeed(stokes.τ.yy,stokesAD.τ.yy),DuplicatedNoNeed(stokes.τ.xy,stokesAD.τ.xy),Const(ρg[1]),DuplicatedNoNeed(ρg[2],ρb),Const(_di[1]),Const(_di[2]),Const(dt * free_surface))
     #update_halo!(stokesAD.τ.xx)
     #update_halo!(stokesAD.τ.yy)
     #update_halo!(stokesAD.τ.xy)

     @parallel (@idx ni.+1) assemble_parameter_matrices!(stokes.EII_pl,Gvb,Gcb,frvb,frcb,Cvb,Ccb,rheology, phase_ratios.center,phase_ratios.vertex)

     @parallel (@idx ni.+1) configcall=update_stresses_center_vertex_psSensTest!(
        @strain(stokes),
        @tensor_center(stokes.ε_pl),
        stokes.EII_pl,
        @tensor_center(stokes.τ),
        (stokes.τ.xy,),
        @tensor_center(stokes.τ_o),
        (stokes.τ_o.xy,),
        θ,
        stokes.P,
        stokes.viscosity.η,
        λ,
        λv,
        stokes.τ.II,
        stokes.viscosity.η_vep, 
        relλ,
        dt,
        θ_dτ,
        rheology, 
        phase_ratios.center,
        phase_ratios.vertex,
        Gvb,
        Gcb,
        frvb,
        frcb,
        Cvb,
        Ccb
    ) AD.autodiff_deferred!(mode,Const(update_stresses_center_vertex_psSensTest!),Const{Nothing},Const(@strain(stokes)),Const(@tensor_center(stokes.ε_pl)),Const(stokes.EII_pl),DuplicatedNoNeed(@tensor_center(stokes.τ),@tensor_center(stokesAD.τ)),DuplicatedNoNeed((stokes.τ.xy,),(stokesAD.τ.xy,)),Const(@tensor_center(stokes.τ_o)),Const((stokes.τ_o.xy,)),Const(θ),Const(stokes.P),DuplicatedNoNeed(stokes.viscosity.η,ηb),Const(λ),Const(λv),Const(stokes.τ.II),Const(stokes.viscosity.η_vep),Const(relλ),Const(dt),Const(θ_dτ),Const(rheology),Const(phase_ratios.center),Const(phase_ratios.vertex),DuplicatedNoNeed(Gvb,stokesAD.Gv),DuplicatedNoNeed(Gcb,stokesAD.Gc),DuplicatedNoNeed(frvb,stokesAD.frv),DuplicatedNoNeed(frcb,stokesAD.frc),DuplicatedNoNeed(Cvb,stokesAD.Cv),DuplicatedNoNeed(Ccb,stokesAD.Cc))

    vertex2center!(stokesAD.G, stokesAD.Gv)
    stokesAD.G .+= stokesAD.Gc
    vertex2center!(stokesAD.fr, stokesAD.frv)
    stokesAD.fr .+= stokesAD.frc
    vertex2center!(stokesAD.C, stokesAD.Cv)
    stokesAD.C .+= stokesAD.Cc
    
    return ηb, ρb

print("############################################\n")
print("Enzyme END\n")
print("############################################\n")

end
