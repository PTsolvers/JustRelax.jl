## 2D VISCO-ELASTIC STOKES SOLVER

# backend trait
function adjoint_solve_VariationalStokes!(stokes::JustRelax.StokesArrays, args...; kwargs)
    out = adjoint_solve_VariationalStokes!(backend(stokes), stokes, args...; kwargs)
    return out
end

# entry point for extensions
function adjoint_solve_VariationalStokes!(::CPUBackendTrait, stokes, args...; kwargs)
    return _adjoint_solve_VS!(stokes, args...; kwargs...)
end

function _adjoint_solve_VS!(
        stokes::JustRelax.StokesArrays,
        stokesAD::JustRelax.StokesArraysAdjoint,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        phase_ratios::JustPIC.PhaseRatios,
        ϕ::JustRelax.RockRatio,
        rheology,
        args,
        dt,
        Glit,
        SensInd,
        SensType,
        grid,
        origin,
        li,
        ana,
        igg::IGG;
        air_phase::Integer = 0,
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        iterMax = 50.0e3,
        iterMin = 1.0e2,
        nout = 500,
        ADout=10,
        b_width = (4, 4, 0),
        verbose = true,
        free_surface = false,
        kwargs...,
    ) where {T}

    # unpack

    _di = inv.(di)
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    (; η, η_vep) = stokes.viscosity
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window = (1, 1))
    update_halo!(ητ)
    # end

    # errors
    err = 2 * ϵ
    iter = 0
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

    # solver loop
    @copy stokes.P0 stokes.P
    wtime0 = 0.0
    relλ = 0.2
    θ = deepcopy(stokes.P)
    λ = @zeros(ni...)
    λv = @zeros(ni .+ 1...)
    Vx_on_Vy = @zeros(size(stokes.V.Vy))
    η0 = deepcopy(η)
    do_visc = true

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase = air_phase)
    displacement2velocity!(stokes, dt, flow_bcs)

    while iter ≤ iterMax
        iterMin < iter && err < ϵ && break

        wtime0 += @elapsed begin
            compute_maxloc!(ητ, η; window = (1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di)

            compute_P!(
                θ,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                stokes.Q,
                ητ,
                rheology,
                phase_ratios,
                dt,
                r,
                θ_dτ,
                args,
            )

            update_ρg!(ρg[2], phase_ratios, rheology, args)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., ϕ, _di...
            )

            update_viscosity!(
                stokes,
                phase_ratios,
                args,
                rheology,
                viscosity_cutoff;
                air_phase = air_phase,
                relaxation = viscosity_relaxation,
            )

            @parallel (@idx ni .+ 1) update_stresses_center_vertex!(
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
                ϕ,
            )
            update_halo!(stokes.τ.xy)

            # @hide_communication b_width begin # communication/computation overlap
            @parallel (@idx ni .+ 1) compute_V!(
                @velocity(stokes)...,
                stokes.R.Rx,
                stokes.R.Ry,
                stokes.P,
                @stress(stokes)...,
                ηdτ,
                ρg...,
                ητ,
                ϕ,
                _di...,
                dt * free_surface,
            )
            # apply boundary conditions
            velocity2displacement!(stokes, dt)
            # free_surface_bcs!(stokes, flow_bcs, η, rheology, phase_ratios, dt, di)
            flow_bcs!(stokes, flow_bcs)
            update_halo!(@velocity(stokes)...)
            # end
        end

        if iter == iterMax
            error("Maximum iteration reached without convergence")
        end

        iter += 1

        if iter % nout == 0 && iter > 1
            errs = (
                norm_mpi(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) /
                    length(stokes.R.Rx),
                norm_mpi(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) /
                    length(stokes.R.Ry),
                norm_mpi(@views stokes.R.RP[ϕ.center .> 0]) /
                    length(@views stokes.R.RP[ϕ.center .> 0]),
            )
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)

            if igg.me == 0 && verbose #((verbose && err > ϵ) || iter == iterMax)
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

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    # compute vorticity
    @parallel (@idx ni .+ 1) compute_vorticity!(
        stokes.ω.xy, @velocity(stokes)..., inv.(di)...
    )

    # accumulate plastic strain tensor
    @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    if rem(Glit+1, ADout) == 0
    ########################
    #### Adjoint Solver ####
    ########################

    free_surface = false

    if  isdefined(Main,:CUDA)
        mode = Enzyme.Reverse
    else
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse,true)
    end

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
    lx, ly       = li
    ϵ       = ϵ
    err     = 2*ϵ
    iter    = 1
    iterMax = 2e4
    nout    = 1e3
    λtemp   = deepcopy(λ)
    λvtemp  = deepcopy(λv)
    #λtemp  .= 0.0  
    #λvtemp .= 0.0 

    print("###################\n")
    print("## Adjoint Solve ##\n")
    print("###################\n")

    exx = @zeros(size(stokesAD.ε.xx))
    eyy = @zeros(size(stokesAD.ε.yy))
    exy = @zeros(size(stokesAD.ε.xy))

    while (iter ≤ iterMax && err > ϵ)

        # reset derivatives to zero
        stokesAD.V.Vx .= 0.0
        stokesAD.V.Vy .= 0.0
        stokesAD.P    .= 0.0

        # rhs
        if SensType == "Vx"
            stokesAD.V.Vx[SensInd] .= -1.0 # ∂J/∂Vx
        elseif SensType == "Vy"
            stokesAD.V.Vy[SensInd] .= -1.0 # ∂J/∂Vy
        elseif SensType == "P"
            stokesAD.P[SensInd]    .= -1.0 # ∂J/∂P
        end

        # initialize the residuals with the adjoint variables to act as a multiplier
        @views stokesAD.R.Rx .= stokesAD.VA.Vx[2:end-1,2:end-1]
        @views stokesAD.R.Ry .= stokesAD.VA.Vy[2:end-1,2:end-1]
        @views stokesAD.R.RP .= stokesAD.PA

        # Force balanace residuals
        @parallel (@idx ni .+ 1) configcall=compute_Res!(
            @velocity(stokes)...,
            stokes.R.Rx,
            stokes.R.Ry,
            stokes.P,
            @stress(stokes)...,
            ηdτ,
            ρg...,
            ητ,
            ϕ,
            _di...,
            dt * free_surface,
            ) AD.autodiff_deferred!(
                mode, 
                Const(compute_Res!), 
                Const{Nothing},
                Const(stokes.V.Vx),   # to prevent stokes.Vx being calculated
                Const(stokes.V.Vy),   # but stokesAD.V.Vx then needs to be zeroed out later
                DuplicatedNoNeed(stokes.R.Rx,stokesAD.R.Rx),
                DuplicatedNoNeed(stokes.R.Ry,stokesAD.R.Ry),
                DuplicatedNoNeed(stokes.P,stokesAD.P),
                DuplicatedNoNeed(stokes.τ.xx,stokesAD.τ.xx),
                DuplicatedNoNeed(stokes.τ.yy,stokesAD.τ.yy),
                DuplicatedNoNeed(stokes.τ.xy,stokesAD.τ.xy),
                Const(ηdτ),
                Const(ρg[1]),
                Const(ρg[2]),
                Const(ητ),
                Const(ϕ),
                Const(_di[1]),
                Const(_di[2]),
                Const(dt * free_surface)
            )

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

        #    # apply free slip or no slip boundary conditions for adjoint solve
            if ((flow_bcs.free_slip[1]) && (xvi[1][1]   == origin[1]) ) stokesAD.τ.xy[1,:]       .= 0.0 end
            if ((flow_bcs.free_slip[2]) && (xvi[1][end] == origin[1] + lx)) stokesAD.τ.xy[end,:] .= 0.0 end
            if ((flow_bcs.free_slip[3]) && (xvi[2][end] == origin[2] + ly)) stokesAD.τ.xy[:,end] .= 0.0 end
            if ((flow_bcs.free_slip[4]) && (xvi[2][1]   == origin[2])) stokesAD.τ.xy[:,1]        .= 0.0 end
#=
            if ana

            @parallel (@idx ni .+ 1) dτdV_viscoelastic(
                @strain(stokesAD),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                @tensor_center(stokesAD.τ),
                (stokesAD.τ.xy,),
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
                ϕ,
                iter
            )

            
            else
=#
            # stress calculation
            stokesAD.dτ.xx   .= stokes.τ.xx
            stokesAD.dτ.yy   .= stokes.τ.yy
            stokesAD.dτ.xy_c .= stokes.τ.xy_c
            stokesAD.dτ.xy   .= stokes.τ.xy

            stokesAD.P0      .= stokes.P
            #λtemp            .= λ
            #λvtemp           .= λv
            @parallel (@idx ni .+ 1) configcall=update_stresses_center_vertexAD!(
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
                1.0,#relλ,
                dt,
                θ_dτ,
                rheology,
                phase_ratios.center,
                phase_ratios.vertex,
                ϕ,
            ) AD.autodiff_deferred!(
                    mode, 
                    Const(update_stresses_center_vertexAD!), 
                    Const{Nothing},
                    DuplicatedNoNeed(@strain(stokes),@strain(stokesAD)),
                    Const(@tensor_center(stokes.ε_pl)),
                    Const(stokes.EII_pl),
                    DuplicatedNoNeed(@tensor_center(stokesAD.dτ),@tensor_center(stokesAD.τ)),
                    DuplicatedNoNeed((stokesAD.dτ.xy,),(stokesAD.τ.xy,)),
                    Const(@tensor_center(stokes.τ_o)),
                    Const((stokes.τ_o.xy,)),
                    Const(θ),
                    Const(stokesAD.P0),
                    Const(stokes.viscosity.η),
                    Const(λtemp),
                    Const(λvtemp),
                    Const(stokes.τ.II),
                    Const(stokes.viscosity.η_vep),
                    Const(1.0),#Const(relλ),
                    Const(dt),
                    Const(θ_dτ),
                    Const(rheology),
                    Const(phase_ratios.center),
                    Const(phase_ratios.vertex),
                    Const(ϕ)
                )
#            end
        
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

        # apply free slip or no slip boundary conditions for adjoint solve
#        if ((flow_bcs.free_slip[1]) && (xvi[1][1]   == origin[1]) ) stokesAD.ε.xy[1,:]       .= 0.0 end
#        if ((flow_bcs.free_slip[2]) && (xvi[1][end] == origin[1] + lx)) stokesAD.ε.xy[end,:] .= 0.0 end
#        if ((flow_bcs.free_slip[3]) && (xvi[2][end] == origin[2] + ly)) stokesAD.ε.xy[:,end] .= 0.0 end
#        if ((flow_bcs.free_slip[4]) && (xvi[2][1]   == origin[2])) stokesAD.ε.xy[:,1]        .= 0.0 end

            @parallel (@idx ni .+ 1) configcall=compute_strain_rateAD!(
                @strain(stokes)...,
                stokes.∇V,
                @velocity(stokes)...,
                ϕ,
                _di...
            ) AD.autodiff_deferred!(
                    mode, 
                    Const(compute_strain_rateAD!), 
                    Const{Nothing},
                    DuplicatedNoNeed(stokes.ε.xx,stokesAD.ε.xx),
                    DuplicatedNoNeed(stokes.ε.yy,stokesAD.ε.yy),
                    DuplicatedNoNeed(stokes.ε.xy,stokesAD.ε.xy),
                    Const(stokes.∇V),
                    DuplicatedNoNeed(stokes.V.Vx,stokesAD.V.Vx),
                    DuplicatedNoNeed(stokes.V.Vy,stokesAD.V.Vy),
                    Const(ϕ),
                    Const(_di[1]),
                    Const(_di[2])
            )

            # multiplier λP for ∂RP/∂V 
            stokesAD.∇V .= -stokesAD.PA
            @parallel (@idx ni) configcall=compute_∇V!(
                stokes.∇V,
                @velocity(stokes),
                ϕ,
                _di
                ) AD.autodiff_deferred!(
                    mode, 
                    Const(compute_∇V!), 
                    Const{Nothing},
                    DuplicatedNoNeed(stokes.∇V,stokesAD.∇V),
                    DuplicatedNoNeed(@velocity(stokes),@velocity(stokesAD)),
                    Const(ϕ),
                    Const(_di)
            )

            # update λV
            @parallel (@idx ni) update_V!(
                stokesAD.VA.Vx,
                stokesAD.VA.Vy,
                stokesAD.V.Vx,
                stokesAD.V.Vy,
                ηdτ,
                ητ,
                ϕ
            )

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

    ########################
    ########################
    ########################

    print("#############################\n")
    print("## Calculate Sensitivities ##\n")
    print("#############################\n")

    if  isdefined(Main,:CUDA)
        mode = Enzyme.Reverse
    else
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse,true)
    end

    stokesAD.τ.xx   .= 0.0
    stokesAD.τ.yy   .= 0.0
    stokesAD.τ.xy_c .= 0.0
    stokesAD.τ.xy   .= 0.0

    stokesAD.G   .= 0.0
    stokesAD.fr  .= 0.0
    stokesAD.C   .= 0.0

    G  = @zeros(size(stokesAD.G))
    fr = @zeros(size(stokesAD.fr))
    C  = @zeros(size(stokesAD.C))
    Gv = @zeros(size(stokesAD.G).+1)
    frv= @zeros(size(stokesAD.fr).+1)
    Cv = @zeros(size(stokesAD.C).+1)

    Gvb  = @zeros(size(stokesAD.G).+1)
    frvb = @zeros(size(stokesAD.fr).+1)
    Cvb  = @zeros(size(stokesAD.C).+1)

    @views stokesAD.R.Rx .= -stokesAD.VA.Vx[2:end-1,2:end-1]
    @views stokesAD.R.Ry .= -stokesAD.VA.Vy[2:end-1,2:end-1]

    @parallel (@idx ni .+ 1) configcall=compute_Res!(
        @velocity(stokes)...,
        stokes.R.Rx,
        stokes.R.Ry,
        stokes.P,
        @stress(stokes)...,
        ηdτ,
        ρg...,
        ητ,
        ϕ,
        _di...,
        dt * free_surface,
        ) AD.autodiff_deferred!(
            mode, 
            Const(compute_Res!), 
            Const{Nothing},
            Const(stokes.V.Vx),
            Const(stokes.V.Vy),
            DuplicatedNoNeed(stokes.R.Rx,stokesAD.R.Rx),
            DuplicatedNoNeed(stokes.R.Ry,stokesAD.R.Ry),
            Const(stokes.P),
            DuplicatedNoNeed(stokes.τ.xx,stokesAD.τ.xx),
            DuplicatedNoNeed(stokes.τ.yy,stokesAD.τ.yy),
            DuplicatedNoNeed(stokes.τ.xy,stokesAD.τ.xy),
            Const(ηdτ),
            Const(ρg[1]),
            DuplicatedNoNeed(ρg[2],stokesAD.ρ),
            Const(ητ),
            Const(ϕ),
            Const(_di[1]),
            Const(_di[2]),
            Const(dt * free_surface)
        )


    @parallel (@idx ni.+1) assemble_parameter_matrices!(
        stokes.EII_pl,
        G,
        fr,
        C,
        Gv,
        frv,
        Cv,
        rheology,
        phase_ratios.center,
        phase_ratios.vertex
    )

    Sens  = (G, fr, C);
    SensA = (stokesAD.G, stokesAD.fr, stokesAD.C);

    if ana
    @parallel (@idx ni.+1) dτdη_viscoelastic(        
        @strain(stokes),
        @tensor_center(stokes.ε_pl),
        stokes.EII_pl,
        @tensor_center(stokesAD.τ),
        (stokesAD.τ.xy,),
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
        ϕ,
        @tensor_center(stokesAD.τ),
        (stokesAD.τ.xy,)
    )
        vertex2center!(stokesAD.dτ.xx, stokesAD.τ.xy);
        stokesAD.η .+= stokesAD.τ.xx .+ stokesAD.τ.yy .+ stokesAD.τ.xy_c .+ stokesAD.dτ.xx;
  
    else

    θ_dτ = 0.0
    stokesAD.dτ.xx   .= 0.0
    stokesAD.dτ.yy   .= 0.0
    stokesAD.dτ.xy_c .= 0.0
    stokesAD.dτ.xy   .= 0.0
    @parallel (@idx ni.+1) configcall=update_stresses_center_vertexADSens!(
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
        ϕ,
        Sens
        ) AD.autodiff_deferred!(
            mode,
            Const(update_stresses_center_vertexADSens!),
            Const{Nothing},
            Const(@strain(stokes)),
            Const(@tensor_center(stokes.ε_pl)),
            Const(stokes.EII_pl),
            DuplicatedNoNeed(@tensor_center(stokesAD.dτ),@tensor_center(stokesAD.τ)),
            DuplicatedNoNeed((stokesAD.dτ.xy,),(stokesAD.τ.xy,)),
            Const(@tensor_center(stokes.τ_o)),
            Const((stokes.τ_o.xy,)),
            Const(θ),
            Const(stokesAD.P0),
            DuplicatedNoNeed(stokes.viscosity.η,stokesAD.η),
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
            Const(ϕ),
            DuplicatedNoNeed(Sens,SensA)
            )
    end          
end


    return (
        iter = iter,
        err_evo1 = err_evo1,
        err_evo2 = err_evo2,
        norm_Rx = norm_Rx,
        norm_Ry = norm_Ry,
        norm_∇V = norm_∇V,
    )
end




































## 2D VISCO-ELASTIC STOKES SOLVER

# backend trait
function adjoint_solve_VariationalStokesDot!(stokes::JustRelax.StokesArrays, args...; kwargs)
    out = adjoint_solve_VariationalStokesDot!(backend(stokes), stokes, args...; kwargs)
    return out
end

# entry point for extensions
function adjoint_solve_VariationalStokesDot!(::CPUBackendTrait, stokes, args...; kwargs)
    return _adjoint_solve_VSDot!(stokes, args...; kwargs...)
end

function _adjoint_solve_VSDot!(
        stokes::JustRelax.StokesArrays,
        stokesAD::JustRelax.StokesArraysAdjoint,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        phase_ratios::JustPIC.PhaseRatios,
        ϕ::JustRelax.RockRatio,
        rheology,
        args,
        dt,
        Glit,
        SensInd,
        SensType,
        grid,
        origin,
        li,
        dM,
        dp,
        visc,
        dens,
        Gdot,
        frdot,
        igg::IGG;
        air_phase::Integer = 0,
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        iterMax = 50.0e3,
        iterMin = 1.0e2,
        nout = 500,
        ADout=10,
        b_width = (4, 4, 0),
        verbose = true,
        free_surface = false,
        kwargs...,
    ) where {T}

        # unpack

        _di = inv.(di)
        (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
        (; η, η_vep) = stokes.viscosity
        ni = size(stokes.P)
    
        # ~preconditioner
        ητ = deepcopy(η)
        # @hide_communication b_width begin # communication/computation overlap
        compute_maxloc!(ητ, η; window = (1, 1))
        update_halo!(ητ)
        # end
    
        # errors
        err = 2 * ϵ
        iter = 0
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
    
        # solver loop
        @copy stokes.P0 stokes.P
        wtime0 = 0.0
        relλ = 0.2
        θ = deepcopy(stokes.P)
        λ = @zeros(ni...)
        λv = @zeros(ni .+ 1...)
        Vx_on_Vy = @zeros(size(stokes.V.Vy))
        η0 = deepcopy(η)
        do_visc = true
    
        for Aij in @tensor_center(stokes.ε_pl)
            Aij .= 0.0
        end
    
        # compute buoyancy forces and viscosity
        compute_ρg!(ρg[end], phase_ratios, rheology, args)
        ρg[end] .= ρg[end] + (dM*dp*dens)
        compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase = air_phase)
        stokes.viscosity.η .= stokes.viscosity.η + (dM*dp*visc)
        displacement2velocity!(stokes, dt, flow_bcs)
        
    
        while iter ≤ iterMax
            iterMin < iter && err < ϵ && break
    
            wtime0 += @elapsed begin
                compute_maxloc!(ητ, η; window = (1, 1))
                update_halo!(ητ)
    
                @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di)
    
                compute_P!(
                    θ,
                    stokes.P0,
                    stokes.R.RP,
                    stokes.∇V,
                    stokes.Q,
                    ητ,
                    rheology,
                    phase_ratios,
                    dt,
                    r,
                    θ_dτ,
                    args,
                )
    
                update_ρg!(ρg[2], phase_ratios, rheology, args)
    
                @parallel (@idx ni .+ 1) compute_strain_rate!(
                    @strain(stokes)..., stokes.∇V, @velocity(stokes)..., ϕ, _di...
                )
    
                update_viscosity!(
                    stokes,
                    phase_ratios,
                    args,
                    rheology,
                    viscosity_cutoff;
                    air_phase = air_phase,
                    relaxation = viscosity_relaxation,
                )
    
                @parallel (@idx ni .+ 1) update_stresses_center_vertexDot!(
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
                    ϕ,
                    dM,
                    dp,
                    Gdot,
                    frdot
                )
                update_halo!(stokes.τ.xy)
    
                # @hide_communication b_width begin # communication/computation overlap
                @parallel (@idx ni .+ 1) compute_V!(
                    @velocity(stokes)...,
                    stokes.R.Rx,
                    stokes.R.Ry,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
                    ητ,
                    ϕ,
                    _di...,
                    dt * free_surface,
                )
                # apply boundary conditions
                velocity2displacement!(stokes, dt)
                # free_surface_bcs!(stokes, flow_bcs, η, rheology, phase_ratios, dt, di)
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
                # end
            end
    
            iter += 1
    
            if iter % nout == 0 && iter > 1
                errs = (
                    norm_mpi(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) /
                        length(stokes.R.Rx),
                    norm_mpi(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) /
                        length(stokes.R.Ry),
                    norm_mpi(@views stokes.R.RP[ϕ.center .> 0]) /
                        length(@views stokes.R.RP[ϕ.center .> 0]),
                )
                push!(norm_Rx, errs[1])
                push!(norm_Ry, errs[2])
                push!(norm_∇V, errs[3])
                err = maximum_mpi(errs)
                push!(err_evo1, err)
                push!(err_evo2, iter)
    
                if igg.me == 0 && verbose #((verbose && err > ϵ) || iter == iterMax)
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
    
            if igg.me == 0 && err ≤ ϵ
                println("Pseudo-transient iterations converged in $iter iterations")
            end
        end
    
        # compute vorticity
        @parallel (@idx ni .+ 1) compute_vorticity!(
            stokes.ω.xy, @velocity(stokes)..., inv.(di)...
        )
    
        # accumulate plastic strain tensor
        @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)
    
        @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
        @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))
    
        return (
            iter = iter,
            err_evo1 = err_evo1,
            err_evo2 = err_evo2,
            norm_Rx = norm_Rx,
            norm_Ry = norm_Ry,
            norm_∇V = norm_∇V,
        )
    end
    
