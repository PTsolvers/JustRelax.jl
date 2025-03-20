
## Adjoint solver

# backend trait
function adjoint_solve!(stokes::JustRelax.StokesArrays, stokesAD::JustRelax.StokesArraysAdjoint, args...; kwargs)
    return adjoint_solve!(backend(stokes), stokes, stokesAD, args...; kwargs)
end

# entry point for extensions
adjoint_solve!(::CPUBackendTrait, stokes, stokesAD, args...; kwargs) = _adjoint_solve!(stokes, stokesAD, args...; kwargs...)

function _adjoint_solve!(
    stokes::JustRelax.StokesArrays,
    stokesAD::JustRelax.StokesArraysAdjoint,
    pt_stokes,
    di::NTuple{2,T},
    flow_bcs::AbstractFlowBoundaryConditions,
    ρg,
    phase_ratios::JustPIC.PhaseRatios,
    rheology,
    args,
    dt,
    Glit,
    SensInd,
    SensType,
    igg::IGG;
    grid,
    origin,
    li,
    viscosity_cutoff=(-Inf, Inf),
    viscosity_relaxation=1e-2,
    iterMax=150e3,
    iterMin=1e2,
    free_surface=false,
    nout=500,
    ADout=10,
    b_width=(4, 4, 0),
    verbose=true,
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
    compute_maxloc!(ητ, η; window=(1, 1))
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
    η0 = deepcopy(η)
    do_visc = true

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end
    Vx_on_Vy = @zeros(size(stokes.V.Vy))

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg, phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    displacement2velocity!(stokes, dt, flow_bcs)

    while iter ≤ iterMax
        iterMin < iter && err < ϵ && break

        wtime0 += @elapsed begin
            compute_maxloc!(ητ, η; window=(1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)

            compute_P!(
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
                args,
            )

            update_ρg!(ρg, phase_ratios, rheology, args)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            # if rem(iter, nout) == 0
            #     @copy η0 η
            # end
            # if do_visc
            update_viscosity!(
                stokes,
                phase_ratios,
                args,
                rheology,
                viscosity_cutoff;
                relaxation=viscosity_relaxation,
            )
            # end

            @parallel (@idx ni .+ 1) update_stresses_center_vertex_ps!(
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
                phase_ratios.xy,
                phase_ratios.yz,
                phase_ratios.xz,
            )
            update_halo!(stokes.τ.xy)

            @parallel (1:(size(stokes.V.Vy, 1) - 2), 1:size(stokes.V.Vy, 2)) interp_Vx∂ρ∂x_on_Vy!(
                Vx_on_Vy, stokes.V.Vx, ρg[2], _di[1]
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    Vx_on_Vy,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                    dt * free_surface,
                )
                # apply boundary conditions
                velocity2displacement!(stokes, dt)
                free_surface_bcs!(stokes, flow_bcs, η, rheology, phase_ratios, dt, di)
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
            end
        end

        iter += 1

        if iter % nout == 0 && iter > 1
            # er_η = norm_mpi(@.(log10(η) - log10(η0)))
            # er_η < 1e-3 && (do_visc = false)
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                @velocity(stokes)...,
                Vx_on_Vy,
                stokes.P,
                @stress(stokes)...,
                ρg...,
                _di...,
                dt * free_surface,
            )
            # errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            errs = (
                norm_mpi(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) /
                length(stokes.R.Rx),
                norm_mpi(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) /
                length(stokes.R.Ry),
                norm_mpi(stokes.R.RP) / length(stokes.R.RP),
            )
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

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    ηb = @zeros(size(stokes.P))
    ρb = @zeros(size(stokes.P))
    ###############################
    ########## Adjoint ############
    ###############################
    indx = 1
    indy = 1
    if rem(Glit+1, ADout) == 0

        # adjoint solver
        adjoint_2D!(
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

        # sensitivity calculation
        ηb, ρb = calc_sensitivity_2D!(
            stokes,
            stokesAD,
            η,
            Vx_on_Vy,
            ρg,
            _di,
            dt,
            free_surface,
            θ,
            λ,
            λv,
            relλ,
            rheology,
            phase_ratios,
            θ_dτ,
            ni,
            )


    end
    ###############################
    ########## Adjoint ############
    ###############################

        # compute vorticity
        @parallel (@idx ni .+ 1) compute_vorticity!(
            stokes.ω.xy, @velocity(stokes)..., inv.(di)...
        )
    
        # accumulate plastic strain tensor
        @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)
    
        @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
        @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
        Ψ_Vx=stokesAD.VA.Vx,
        Ψ_Vy=stokesAD.VA.Vy,
        Ψ_P=stokesAD.PA,
        ηb=ηb,
        ρb=ρb,
    )
end
