## 3D VISCO-ELASTIC STOKES SOLVER

# backend trait
function solve_VariationalStokes!(stokes::JustRelax.StokesArrays, args...; kwargs)
    solve_VariationalStokes!(backend(stokes), stokes, args...; kwargs)
    return nothing
end

# entry point for extensions
function solve_VariationalStokes!(::CPUBackendTrait, stokes, args...; kwargs)
    return _solve_VS!(stokes, args...; kwargs...)
end

# GeoParams and multiple phases
function _solve_VS!(
        stokes::JustRelax.StokesArrays,
        pt_stokes,
        di::NTuple{3, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        phase_ratios::JustPIC.PhaseRatios,
        ϕ::JustRelax.RockRatio,
        rheology::NTuple{N, AbstractMaterialParamsStruct},
        args,
        dt,
        igg::IGG;
        air_phase::Integer = 0,
        iterMax = 10.0e3,
        nout = 500,
        b_width = (4, 4, 4),
        verbose = true,
        viscosity_relaxation = 1.0e-2,
        viscosity_cutoff = (-Inf, Inf),
        kwargs...,
    ) where {T, N}

    ## UNPACK

    # solver related
    ϵ_rel = pt_stokes.ϵ_rel
    ϵ_abs = pt_stokes.ϵ_abs
    # geometry
    _di = @. 1 / di
    ni = size(stokes.P)
    (; η, η_vep) = stokes.viscosity

    # errors
    err_it1 = 1.0
    err = Inf
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Int64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_Rz = Float64[]
    norm_∇V = Float64[]

    @copy stokes.P0 stokes.P
    θ = deepcopy(stokes.P)
    λ = @zeros(ni...)
    λv_yz = @zeros(size(stokes.τ.yz)...)
    λv_xz = @zeros(size(stokes.τ.xz)...)
    λv_xy = @zeros(size(stokes.τ.xy)...)

    # solver loop
    wtime0 = 0.0
    ητ = deepcopy(η)

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg, phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, air_phase, viscosity_cutoff)

    # convert displacement to velocity
    displacement2velocity!(stokes, dt, flow_bcs)

    while iter < 2 || (((err / err_it1) > ϵ_rel && err > ϵ_abs) && iter ≤ iterMax)
        wtime0 += @elapsed begin
            # ~preconditioner
            compute_maxloc!(ητ, η)
            update_halo!(ητ)

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di...)
            compute_P!(
                θ,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                stokes.Q,
                ητ,
                rheology,
                phase_ratios.center,
                dt,
                pt_stokes.r,
                pt_stokes.θ_dτ,
                args,
            )

            @parallel (@idx ni) compute_strain_rate!(
                stokes.∇V, @strain(stokes)..., @velocity(stokes)..., ϕ, _di...
            )

            # Update buoyancy
            update_ρg!(ρg, phase_ratios, rheology, args)

            # Update viscosity
            update_viscosity!(
                stokes,
                phase_ratios,
                args,
                rheology,
                viscosity_cutoff;
                air_phase = air_phase,
                relaxation = viscosity_relaxation,
            )
            # update_stress!(stokes, θ, λ, phase_ratios, rheology, dt, pt_stokes.θ_dτ)

            @parallel (@idx ni .+ 1) update_stresses_center_vertex!(
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                @tensor_center(stokes.τ),
                (stokes.τ.yz, stokes.τ.xz, stokes.τ.xy),
                @tensor_center(stokes.τ_o),
                (stokes.τ_o.yz, stokes.τ_o.xz, stokes.τ_o.xy),
                θ,
                stokes.P,
                stokes.viscosity.η,
                λ,
                (λv_yz, λv_xz, λv_xy),
                stokes.τ.II,
                stokes.viscosity.η_vep,
                0.2,
                dt,
                pt_stokes.θ_dτ,
                rheology,
                phase_ratios.center,
                phase_ratios.vertex,
                phase_ratios.xy,
                phase_ratios.yz,
                phase_ratios.xz,
                ϕ,
            )
            update_halo!(stokes.τ.yz)
            update_halo!(stokes.τ.xz)
            update_halo!(stokes.τ.xy)

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    @residuals(stokes.R)...,
                    stokes.P,
                    ρg...,
                    @stress(stokes)...,
                    ητ,
                    pt_stokes.ηdτ,
                    ϕ,
                    _di...,
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
            cont += 1
            for (norm_Ri, Ri) in zip((norm_Rx, norm_Ry, norm_Rz), @residuals(stokes.R))
                push!(
                    norm_Ri,
                    norm_mpi(Ri[2:(end - 1), 2:(end - 1), 2:(end - 1)]) / length(Ri),
                )
            end
            push!(norm_∇V, norm_mpi(stokes.R.RP) / length(stokes.R.RP))
            err = max(norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            err_it1 = max(norm_Rx[1], norm_Ry[1], norm_Rz[1], norm_∇V[1])
            rel_err = err / err_it1

            if igg.me == 0 && ((verbose && (err / err_it1) > ϵ_rel && err > ϵ_abs) || iter == iterMax)
                @printf(
                    "iter = %d, abs_err = %1.3e, rel_err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    rel_err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_Rz[cont],
                    norm_∇V[cont]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    av_time = wtime0 / (iter - 1) # average time per iteration

    # compute vorticity
    @parallel (@idx ni .+ 1) compute_vorticity!(
        stokes.ω.yz, stokes.ω.xz, stokes.ω.xy, @velocity(stokes)..., inv.(di)...
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
        norm_Rz = norm_Rz,
        norm_∇V = norm_∇V,
        time = wtime0,
        av_time = av_time,
    )
end
