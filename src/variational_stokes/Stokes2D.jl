## 2D VISCO-ELASTIC STOKES SOLVER

# backend trait
function solve_VariationalStokes!(stokes::JustRelax.StokesArrays, args...; kwargs)
    out = solve_VariationalStokes!(backend(stokes), stokes, args...; kwargs)
    return out
end

# entry point for extensions
function solve_VariationalStokes!(::CPUBackendTrait, stokes, args...; kwargs)
    return _solve_VS!(stokes, args...; kwargs...)
end

function _solve_VS!(
        stokes::JustRelax.StokesArrays,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        phase_ratios::JustPIC.PhaseRatios,
        ϕ::JustRelax.RockRatio,
        rheology,
        args,
        dt,
        igg::IGG;
        air_phase::Integer = 0,
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        iterMax = 50.0e3,
        iterMin = 1.0e2,
        nout = 500,
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
