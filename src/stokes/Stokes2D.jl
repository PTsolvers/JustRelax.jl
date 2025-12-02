## 2D STOKES MODULE
function update_τ_o!(stokes::JustRelax.StokesArrays)
    @parallel (@idx size(τxy)) multi_copy!(
        @tensor_center(stokes.τ_o), @tensor_center(stokes.τ)
    )
    return nothing
end

## 2D VISCO-ELASTIC STOKES SOLVER

# backend trait
function solve!(stokes::JustRelax.StokesArrays, args...; kwargs)
    return solve!(backend(stokes), stokes, args...; kwargs)
end

# entry point for extensions
solve!(::CPUBackendTrait, stokes, args...; kwargs) = _solve!(stokes, args...; kwargs...)

function _solve!(
        stokes::JustRelax.StokesArrays,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        K,
        dt,
        igg::IGG;
        iterMax = 10.0e3,
        nout = 500,
        b_width = (4, 4, 1),
        verbose = true,
        kwargs...,
    ) where {T}

    # unpack
    _di = _dx, _dy = inv.(di)
    (; ϵ_rel, ϵ_abs, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, stokes.viscosity.η; window = (1, 1))
    update_halo!(ητ)
    # end

    # errors
    err_it1 = 1.0
    err = 1.0
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # convert displacement to velocity
    displacement2velocity!(stokes, dt, flow_bcs)

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (((err / err_it1) > ϵ_rel && err > ϵ_abs) && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.RP, stokes.∇V, stokes.Q, η, K, dt, r, θ_dτ
            )
            @parallel (@idx ni) compute_τ!(@stress(stokes)..., @strain(stokes)..., η, θ_dτ)
            @hide_communication b_width begin
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                    dt,
                )
                # apply boundary conditions
                velocity2displacement!(stokes, dt)
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                @velocity(stokes)...,
                stokes.P,
                @stress(stokes)...,
                ρg...,
                _di...,
                dt,
            )
            Vmin, Vmax = extrema(stokes.V.Vx)
            Pmin, Pmax = extrema(stokes.P)
            push!(
                norm_Rx,
                norm_mpi(stokes.R.Rx) / (Pmax - Pmin) * lx / sqrt(length(stokes.R.Rx)),
            )
            push!(
                norm_Ry,
                norm_mpi(stokes.R.Ry) / (Pmax - Pmin) * lx / sqrt(length(stokes.R.Ry)),
            )
            push!(
                norm_∇V, norm_mpi(stokes.∇V) / (Vmax - Vmin) * lx / sqrt(length(stokes.∇V))
            )

            err = maximum_mpi(norm_Rx[end], norm_Ry[end], norm_∇V[end])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            err_it1 = maximum_mpi([norm_Rx[1], norm_Ry[1], norm_∇V[1]])
            rel_err = err / err_it1

            if igg.me == 0 && ((verbose && (err / err_it1) > ϵ_rel && err > ϵ_abs) || iter == iterMax)
                @printf(
                    "Total steps = %d, abs_err = %1.3e , rel_err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    rel_err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

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

# visco-elastic solver
function _solve!(
        stokes::JustRelax.StokesArrays,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        G,
        K,
        dt,
        igg::IGG;
        iterMax = 10.0e3,
        nout = 500,
        b_width = (4, 4, 1),
        verbose = true,
        kwargs...,
    ) where {T}

    # unpack
    _di = inv.(di)
    (; ϵ_rel, ϵ_abs, r, θ_dτ) = pt_stokes
    (; η) = stokes.viscosity
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window = (1, 1))
    update_halo!(ητ)
    # end

    # errors
    err_it1 = 1.0
    err = 1.0
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # convert displacement to velocity
    displacement2velocity!(stokes, dt, flow_bcs)

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (((err / err_it1) > ϵ_rel && err > ϵ_abs) && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)

            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, stokes.Q, ητ, K, G*dt, dt, r, θ_dτ
            )

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            @parallel (@idx ni) compute_τ!(
                @stress(stokes)...,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                η,
                G,
                θ_dτ,
                dt,
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    pt_stokes.ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                )
                # free slip boundary conditions
                velocity2displacement!(stokes, dt)
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )

            errs = (
                norm_mpi(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) /
                    √((nx_g() - 2) * (ny_g() - 1)),
                norm_mpi(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) /
                    √((nx_g() - 1) * (ny_g() - 2)),
                norm_mpi(stokes.R.RP) / √(nx_g() * ny_g()),
            )

            # errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            err_it1 = maximum_mpi([norm_Rx[1], norm_Ry[1], norm_∇V[1]])
            rel_err = err / err_it1

            if igg.me == 0 && ((verbose && (err / err_it1) > ϵ_rel && err > ϵ_abs) || iter == iterMax)
                @printf(
                    "Total steps = %d, abs_err = %1.3e , rel_err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    rel_err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
        end

        if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

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

# GeoParams: general (visco-elasto-plastic) solver

function _solve!(
        stokes::JustRelax.StokesArrays,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        rheology::MaterialParams,
        args,
        dt,
        igg::IGG;
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        iterMax = 10.0e3,
        nout = 500,
        b_width = (4, 4, 0),
        verbose = true,
        free_surface = false,
        kwargs...,
    ) where {T}

    # unpack
    _di = inv.(di)
    (; ϵ_rel, ϵ_abs, r, θ_dτ) = pt_stokes
    (; η, η_vep) = stokes.viscosity
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window = (1, 1))
    update_halo!(ητ)
    # end

    Kb = get_Kb(rheology)

    # errors
    err_it1 = 1.0
    err = 1.0
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # solver loop
    wtime0 = 0.0
    λ = @zeros(ni...)
    θ = @zeros(ni...)
    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window = (1, 1))
    update_halo!(ητ)
    # end

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg[end], rheology, args)
    compute_viscosity!(stokes, args, rheology, viscosity_cutoff)

    # convert displacement to velocity
    displacement2velocity!(stokes, dt, flow_bcs)

    while iter < 2 || (((err / err_it1) > ϵ_rel && err > ϵ_abs) && iter ≤ iterMax)
        wtime0 += @elapsed begin
            compute_maxloc!(ητ, η; window = (1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, stokes.Q, η, Kb, dt, r, θ_dτ
            )

            update_ρg!(ρg[2], rheology, args)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            compute_viscosity_τII!(
                stokes, args, rheology, viscosity_cutoff; relaxation = viscosity_relaxation
            )

            compute_maxloc!(ητ, η; window = (1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                stokes.τ.II,
                @tensor(stokes.τ_o),
                @strain(stokes),
                @plastic_strain(stokes),
                stokes.EII_pl,
                stokes.P,
                θ,
                η,
                η_vep,
                λ,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
                args,
            )
            center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            update_halo!(stokes.τ.xy)

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    pt_stokes.ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                    dt * free_surface,
                )
                # apply boundary conditions
                velocity2displacement!(stokes, dt)
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                @velocity(stokes)...,
                stokes.P,
                @stress(stokes)...,
                ρg...,
                _di...,
                dt * free_surface,
            )

            errs = (
                norm_mpi(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) /
                    √((nx_g() - 2) * (ny_g() - 1)),
                norm_mpi(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) /
                    √((nx_g() - 1) * (ny_g() - 2)),
                norm_mpi(stokes.R.RP) / √(nx_g() * ny_g()),
            )

            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            err_it1 = maximum_mpi([norm_Rx[1], norm_Ry[1], norm_∇V[1]])
            rel_err = err / err_it1

            if igg.me == 0 && ((verbose && (err / err_it1) > ϵ_rel && err > ϵ_abs) || iter == iterMax)
                @printf(
                    "Total steps = %d, abs_err = %1.3e , rel_err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    rel_err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    stokes.P .= θ # θ = P + plastic_overpressure

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

    return (
        iter = iter,
        err_evo1 = err_evo1,
        err_evo2 = err_evo2,
        norm_Rx = norm_Rx,
        norm_Ry = norm_Ry,
        norm_∇V = norm_∇V,
    )
end

## With phase ratios

function _solve!(
        stokes::JustRelax.StokesArrays,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        dt,
        igg::IGG;
        strain_increment = false,
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

    # unpack

    _di = inv.(di)
    _dt = inv.(dt)
    (; ϵ_rel, ϵ_abs, r, θ_dτ, ηdτ) = pt_stokes
    (; η, η_vep) = stokes.viscosity
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window = (1, 1))
    update_halo!(ητ)
    # end

    # errors
    err_it1 = 1.0
    err = 1.0
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
    relλ = λ_relaxation
    θ = deepcopy(stokes.P)
    λ = @zeros(ni...)
    λv = @zeros(ni .+ 1...)
    η0 = deepcopy(η)
    do_visc = true

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg, phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    displacement2velocity!(stokes, dt, flow_bcs)

    while iter ≤ iterMax
        iterMin < iter && ((err / err_it1) < ϵ_rel || err < ϵ_abs) && break

        wtime0 += @elapsed begin
            compute_maxloc!(ητ, η; window = (1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)

            if strain_increment
                @parallel (@idx ni) compute_∇V!(stokes.∇U, @displacement(stokes), _di)
            end

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

            update_ρg!(ρg, phase_ratios, rheology, args)

            if strain_increment
                @parallel (@idx ni .+ 1) compute_strain_rate!(
                    @strain_increment(stokes)..., stokes.∇U, @displacement(stokes)..., _di...
                )

                @parallel (@idx ni .+ 1) compute_strain_rate_from_increment!(
                    @strain(stokes)..., @strain_increment(stokes)..., _dt
                )
            else
                @parallel (@idx ni .+ 1) compute_strain_rate!(
                    @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
                )
            end

            update_viscosity_τII!(
                stokes,
                phase_ratios,
                args,
                rheology,
                viscosity_cutoff;
                relaxation = viscosity_relaxation,
            )
            # end

            if strain_increment
                @parallel (@idx ni .+ 1) update_stresses_center_vertex_ps!(
                    @strain(stokes),
                    @strain_increment(stokes),
                    @plastic_strain(stokes),
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
                    phase_ratios.xz
                )
            else
                @parallel (@idx ni .+ 1) update_stresses_center_vertex_ps!(
                    @strain(stokes),
                    @plastic_strain(stokes),
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
                )
            end

            update_halo!(stokes.τ.xy)

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
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
                stokes.P,
                @stress(stokes)...,
                ρg...,
                _di...,
                dt * free_surface,
            )

            errs = (
                norm_mpi(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) /
                    √((nx_g() - 2) * (ny_g() - 1)),
                norm_mpi(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) /
                    √((nx_g() - 1) * (ny_g() - 2)),
                norm_mpi(stokes.R.RP) / √(nx_g() * ny_g()),
            )

            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            err_it1 = maximum_mpi([norm_Rx[1], norm_Ry[1], norm_∇V[1]])
            rel_err = err / err_it1

            if igg.me == 0 #&& ((verbose && err > ϵ_rel) || iter == iterMax)
                @printf(
                    "Total steps = %d, abs_err = %1.3e , rel_err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    rel_err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
            println("Pseudo-transient iterations converged in $iter iterations")
        end
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

    return (
        iter = iter,
        err_evo1 = err_evo1,
        err_evo2 = err_evo2,
        norm_Rx = norm_Rx,
        norm_Ry = norm_Ry,
        norm_∇V = norm_∇V,
    )
end
