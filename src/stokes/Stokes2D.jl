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
    di::NTuple{2,T},
    flow_bcs::FlowBoundaryConditions,
    ρg,
    K,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 1),
    verbose=true,
    kwargs...,
) where {T}

    # unpack
    _dx, _dy = inv.(di)
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    (; η) = stokes.viscosity
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

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.RP, stokes.∇V, η, K, dt, r, θ_dτ
            )
            @parallel (@idx ni .+ 1) compute_τ!(
                @stress(stokes)..., @strain(stokes)..., η, θ_dτ
            )

            @hide_communication b_width begin
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    pt_stokes.ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                    dt,
                )
                # apply boundary conditions
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
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
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

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

# visco-elastic solver
function _solve!(
    stokes::JustRelax.StokesArrays,
    pt_stokes,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    G,
    K,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 1),
    verbose=true,
    kwargs...,
) where {T}

    # unpack
    _di = inv.(di)
    (; ϵ, r, θ_dτ) = pt_stokes
    (; η) = stokes.viscosity
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

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, η, K, dt, r, θ_dτ
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
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )
            errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

# GeoParams: general (visco-elasto-plastic) solver

function _solve!(
    stokes::JustRelax.StokesArrays,
    pt_stokes,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    rheology::MaterialParams,
    args,
    dt,
    igg::IGG;
    viscosity_cutoff=(-Inf, Inf),
    viscosity_relaxation=1e-2,
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 0),
    verbose=true,
    free_surface=false,
    kwargs...,
) where {T}

    # unpack
    _di = inv.(di)
    (; ϵ, r, θ_dτ) = pt_stokes
    (; η, η_vep) = stokes.viscosity
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window=(1, 1))
    update_halo!(ητ)
    # end

    Kb = get_Kb(rheology)

    # errors
    err = 2 * ϵ
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
    Vx_on_Vy = @zeros(size(stokes.V.Vy))

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, η, Kb, dt, r, θ_dτ
            )

            update_ρg!(ρg[2], rheology, args)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            update_viscosity!(
                stokes, args, rheology, viscosity_cutoff; relaxation=viscosity_relaxation
            )
            compute_maxloc!(ητ, η; window=(1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                @tensor(stokes.τ_o),
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                stokes.P,
                θ,
                η,
                η_vep,
                λ,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )
            center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            update_halo!(stokes.τ.xy)

            @parallel (1:(size(stokes.V.Vy, 1) - 2), 1:size(stokes.V.Vy, 2)) interp_Vx_on_Vy!(
                Vx_on_Vy, stokes.V.Vx
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    Vx_on_Vy,
                    θ,
                    @stress(stokes)...,
                    pt_stokes.ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                    dt * free_surface,
                )
                # apply boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
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

            errs = maximum.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
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

    stokes.P .= θ

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    # accumulate plastic strain tensor
    @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

## With phase ratios

function _solve!(
    stokes::JustRelax.StokesArrays,
    pt_stokes,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    phase_ratios::JustRelax.PhaseRatio,
    rheology,
    args,
    dt,
    igg::IGG;
    viscosity_cutoff=(-Inf, Inf),
    viscosity_relaxation=1e-2,
    iterMax=50e3,
    iterMin=1e2,
    free_surface=false,
    nout=500,
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
    θ = @zeros(ni...)
    λ = @zeros(ni...)
    η0 = deepcopy(η)
    do_visc = true

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end
    Vx_on_Vy = @zeros(size(stokes.V.Vy))
    
    ρg_bg = 2700 * 9.81
    Plitho = reverse(cumsum(reverse((ρg[2] .+ ρg_bg) .* di[2], dims=2), dims=2), dims=2)
    args.P .= stokes.P .+ Plitho .- minimum(stokes.P)
  
    while iter ≤ iterMax
        iterMin < iter && err < ϵ && break

        wtime0 += @elapsed begin
            compute_maxloc!(ητ, η; window=(1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)

            compute_P!(
                stokes.P,
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

            # stokes.P[1, 1] = stokes.P[2, 1]
            # stokes.P[end, 1] = stokes.P[end - 1, 1]
            # stokes.P[1, end] = stokes.P[2, end]
            # stokes.P[end, end] = stokes.P[end - 1, end]

            update_ρg!(ρg[2], phase_ratios, rheology, args)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            
            if rem(iter, nout) == 0
                @copy η0 η
            end
            if do_visc
                update_viscosity!(
                    stokes,
                    phase_ratios,
                    args,
                    rheology,
                    viscosity_cutoff;
                    relaxation=viscosity_relaxation,
                )
            end

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                @tensor_center(stokes.τ_o),
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                args.P,
                # stokes.P,
                θ,
                η,
                η_vep,
                λ,
                phase_ratios.center,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )
            center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            update_halo!(stokes.τ.xy)

            # stokes.τ.yy[:, end] .= Plitho[:, end]

            @parallel (1:(size(stokes.V.Vy, 1) - 2), 1:size(stokes.V.Vy, 2)) interp_Vx_on_Vy!(
                Vx_on_Vy, stokes.V.Vx
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    Vx_on_Vy,
                    stokes.P,
                    @stress(stokes)...,
                    pt_stokes.ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                    dt * free_surface,
                )
                # apply boundary conditions
                # free_surface_bcs!(stokes, flow_bcs, args, η, rheology, phase_ratios, dt, di)
                free_surface_bcs!(stokes, flow_bcs, η, rheology, phase_ratios, dt, di)
                # free_surface_bcs!(stokes, flow_bcs, η, rheology, phase_ratios, dt, di)
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
            end
        end

        iter += 1

        if iter % nout == 0 && iter > 1
            er_η = norm_mpi(@.(log10(η) - log10(η0)))
            er_η < 1e-3 && (do_visc = false)
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                @velocity(stokes)...,
                Vx_on_Vy,
                stokes.P,
                @stress(stokes)...,
                ρg[1],
                ρg[2],
                _di...,
                dt * free_surface,
            )
            # errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            errs = (
                norm_mpi(stokes.R.Rx) / length(stokes.R.Rx),
                norm_mpi(stokes.R.Ry) / length(stokes.R.Ry),
                norm_mpi(stokes.R.RP) / length(stokes.R.RP),
            )
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
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

        if igg.me == 0 && err ≤ ϵ && iter ≥ 20000
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    # stokes.P .= θ

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    # accumulate plastic strain tensor
    @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)
    # compute_vorticity!(stokes, di)

    # @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    # @parallel (@idx ni) multi_copy!(
        # @tensor_center(stokes.τ_o), @tensor_center(stokes.τ)
    # )

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end
