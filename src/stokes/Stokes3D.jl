@parallel function update_τ_o!(
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o, τxx, τyy, τzz, τxy, τxz, τyz
)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τzz_o) = @all(τzz)
    @all(τxy_o) = @all(τxy)
    @all(τxz_o) = @all(τxz)
    @all(τyz_o) = @all(τyz)
    return nothing
end

function update_τ_o!(stokes::StokesArrays)
    @parallel update_τ_o!(@tensor(stokes.τ_o)..., @stress(stokes)...)
end

## 3D VISCO-ELASTIC STOKES SOLVER
solve!(stokes::StokesArrays, args...) = solve!(CPUBackendTrait(stokes), stokes, args...)

function solve!(
    ::CPUBackendTrait,
    stokes::StokesArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    flow_bcs,
    ρg,
    K,
    G,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {T}

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    ni = size(stokes.P)
    (; η) = stokes.viscosity

    # ~preconditioner
    ητ = deepcopy(η)
    compute_maxloc!(ητ, η)
    update_halo!(ητ)

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Int64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_Rz = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                K,
                dt,
                pt_stokes.r,
                pt_stokes.θ_dτ,
            )
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                stokes.∇V, @strain(stokes)..., @velocity(stokes)..., _di...
            )
            @parallel (@idx ni .+ 1) compute_τ!(
                @stress(stokes)...,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                η,
                G,
                dt,
                pt_stokes.θ_dτ,
            )
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.R.Rx,
                    stokes.R.Ry,
                    stokes.R.Rz,
                    stokes.P,
                    ρg...,
                    @stress(stokes)...,
                    ητ,
                    pt_stokes.ηdτ,
                    _di...,
                )
                # apply boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            push!(norm_Rx, maximum_mpi(abs.(stokes.R.Rx)))
            push!(norm_Ry, maximum_mpi(abs.(stokes.R.Ry)))
            push!(norm_Rz, maximum_mpi(abs.(stokes.R.Rz)))
            push!(norm_∇V, maximum_mpi(abs.(stokes.R.RP)))
            err = max(norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_Rz[cont],
                    norm_∇V[cont]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    av_time = wtime0 / (iter - 1) # average time per iteration
    update_τ_o!(stokes) # copy τ into τ_o

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_Rz=norm_Rz,
        norm_∇V=norm_∇V,
        time=wtime0,
        av_time=av_time,
    )
end

## 3D VISCO-ELASTO-PLASTIC STOKES SOLVER WITH GeoParams.jl

function solve!(
    ::CPUBackendTrait,
    stokes::StokesArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    flow_bcs::FlowBoundaryConditions,
    ρg,
    rheology::MaterialParams,
    args,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {T}

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    ni = size(stokes.P)
    (; η, η_vep) = stokes.viscosity

    # ~preconditioner
    ητ = deepcopy(η)

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Int64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_Rz = Float64[]
    norm_∇V = Float64[]

    Kb = get_Kb(rheology)
    G = get_shear_modulus(rheology)
    @copy stokes.P0 stokes.P
    λ = @zeros(ni...)
    θ = @zeros(ni...)

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            compute_maxloc!(ητ, η)
            update_halo!(ητ)

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel (@idx ni) compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                ητ,
                Kb,
                dt,
                pt_stokes.r,
                pt_stokes.θ_dτ,
            )
            @parallel (@idx ni) compute_strain_rate!(
                stokes.∇V, @strain(stokes)..., @velocity(stokes)..., _di...
            )

            # # Update buoyancy
            # @parallel (@idx ni) compute_ρg!(ρg[3], rheology, args)

            ν = 1e-3
            @parallel (@idx ni) compute_viscosity!(
                η,
                1.0,
                phase_ratios.center,
                @strain(stokes)...,
                args,
                rheology,
                viscosity_cutoff,
            )

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                stokes.τ.II,
                @tensor(stokes.τ_o),
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                stokes.P,
                θ,
                η,
                @ones(ni...),
                λ,
                tupleize(rheology), # needs to be a tuple
                dt,
                pt_stokes.θ_dτ,
            )

            @parallel (@idx ni .+ 1) compute_τ_vertex!(
                @shear(stokes.τ)...,
                @shear(stokes.τ_o)...,
                @shear(stokes.ε)...,
                η_vep,
                G,
                dt,
                pt_stokes.θ_dτ,
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.R.Rx,
                    stokes.R.Ry,
                    stokes.R.Rz,
                    stokes.P,
                    ρg...,
                    @stress(stokes)...,
                    ητ,
                    pt_stokes.ηdτ,
                    _di...,
                )
                # apply boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            push!(norm_Rx, maximum_mpi(abs.(stokes.R.Rx)))
            push!(norm_Ry, maximum_mpi(abs.(stokes.R.Ry)))
            push!(norm_Rz, maximum_mpi(abs.(stokes.R.Rz)))
            push!(norm_∇V, maximum_mpi(abs.(stokes.R.RP)))
            err = max(norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && (verbose || iter == iterMax)
                @printf(
                    "iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_Rz[cont],
                    norm_∇V[cont]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    av_time = wtime0 / (iter - 1) # average time per iteration
    update_τ_o!(stokes) # copy τ into τ_o

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_Rz=norm_Rz,
        norm_∇V=norm_∇V,
        time=wtime0,
        av_time=av_time,
    )
end

# GeoParams and multiple phases
function solve!(
    ::CPUBackendTrait,
    stokes::StokesArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    flow_bc::FlowBoundaryConditions,
    ρg,
    phase_ratios::PhaseRatio,
    rheology::NTuple{N,AbstractMaterialParamsStruct},
    args,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
    viscosity_cutoff=(-Inf, Inf),
) where {T,N}

    ## UNPACK

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    ni = size(stokes.P)
    (; η, η_vep) = stokes.viscosity

    # errors
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
    λ = @zeros(ni...)
    θ = @zeros(ni...)

    # solver loop
    wtime0 = 0.0
    ητ = deepcopy(η)

    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            # ~preconditioner
            compute_maxloc!(ητ, η)
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
                pt_stokes.r,
                pt_stokes.θ_dτ,
                args,
            )

            @parallel (@idx ni) compute_strain_rate!(
                stokes.∇V, @strain(stokes)..., @velocity(stokes)..., _di...
            )

            # Update buoyancy
            @parallel (@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, args)

            # Update viscosity
            ν = 1e-2
            @parallel (@idx ni) compute_viscosity!(
                η,
                ν,
                phase_ratios.center,
                @strain(stokes)...,
                args,
                rheology,
                viscosity_cutoff,
            )

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                stokes.τ.II,
                @tensor_center(stokes.τ_o),
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                stokes.P,
                θ,
                η,
                η_vep,
                λ,
                phase_ratios.center,
                tupleize(rheology), # needs to be a tuple
                dt,
                pt_stokes.θ_dτ,
            )
            free_surface_bcs!(stokes, flow_bc)

            @parallel (@idx ni .+ 1) center2vertex!(
                stokes.τ.yz,
                stokes.τ.xz,
                stokes.τ.xy,
                stokes.τ.yz_c,
                stokes.τ.xz_c,
                stokes.τ.xy_c,
            )
            update_halo!(stokes.τ.yz, stokes.τ.xz, stokes.τ.xy)

            # @parallel (@idx ni .+ 1) compute_τ_vertex!(
            #     @shear(stokes.τ)..., @shear(stokes.ε)..., η_vep, pt_stokes.θ_dτ
            # )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    @residuals(stokes.R)...,
                    θ,
                    ρg...,
                    @stress(stokes)...,
                    ητ,
                    pt_stokes.ηdτ,
                    _di...,
                )
                # apply boundary conditions
                free_surface_bcs!(stokes, flow_bc, η, rheology, phase_ratios, dt, di)
                flow_bcs!(stokes, flow_bc)
                update_halo!(@velocity(stokes)...)
            end
        end

        stokes.P .= θ

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            for (norm_Ri, Ri) in zip((norm_Rx, norm_Ry, norm_Rz), @residuals(stokes.R))
                push!(norm_Ri, maximum(abs.(Ri)))
            end
            push!(norm_∇V, maximum(abs.(stokes.R.RP)))
            err = max(norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && (verbose || iter == iterMax)
                @printf(
                    "iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_Rz[cont],
                    norm_∇V[cont]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    av_time = wtime0 / (iter - 1) # average time per iteration

    stokes.P .= θ

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_Rz=norm_Rz,
        norm_∇V=norm_∇V,
        time=wtime0,
        av_time=av_time,
    )
end
