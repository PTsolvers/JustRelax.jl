## DIMENSION AGNOSTIC ELASTIC KERNELS

@parallel function elastic_iter_params!(
    dτ_Rho::AbstractArray,
    Gdτ::AbstractArray,
    ητ::AbstractArray,
    Vpdτ::T,
    G::T,
    dt::M,
    Re::T,
    r::T,
    max_li::T,
) where {T,M}
    @all(dτ_Rho) = Vpdτ * max_li / Re / (one(T) / (one(T) / @all(ητ) + one(T) / (G * dt)))
    @all(Gdτ) = Vpdτ^2 / @all(dτ_Rho) / (r + T(2.0))
    return nothing
end

@parallel function elastic_iter_params!(
    dτ_Rho::AbstractArray,
    Gdτ::AbstractArray,
    ητ::AbstractArray,
    Vpdτ::T,
    G::AbstractArray,
    dt::M,
    Re::T,
    r::T,
    max_li::T,
) where {T,M}
    @all(dτ_Rho) =
        Vpdτ * max_li / Re / (one(T) / (one(T) / @all(ητ) + one(T) / (@all(G) * dt)))
    @all(Gdτ) = Vpdτ^2 / @all(dτ_Rho) / (r + T(2.0))
    return nothing
end

## 3D ELASTICITY MODULE

module Stokes3D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using JustRelax
using CUDA
using LinearAlgebra
using Printf
using GeoParams

import JustRelax: elastic_iter_params!, PTArray, Velocity, SymmetricTensor, pureshear_bc!
import JustRelax:
    Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic, IGG
import JustRelax: compute_maxloc!, solve!
import JustRelax: mean_mpi, norm_mpi, minimum_mpi, maximum_mpi

export solve!, pureshear_bc!

include("PressureKernels.jl")
include("StressKernels.jl")
include("VelocityKernels.jl")

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

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,3}) where {A,B,C,D}
    @parallel update_τ_o!(@tensor(stokes.τ_o)..., @stress(stokes)...)
end

## BOUNDARY CONDITIONS 

function JustRelax.pureshear_bc!(
    stokes::StokesArrays, di::NTuple{3,T}, li::NTuple{3,T}, εbg
) where {T}
    # unpack
    Vx, _, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz
    dx, _, dz = di
    lx, _, lz = li
    # Velocity pure shear boundary conditions
    stokes.V.Vx .= PTArray([
        -εbg * ((i - 1) * dx - 0.5 * lx) for i in 1:size(Vx, 1), j in 1:size(Vx, 2),
        k in 1:size(Vx, 3)
    ])
    return stokes.V.Vz .= PTArray([
        εbg * ((k - 1) * dz - 0.5 * lz) for i in 1:size(Vz, 1), j in 1:size(Vz, 2),
        k in 1:size(Vz, 3)
    ])
end

## 3D VISCO-ELASTIC STOKES SOLVER 

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,3},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    flow_bcs,
    ρg,
    η,
    K,
    G,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T}

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    #     @parallel compute_maxloc!(ητ, η)
    #     update_halo!(ητ)
    # end
    compute_maxloc!(ητ, η)
    update_halo!(ητ)
    @parallel (1:size(ητ, 2), 1:size(ητ, 3)) free_slip_x!(ητ)
    @parallel (1:size(ητ, 1), 1:size(ητ, 3)) free_slip_y!(ητ)
    @parallel (1:size(ητ, 1), 1:size(ητ, 2)) free_slip_z!(ητ)

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
            @parallel (@idx ni) compute_∇V!(
                stokes.∇V, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz, _di...
            )
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
                stokes.∇V,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.zz,
                stokes.ε.yz,
                stokes.ε.xz,
                stokes.ε.xy,
                stokes.V.Vx,
                stokes.V.Vy,
                stokes.V.Vz,
                _di...,
            )
            @parallel (@idx ni .+ 1) compute_τ!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.zz,
                stokes.τ.yz,
                stokes.τ.xz,
                stokes.τ.xy,
                stokes.τ_o.xx,
                stokes.τ_o.yy,
                stokes.τ_o.zz,
                stokes.τ_o.yz,
                stokes.τ_o.xz,
                stokes.τ_o.xy,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.zz,
                stokes.ε.yz,
                stokes.ε.xz,
                stokes.ε.xy,
                η,
                G,
                dt,
                pt_stokes.θ_dτ,
            )
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    stokes.V.Vx,
                    stokes.V.Vy,
                    stokes.V.Vz,
                    stokes.R.Rx,
                    stokes.R.Ry,
                    stokes.R.Rz,
                    stokes.P,
                    ρg[1],
                    ρg[2],
                    ρg[3],
                    stokes.τ.xx,
                    stokes.τ.yy,
                    stokes.τ.zz,
                    stokes.τ.yz,
                    stokes.τ.xz,
                    stokes.τ.xy,
                    ητ,
                    pt_stokes.ηdτ,
                    _di...,
                )
                update_halo!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            end

            flow_bcs!(stokes, flow_bcs)
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

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,3},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    flow_bcs::FlowBoundaryConditions,
    ρg,
    η,
    η_vep,
    rheology::MaterialParams,
    args,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T}

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η)
    update_halo!(ητ)
    # end
    # @parallel (1:ny, 1:nz) free_slip_x!(ητ)
    # @parallel (1:nx, 1:nz) free_slip_y!(ητ)
    # @parallel (1:nx, 1:ny) free_slip_z!(ητ)

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
    G = get_G(rheology)
    @copy stokes.P0 stokes.P
    λ = @zeros(ni...)

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel (@idx ni) compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                Kb,
                dt,
                pt_stokes.r,
                pt_stokes.θ_dτ,
            )
            @parallel (@idx ni) compute_strain_rate!(
                stokes.∇V, @strain(stokes)..., @velocity(stokes)..., _di...
            )

            # Update buoyancy
            @parallel (@idx ni) compute_ρg!(ρg[3], rheology, args)

            ν = 1e-3
            @parallel (@idx ni) compute_viscosity!(η, ν, @strain(stokes)..., args, rheology)
            compute_maxloc!(ητ, η)
            update_halo!(ητ)

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                stokes.P,
                η,
                η_vep,
                λ,
                rheology, # needs to be a tuple
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
                update_halo!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            end
            flow_bcs!(stokes, flow_bcs)
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
function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,3},
    thermal::ThermalArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    flow_bc::FlowBoundaryConditions,
    ρg,
    η,
    η_vep,
    phase_ratios::PhaseRatio,
    rheology::NTuple{N,AbstractMaterialParamsStruct},
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T,N}

    ## UNPACK

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    ni = nx, ny, nz = size(stokes.P)
    # z = LinRange(di[3]*0.5, 1.0-di[3]*0.5, nz)

    # ~preconditioner
    ητ = deepcopy(η)
    compute_maxloc!(ητ, η)
    update_halo!(ητ)
    # @hide_communication b_width begin # communication/computation overlap
    #     @parallel compute_maxloc!(ητ, η)
    #     update_halo!(ητ)
    # end
    @parallel (1:ny, 1:nz) free_slip_x!(ητ)
    @parallel (1:nx, 1:nz) free_slip_y!(ητ)
    @parallel (1:nx, 1:ny) free_slip_z!(ητ)

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

    # Kb = get_Kb(rheology)
    # G  = get_G(rheology)
    @copy stokes.P0 stokes.P
    # λ = @zeros(ni...)

    # solver loop
    wtime0 = 0.0
    boo = false
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel (@idx ni) compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                rheology,
                phase_ratios.center,
                dt,
                pt_stokes.r,
                pt_stokes.θ_dτ,
            )

            @parallel (@idx ni) compute_strain_rate!(
                stokes.∇V, @strain(stokes)..., @velocity(stokes)..., _di...
            )

            # Update viscosity
            args_ηv = (; T=thermal.T, P=stokes.P, dt=Inf)
            ν = iter ≥ nout ? 0.05 : 0.0
            # ν = 0.0
            # ν = (iter, 1000) == 0 ? 0.5 : 1.0
            if err < 1e-3 && !boo
                boo = true
                println("Going non-linear at iteration $iter")
            end
            if boo
                # Update buoyancy
                @parallel (@idx ni) compute_ρg!(
                    ρg[3], phase_ratios.center, rheology, (T=thermal.T, P=stokes.P)
                )

                ν = 1e-3
                @parallel (@idx ni) compute_viscosity!(
                    η, ν, phase_ratios.center, @strain(stokes)..., args_ηv, rheology
                )
                # @hide_communication b_width begin # communication/computation overlap
                @parallel compute_maxloc!(ητ, η)
                update_halo!(ητ)
                # end
                @parallel (1:ny, 1:nz) free_slip_x!(ητ)
                @parallel (1:nx, 1:nz) free_slip_y!(ητ)
                @parallel (1:nx, 1:ny) free_slip_z!(ητ)
            end

            @parallel (@idx ni) compute_τ_nonlinear!(
                @stress_center(stokes)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                stokes.P,
                η,
                η_vep,
                phase_ratios.center,
                rheology, # needs to be a tuple
                dt,
                pt_stokes.θ_dτ,
            )

            # @parallel center2vertex!(
            #     stokes.τ.yz, stokes.τ.xz, stokes.τ.xy, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c
            # )

            # args_η = (; P = stokes.P, T=thermal.T)
            # @parallel (@idx ni) compute_τ_gp!(
            #     @stress_center(stokes)...,
            #     stokes.τ.II,
            #     @tensor(stokes.τ_o)...,
            #     @strain(stokes)...,
            #     η,
            #     η_vep,
            #     args_η,
            #     thermal.T,
            #     rheology, # needs to be a tuple
            #     dt,
            #     pt_stokes.θ_dτ,
            # )

            @parallel (@idx ni .+ 1) compute_τ_vertex!(
                @shear(stokes.τ)...,
                @shear(stokes.τ_o)...,
                @shear(stokes.ε)...,
                η_vep,
                rheology,
                phase_ratios.center,
                dt,
                pt_stokes.θ_dτ,
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    @residuals(stokes.R)...,
                    stokes.P,
                    ρg...,
                    @stress(stokes)...,
                    ητ,
                    pt_stokes.ηdτ,
                    _di...,
                )
                update_halo!(@velocity(stokes)...)
            end
            # apply_free_slip!(flow_bc, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            flow_bcs!(stokes, flow_bc, di)
        end

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

end # END OF MODULE
