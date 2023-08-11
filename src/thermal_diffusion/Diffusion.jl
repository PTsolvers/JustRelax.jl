struct ThermalParameters{T}
    K::T # thermal conductivity
    ρCp::T # density * heat capacity
end




# 1D THERMAL DIFFUSION MODULE

module ThermalDiffusion1D

using ParallelStencil
using ParallelStencil.FiniteDifferences1D
using JustRelax
using LinearAlgebra
using Printf
using CUDA

import JustRelax: ThermalParameters, solve!, assign!, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs

export solve!

## KERNELS

@parallel function compute_flux!(qTx, qTx2, T, K, θr_dτ, _dx)
    @all(qTx) = (@all(qTx) * @all(θr_dτ) - @all(K) * @d(T) * _dx) / (1.0 + @all(θr_dτ))
    @all(qTx2) = -@all(K) * @d(T) * _dx
    return nothing
end

@parallel function compute_update!(T, Told, qTx, ρCp, dτ_ρ, _dt, _dx)
    @inn(T) =
        @inn(T) +
        @all(dτ_ρ) * ((-(@d(qTx) * _dx)) - @all(ρCp) * (@inn(T) - @inn(Told)) * _dt)
    return nothing
end

@parallel function check_res!(ResT, T, Told, qTx2, ρCp, _dt, _dx)
    @all(ResT) = -@all(ρCp) * (@inn(T) - @inn(Told)) * _dt - @d(qTx2) * _dx
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    pt_thermal::PTThermalCoeffs,
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,1}},
    ni::NTuple{1,Integer},
    di::NTuple{1,_T},
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {_T,M<:AbstractArray{<:Any,1}}
    @assert size(thermal.T) == ni

    # Compute some constant stuff
    _dt = 1 / dt
    _dx = 1 / di[1]
    _sqrt_len_RT = 1.0 / sqrt(length(thermal.ResT))
    ϵ = pt_thermal.ϵ

    # errors 
    iter_count = Int64[]
    norm_ResT = Float64[]

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0e0
    err = 2 * ϵ
    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            @parallel compute_flux!(
                thermal.qTx,
                thermal.qTx2,
                thermal.T,
                thermal_parameters.K,
                pt_thermal.θr_dτ,
                _dx,
            )

            @parallel compute_update!(
                thermal.T,
                thermal.Told,
                thermal.qTx,
                thermal_parameters.ρCp,
                pt_thermal.dτ_ρ,
                _dt,
                _dx,
            )
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    thermal.qTx2,
                    thermal_parameters.ρCp,
                    _dt,
                    _dx,
                )
            end

            err = norm(thermal.ResT) * _sqrt_len_RT

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if verbose && (err < ϵ) || (iter === iterMax)
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    if iter < iterMax
        @printf("Converged in %d iterations witn err = %1.3e \n", iter, err)
    else
        println("Model not fully converged")
    end

    av_time = wtime0 / iter # average time per iteration

    @parallel assign!(thermal.Told, thermal.T)

    if isnan(err)
        error("NaN")
    end

    return (
        iter=iter, err_evo1=norm_ResT, err_evo2=iter_count, time=wtime0, av_time=av_time
    )
end

end

# 2D THERMAL DIFFUSION MODULE

module ThermalDiffusion2D

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using JustRelax
using LinearAlgebra
using CUDA
using Printf

import JustRelax: ThermalParameters, solve!, assign!, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!

export solve!

## KERNELS

@parallel function compute_flux!(qTx, qTy, qTx2, qTy2, T, K, θr_dτ, _dx, _dy)
    @all(qTx) = (@all(qTx) * @all(θr_dτ) - @all(K) * @d_xi(T) * _dx) / (1.0 + @all(θr_dτ))
    @all(qTy) = (@all(qTy) * @all(θr_dτ) - @all(K) * @d_yi(T) * _dy) / (1.0 + @all(θr_dτ))
    @all(qTx2) = -@all(K) * @d_xi(T) * _dx
    @all(qTy2) = -@all(K) * @d_yi(T) * _dy
    return nothing
end

@parallel function compute_update!(T, Told, qTx, qTy, ρCp, dτ_ρ, _dt, _dx, _dy)
    @inn(T) =
        @inn(T) +
        @all(dτ_ρ) * (
            (-(@d_xa(qTx) * _dx + @d_ya(qTy) * _dy)) -
            @all(ρCp) * (@inn(T) - @inn(Told)) * _dt
        )
    return nothing
end

@parallel function check_res!(ResT, T, Told, qTx2, qTy2, ρCp, _dt, _dx, _dy)
    @all(ResT) =
        -@all(ρCp) * (@inn(T) - @inn(Told)) * _dt - (@d_xa(qTx2) * _dx + @d_ya(qTy2) * _dy)
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    pt_thermal::PTThermalCoeffs,
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,2}},
    thermal_bc::NamedTuple,
    ni::NTuple{2,Integer},
    di::NTuple{2,_T},
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {_T,M<:AbstractArray{<:Any,2}}
    @assert size(thermal.T) == ni

    # Compute some constant stuff
    _dt = 1 / dt
    _dx, _dy = @. 1 / di
    _sq_len_RT = 1.0 / sqrt(length(thermal.ResT))
    ϵ = pt_thermal.ϵ

    # errors 
    iter_count = Int64[]
    norm_ResT = Float64[]

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0e0
    err = 2 * ϵ
    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            @parallel compute_flux!(
                thermal.qTx,
                thermal.qTy,
                thermal.qTx2,
                thermal.qTy2,
                thermal.T,
                thermal_parameters.K,
                pt_thermal.θr_dτ,
                _dx,
                _dy,
            )

            @parallel compute_update!(
                thermal.T,
                thermal.Told,
                thermal.qTx,
                thermal.qTy,
                thermal_parameters.ρCp,
                pt_thermal.dτ_ρ,
                _dt,
                _dx,
                _dy,
            )

            thermal_boundary_conditions!(thermal_bc, thermal.T)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    thermal.qTx2,
                    thermal.qTy2,
                    thermal_parameters.ρCp,
                    _dt,
                    _dx,
                    _dy,
                )
            end

            err = norm(thermal.ResT) * _sq_len_RT

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if verbose && (err < ϵ) || (iter === iterMax)
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    if iter < iterMax
        @printf("Converged in %d iterations witn err = %1.3e \n", iter, err)
    else
        println("Model not fully converged")
    end

    av_time = wtime0 / iter # average time per iteration

    @parallel assign!(thermal.Told, thermal.T)

    if isnan(err)
        error("NaN")
    end

    return (
        iter=iter, err_evo1=norm_ResT, err_evo2=iter_count, time=wtime0, av_time=av_time
    )
end

end

# 3D THERMAL DIFFUSION MODULE

module ThermalDiffusion3D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using JustRelax
using MPI
using Printf
using CUDA

import JustRelax:
    IGG, ThermalParameters, solve!, assign!, norm_mpi, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!

export solve!

## Kernels

@parallel function compute_flux!(
    qTx, qTy, qTz, qTx2, qTy2, qTz2, T, K, θr_dτ, _dx, _dy, _dz
)
    @all(qTx) = (@all(qTx) * @all(θr_dτ) - @all(K) * @d_xi(T) * _dx) / (1.0 + @all(θr_dτ))
    @all(qTy) = (@all(qTy) * @all(θr_dτ) - @all(K) * @d_yi(T) * _dy) / (1.0 + @all(θr_dτ))
    @all(qTz) = (@all(qTz) * @all(θr_dτ) - @all(K) * @d_zi(T) * _dz) / (1.0 + @all(θr_dτ))
    @all(qTx2) = -@all(K) * @d_xi(T) * _dx
    @all(qTy2) = -@all(K) * @d_yi(T) * _dy
    @all(qTz2) = -@all(K) * @d_zi(T) * _dz
    return nothing
end

@parallel function compute_update!(T, Told, qTx, qTy, qTz, ρCp, dτ_ρ, _dt, _dx, _dy, _dz)
    @inn(T) =
        @inn(T) +
        @all(dτ_ρ) * (
            (-(@d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz)) -
            @all(ρCp) * (@inn(T) - @inn(Told)) * _dt
        )
    return nothing
end

@parallel function compute_res!(ResT, T, Told, qTx2, qTy2, qTz2, ρCp, _dt, _dx, _dy, _dz)
    @all(ResT) =
        -@all(ρCp) * (@inn(T) - @inn(Told)) * _dt -
        (@d_xa(qTx2) * _dx + @d_ya(qTy2) * _dy + @d_za(qTz2) * _dz)
    return nothing
end

## Solver

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    pt_thermal::PTThermalCoeffs,
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,3}},
    thermal_bc::NamedTuple,
    ni::NTuple{3,Integer},
    di::NTuple{3,_T},
    igg::IGG,
    dt;
    iterMax=10e3,
    nout=500,
    b_width=(1, 1, 1),
    verbose=true,
) where {_T,M<:AbstractArray{<:Any,3}}
    @assert size(thermal.T) == ni

    # Compute some constant stuff
    _dt = 1 / dt
    _dx, _dy, _dz = @. 1 / di
    nx, ny, nz = ni
    _sqrt_len_ResT_g =
        1 / √(
            ((nx - 2 - 2) * igg.dims[1] + 2) *
            ((ny - 2 - 2) * igg.dims[2] + 2) *
            ((nz - 2 - 2) * igg.dims[3] + 2),
        )
    ϵ = pt_thermal.ϵ

    # errors 
    iter_count = Int64[]
    norm_ResT = Float64[]

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0e0
    err = 2 * ϵ
    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            @parallel compute_flux!(
                thermal.qTx,
                thermal.qTy,
                thermal.qTz,
                thermal.qTx2,
                thermal.qTy2,
                thermal.qTz2,
                thermal.T,
                thermal_parameters.K,
                pt_thermal.θr_dτ,
                _dx,
                _dy,
                _dz,
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_update!(
                    thermal.T,
                    thermal.Told,
                    thermal.qTx,
                    thermal.qTy,
                    thermal.qTz,
                    thermal_parameters.ρCp,
                    pt_thermal.dτ_ρ,
                    _dt,
                    _dx,
                    _dy,
                    _dz,
                )
                update_halo!(thermal.T)
            end

            thermal_boundary_conditions!(thermal_bc, thermal.T)
        end
        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel compute_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    thermal.qTx,
                    thermal.qTy,
                    thermal.qTz,
                    thermal_parameters.ρCp,
                    _dt,
                    _dx,
                    _dy,
                    _dz,
                )
            end

            err = norm_mpi(thermal.ResT) * _sqrt_len_ResT_g

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if igg.me == 0 && (verbose && err < ϵ || iter == iterMax)
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    if iter < iterMax
        @printf("Converged in %d iterations witn err = %1.3e \n", iter, err)
    else
        println("Model not fully converged")
    end

    av_time = wtime0 / iter # average time per iteration

    @parallel assign!(thermal.Told, thermal.T)

    if isnan(err)
        error("NaN")
    end

    return (
        iter=iter, err_evo1=norm_ResT, err_evo2=iter_count, time=wtime0, av_time=av_time
    )
end

end
