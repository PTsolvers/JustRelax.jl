struct ThermalParameters{T}
    κ::T # thermal diffusivity

    function ThermalParameters(K::AbstractArray, ρCp::AbstractArray)
        κ = K ./ ρCp
        return new{typeof(κ)}(κ)
    end
end

@parallel function update_T!(ΔT, T, Told)
    @all(ΔT) = @all(T) - @all(Told)
    return nothing
end

## GeoParams

function compute_diffusivity(rheology::MaterialParams, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end
function compute_diffusivity(rheology::MaterialParams, ρ::Number, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * ρ)
end

# 1D THERMAL DIFFUSION MODULE

module ThermalDiffusion1D

using ParallelStencil
using ParallelStencil.FiniteDifferences1D
using JustRelax
using LinearAlgebra
using Printf
using CUDA

import JustRelax:
    ThermalParameters, solve!, assign!, thermal_boundary_conditions!, update_T!
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
    @parallel assign!(thermal.Told, thermal.T)

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

    @parallel update_T!(thermal.ΔT, thermal.T, thermal.Told)

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
using CUDA
# using MPI
using GeoParams

import JustRelax: ThermalParameters, solve!, assign!, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!, compute_diffusivity

export solve!

## KERNELS

@parallel function compute_flux!(qTx, qTy, T, κ, _dx, _dy)
    @all(qTx) = -@av_xi(κ) * @d_xi(T) * _dx
    @all(qTy) = -@av_yi(κ) * @d_yi(T) * _dy
    return nothing
end

@parallel_indices (i, j) function compute_flux!(
    qTx, qTy, T, rheology::MaterialParams, args, _dx, _dy
)
    @inline dTdxi(i, j) = (T[i + 1, j + 1] - T[i, j + 1]) * _dx
    @inline dTdyi(i, j) = (T[i + 1, j + 1] - T[i + 1, j]) * _dy

    if i ≤ size(qTx, 1) && j ≤ size(qTx, 2)
        qTx[i, j] = -compute_diffusivity(rheology, args) * dTdxi(i, j)
    end

    if i ≤ size(qTy, 1) && j ≤ size(qTy, 2)
        qTy[i, j] = -compute_diffusivity(rheology, args) * dTdyi(i, j)
    end

    return nothing
end

@parallel_indices (i, j) function advect_T!(dT_dt, qTx, qTy, T, Vx, Vy, _dx, _dy)
    if (i ≤ size(dT_dt, 1) && j ≤ size(dT_dt, 2))
        
        Vxᵢⱼ = 0.5 * (Vx[i + 2, j + 2] + Vx[i + 1, j + 2])
        Vyᵢⱼ = 0.5 * (Vy[i + 2, j + 2] + Vy[i + 2, j + 1])

        dT_dt[i, j] =
            -((qTx[i + 1, j] - qTx[i, j]) * _dx + (qTy[i, j + 1] - qTy[i, j]) * _dy) -
            (Vxᵢⱼ > 0) * Vxᵢⱼ * (T[i + 1, j + 1] - T[i    , j + 1]) * _dx -
            (Vxᵢⱼ < 0) * Vxᵢⱼ * (T[i + 2, j + 1] - T[i + 1, j + 1]) * _dx -
            (Vyᵢⱼ > 0) * Vyᵢⱼ * (T[i + 1, j + 1] - T[i + 1, j    ]) * _dy -
            (Vyᵢⱼ < 0) * Vyᵢⱼ * (T[i + 1, j + 2] - T[i + 1, j + 1]) * _dy
    end
    return nothing
end

@parallel function advect_T!(dT_dt, qTx, qTy, _dx, _dy)
    @all(dT_dt) = -(@d_xa(qTx) * _dx + @d_ya(qTy) * _dy)
    return nothing
end

@parallel function update_T!(T, dT_dt, dt)
    @inn(T) = @inn(T) + @all(dT_dt) * dt
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,2}},
    thermal_bc::NamedTuple,
    di::NTuple{2,_T},
    dt,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, thermal_parameters.κ, _dx, _dy
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, _dx, _dy)
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,2}},
    stokes,
    thermal_bc::NamedTuple,
    di::NTuple{2,_T},
    dt,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, thermal_parameters.κ, _dx, _dy
    )
    @parallel advect_T!(
        thermal.dT_dt,
        thermal.qTx,
        thermal.qTy,
        thermal.T,
        stokes.V.Vx,
        stokes.V.Vy,
        _dx,
        _dy,
    )
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# GEOPARAMS VERSION

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::NamedTuple,
    rheology::MaterialParams,
    args::NamedTuple,
    di::NTuple{2,_T},
    dt;
    advection=true,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)
    nx, ny = size(thermal.T)

    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, rheology, args, _dx, _dy
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, _dx, _dy)
    @show extrema(thermal.dT_dt)
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# Upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::NamedTuple,
    stokes,
    rheology::MaterialParams,
    args::NamedTuple,
    di::NTuple{2,_T},
    dt;
    advection=true,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)
    nx, ny = size(thermal.T)

    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, rheology, args, _dx, _dy
    )
    @parallel advect_T!(
        thermal.dT_dt,
        thermal.qTx,
        thermal.qTy,
        thermal.T,
        stokes.V.Vx,
        stokes.V.Vy,
        _dx,
        _dy,
    )
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
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
using GeoParams

import JustRelax:
    IGG, ThermalParameters, solve!, assign!, norm_mpi, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!, compute_diffusivity

export solve!

## KERNELS

@parallel function compute_flux!(qTx, qTy, qTz, T, κ, _dx, _dy, _dz)
    @all(qTx) = -@av_xi(κ) * @d_xi(T) * _dx
    @all(qTy) = -@av_yi(κ) * @d_yi(T) * _dy
    @all(qTz) = -@av_yi(κ) * @d_zi(T) * _dz
    return nothing
end

@parallel_indices (i, j, k) function compute_flux!(
    qTx, qTy, qTz, T, rheology::MaterialParams, args, _dx, _dy, _dz
)
    @inline dTdxi(i, j, k) = (T[i + 1, j + 1, k + 1] - T[i, j + 1, k + 1]) * _dx
    @inline dTdyi(i, j, k) = (T[i + 1, j + 1, k + 1] - T[i + 1, j, k + 1]) * _dy
    @inline dTdzi(i, j, k) = (T[i + 1, j + 1, k + 1] - T[i + 1, j + 1, k]) * _dz

    if i ≤ size(qTx, 1) && j ≤ size(qTx, 2) && k ≤ size(qTx, 3)
        qTx[i, j, k] = -compute_diffusivity(rheology, args) * dTdxi(i, j, k)
    end

    if i ≤ size(qTy, 1) && j ≤ size(qTy, 2) && k ≤ size(qTy, 3)
        qTy[i, j, k] = -compute_diffusivity(rheology, args) * dTdyi(i, j, k)
    end

    if i ≤ size(qTz, 1) && j ≤ size(qTz, 2) && k ≤ size(qTz, 3)
        qTz[i, j, k] = -compute_diffusivity(rheology, args) * dTdzi(i, j, k)
    end

    return nothing
end

# @parallel function update_T!(T, qTx, qTy, qTz, _dx, _dy, _dz, dt)
#     @inn(T) = @inn(T) - (@d_xa(qTx)*_dx + @d_ya(qTy)*_dy + @d_za(qTz)*_dz) * dt
#     return nothing
# end

@parallel_indices (i, j, k) function advect_T!(
    dT_dt, qTx, qTy, qTz, T, Vx, Vy, Vz, _dx, _dy, _dz
)
    if i ≤ size(dT_dt, 1) && j ≤ size(dT_dt, 2) && k ≤ size(dT_dt, 3)
        dT_dt[i, j, k] =
            -(
                (qTx[i + 1, j, k] - qTx[i, j, k]) * _dx +
                (qTy[i, j + 1, k] - qTy[i, j, k]) * _dy +
                (qTz[i, j, k + 1] - qTz[i, j, k]) * _dz
            ) -
            (Vx[i + 1, j + 1, k + 1] > 0) *
            Vx[i + 1, j + 1, k + 1] *
            (T[i + 1, j + 1, k + 1] - T[i, j + 1, k + 1]) *
            _dx -
            (Vx[i + 2, j + 1, k + 1] < 0) *
            Vx[i + 2, j + 1, k + 1] *
            (T[i + 2, j + 1, k + 1] - T[i + 1, j + 1, k + 1]) *
            _dx -
            (Vy[i + 1, j + 1, k + 1] > 0) *
            Vy[i + 1, j + 1, k + 1] *
            (T[i + 1, j + 1, k + 1] - T[i + 1, j, k + 1]) *
            _dy -
            (Vy[i + 1, j + 2, k + 1] < 0) *
            Vy[i + 1, j + 2, k + 1] *
            (T[i + 1, j + 2, k + 1] - T[i + 1, j + 1, k + 1]) *
            _dy -
            (Vz[i + 1, j + 1, k + 1] > 0) *
            Vz[i + 1, j + 1, k + 1] *
            (T[i + 1, j + 1, k + 1] - T[i + 1, j + 1, k]) *
            _dz -
            (Vz[i + 1, j + 1, k + 2] < 0) *
            Vz[i + 1, j + 1, k + 2] *
            (T[i + 1, j + 1, k + 2] - T[i + 1, j + 1, k + 1]) *
            _dz
    end
    return nothing
end

@parallel function advect_T!(dT_dt, qTx, qTy, qTz, _dx, _dy, _dz)
    @all(dT_dt) = -(@d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz)
    return nothing
end

@parallel function update_T!(T, dT_dt, dt)
    @inn(T) = @inn(T) + @all(dT_dt) * dt
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,3}},
    thermal_bc::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx,
        thermal.qTy,
        thermal.qTz,
        thermal.T,
        thermal_parameters.κ,
        _dx,
        _dy,
        _dz,
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, thermal.qTz, _dx, _dy, _dz)
    @hide_communication b_width begin # communication/computation overlap
        @parallel update_T!(thermal.T, thermal.dT_dt, dt)
        update_halo!(thermal.T)
    end
    thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,3}},
    thermal_bc::NamedTuple,
    stokes,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx,
        thermal.qTy,
        thermal.qTz,
        thermal.T,
        thermal_parameters.κ,
        _dx,
        _dy,
        _dz,
    )
    @hide_communication b_width begin # communication/computation overlap
        @parallel advect_T!(
            thermal.dT_dt,
            thermal.qTx,
            thermal.qTy,
            thermal.qTz,
            thermal.T,
            stokes.V.Vx,
            stokes.V.Vy,
            stokes.V.Vz,
            _dx,
            _dy,
            _dz,
        )
        update_halo!(thermal.T)
    end
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# GEOPARAMS VERSION

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::NamedTuple,
    rheology::MaterialParams,
    args::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _dx, _dy, _dz = inv.(di)
    nx, ny, nz = size(thermal.T)

    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1), 1:(nz - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, rheology, args, _dx, _dy, _dz
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, thermal.qTz, _dx, _dy, _dz)
    @hide_communication b_width begin # communication/computation overlap
        @parallel update_T!(thermal.T, thermal.dT_dt, dt)
        update_halo!(thermal.T)
    end
    thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# upwind advection 
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::NamedTuple,
    stokes,
    rheology::MaterialParams,
    args::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _dx, _dy, _dz = inv.(di)
    nx, ny, nz = size(thermal.T)

    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1), 1:(nz - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, rheology, args, _dx, _dy, _dz
    )
    @hide_communication b_width begin # communication/computation overlap
        @parallel advect_T!(
            thermal.dT_dt,
            thermal.qTx,
            thermal.qTy,
            thermal.qTz,
            thermal.T,
            stokes.V.Vx,
            stokes.V.Vy,
            stokes.V.Vz,
            _dx,
            _dy,
            _dz,
        )
        update_halo!(thermal.T)
    end
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

end
