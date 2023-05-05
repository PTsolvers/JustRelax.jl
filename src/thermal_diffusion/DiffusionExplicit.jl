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

@inline function compute_diffusivity(rheology, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end

function compute_diffusivity(rheology, phase, args)
    return compute_conductivity(rheology, phase, args) *
           inv(compute_heatcapacity(rheology, phase, args) * compute_density(rheology, phase, args))
end

@inline function compute_diffusivity(rheology, ρ::Number, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * ρ)
end

function compute_diffusivity(rheology, ρ, phase, args)
    return compute_conductivity(rheology, phase, args) *
           inv(compute_heatcapacity(rheology, phase, args) * ρ)
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
    qTx, qTy, T, rheology, args, _dx, _dy
)
    i1, j1 = @add 1 i j # augment indices by 1
    nPx = size(args.P, 1)

    @inbounds if all((i, j) .≤ size(qTx))
        Tx = (T[i1, j1] + T[i, j1]) * 0.5
        Pvertex = (args.P[clamp(i - 1, 1, nPx), j1] + args.P[clamp(i - 1, 1, nPx), j]) * 0.5
        argsx = (; T=Tx, P=Pvertex)
        qTx[i, j] = -compute_diffusivity(rheology, argsx) * (T[i1, j1] - T[i, j1]) * _dx
    end

    @inbounds if all((i, j) .≤ size(qTy))
        Ty = (T[i1, j1] + T[i1, j]) * 0.5
        Pvertex = (args.P[clamp(i, 1, nPx), j] + args.P[clamp(i - 1, 1, nPx), j]) * 0.5
        argsy = (; T=Ty, P=Pvertex)
        qTy[i, j] = -compute_diffusivity(rheology, argsy) * (T[i1, j1] - T[i1, j]) * _dy
    end

    return nothing
end

@parallel_indices (i, j) function compute_flux!(
    qTx, qTy, T, phases, rheology, args, _dx, _dy
)
    i1, j1 = @add 1 i j # augment indices by 1
    nPx = size(args.P, 1)

    @inbounds if all((i, j) .≤ size(qTx))
        Tx = (T[i1, j1] + T[i, j1]) * 0.5
        Pvertex = (args.P[clamp(i - 1, 1, nPx), j1] + args.P[clamp(i - 1, 1, nPx), j]) * 0.5
        argsx = (; T=Tx, P=Pvertex)
        qTx[i, j] = -compute_diffusivity(rheology, phases[i, j], ntuple_idx(argsx, i, j)) * (T[i1, j1] - T[i, j1]) * _dx
    end

    @inbounds if all((i, j) .≤ size(qTy))
        Ty = (T[i1, j1] + T[i1, j]) * 0.5
        Pvertex = (args.P[clamp(i, 1, nPx), j] + args.P[clamp(i - 1, 1, nPx), j]) * 0.5
        argsy = (; T=Ty, P=Pvertex)
        qTy[i, j] = -compute_diffusivity(rrheology, phases[i, j], ntuple_idx(argsy, i, j)) * (T[i1, j1] - T[i1, j]) * _dy
    end

    return nothing
end

@parallel_indices (i, j) function advect_T!(dT_dt, qTx, qTy, T, Vx, Vy, _dx, _dy)
    if all((i, j) .≤ size(dT_dt))
        i1, j1 = @add 1 i j # augment indices by 1
        i2, j2 = @add 2 i j # augment indices by 2

        @inbounds begin
            Vxᵢⱼ = 0.5 * (Vx[i1, j2] + Vx[i, j2])
            Vyᵢⱼ = 0.5 * (Vy[i1, j2] + Vy[i1, j1])

            dT_dt[i, j] =
                -((qTx[i1, j] - qTx[i, j]) * _dx + (qTy[i, j1] - qTy[i, j]) * _dy) -
                (Vxᵢⱼ > 0) * Vxᵢⱼ * (T[i1, j1] - T[i, j1]) * _dx -
                (Vxᵢⱼ < 0) * Vxᵢⱼ * (T[i2, j1] - T[i1, j1]) * _dx -
                (Vyᵢⱼ > 0) * Vyᵢⱼ * (T[i1, j1] - T[i1, j]) * _dy -
                (Vyᵢⱼ < 0) * Vyᵢⱼ * (T[i1, j2] - T[i1, j1]) * _dy
        end
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
    thermal_bc::TemperatureBoundaryConditions,
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
    thermal_bcs!(thermal.T, thermal_bc)

    # thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# GEOPARAMS VERSION

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::TemperatureBoundaryConditions,
    rheology::NTuple{N,MaterialParams},
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
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_bcs!(thermal.T, thermal_bc)

    # thermal_boundary_conditions!(thermal_bc, thermal.T)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# Upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::TemperatureBoundaryConditions,
    stokes,
    rheology::NTuple{N,MaterialParams},
    args::NamedTuple,
    di::NTuple{2,_T},
    dt,
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
    thermal_bcs!(thermal.T, thermal_bc)

    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end


# Upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::TemperatureBoundaryConditions,
    stokes,
    phases,
    rheology::NTuple{N,MaterialParams},
    args::NamedTuple,
    di::NTuple{2,_T},
    dt,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)
    nx, ny = size(thermal.T)
    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, phases, rheology, args, _dx, _dy
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
    thermal_bcs!(thermal.T, thermal_bc)

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
    qTx, qTy, qTz, T, rheology, args, _dx, _dy, _dz
)
    i1, j1, k1 = (i, j, k) .+ 1  # augment indices by 1
    nx, ny, nz = size(args.P)

    @inbounds begin
        if all((i, j, k) .≤ size(qTx))
            Tx = (T[i1, j1, k1] + T[i, j1, k1]) * 0.5
            Pvertex = 0.0
            for jj in 0:1, kk in 0:1
                Pvertex += args.P[i, clamp(j + jj, 1, ny), clamp(k + kk, 1, nz)]
            end
            argsx = (; T=Tx, P=Pvertex * 0.25)
            qTx[i, j, k] =
                -compute_diffusivity(rheology, argsx) * (T[i1, j1, k1] - T[i, j1, k1]) * _dx
        end

        if all((i, j, k) .≤ size(qTy))
            Ty = (T[i1, j1, k1] + T[i1, j, k1]) * 0.5
            Pvertex = 0.0
            for kk in 0:1, ii in 0:1
                args.P[clamp(i + ii, 1, nx), j, clamp(k + kk, 1, nz)]
            end
            argsy = (; T=Ty, P=Pvertex * 0.25)
            qTy[i, j, k] =
                -compute_diffusivity(rheology, argsy) * (T[i1, j1, k1] - T[i1, j, k1]) * _dy
        end

        if all((i, j, k) .≤ size(qTz))
            Tz = (T[i1, j1, k1] + T[i1, j1, k]) * 0.5
            Pvertex = 0.0
            for jj in 0:1, ii in 0:1
                args.P[clamp(i + ii, 1, nx), clamp(j + jj, 1, ny), k]
            end
            argsz = (; T=Tz, P=Pvertex * 0.25)
            qTz[i, j, k] =
                -compute_diffusivity(rheology, argsz) * (T[i1, j1, k1] - T[i1, j1, k]) * _dz
        end
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_flux!(
    qTx, qTy, qTz, T, phases, rheology, args, _dx, _dy, _dz
)
    i1, j1, k1 = (i, j, k) .+ 1  # augment indices by 1
    nx, ny, nz = size(args.P)

    @inbounds begin
        if all((i, j, k) .≤ size(qTx))
            Tx = (T[i1, j1, k1] + T[i, j1, k1]) * 0.5
            Pvertex = 0.0
            for jj in 0:1, kk in 0:1
                Pvertex += args.P[i, clamp(j + jj, 1, ny), clamp(k + kk, 1, nz)]
            end
            argsx = (; T=Tx, P=Pvertex * 0.25)
            qTx[i, j, k] =
                -compute_diffusivity(rheology, phases[i, j, k], ntuple_idx(argsx, i, j, k)) * (T[i1, j1, k1] - T[i, j1, k1]) * _dx
        end

        if all((i, j, k) .≤ size(qTy))
            Ty = (T[i1, j1, k1] + T[i1, j, k1]) * 0.5
            Pvertex = 0.0
            for kk in 0:1, ii in 0:1
                args.P[clamp(i + ii, 1, nx), j, clamp(k + kk, 1, nz)]
            end
            argsy = (; T=Ty, P=Pvertex * 0.25)
            qTy[i, j, k] =
                -compute_diffusivity(rheology, phases[i, j, k], ntuple_idx(argsy, i, j, k)) * (T[i1, j1, k1] - T[i1, j, k1]) * _dy
        end

        if all((i, j, k) .≤ size(qTz))
            Tz = (T[i1, j1, k1] + T[i1, j1, k]) * 0.5
            Pvertex = 0.0
            for jj in 0:1, ii in 0:1
                args.P[clamp(i + ii, 1, nx), clamp(j + jj, 1, ny), k]
            end
            argsz = (; T=Tz, P=Pvertex * 0.25)
            qTz[i, j, k] =
                -compute_diffusivity(rheology, phases[i, j, k], ntuple_idx(argsz, i, j, k)) * (T[i1, j1, k1] - T[i1, j1, k]) * _dz
        end
    end

    return nothing
end

@parallel_indices (i, j, k) function advect_T!(
    dT_dt, qTx, qTy, qTz, T, Vx, Vy, Vz, _dx, _dy, _dz
)
    if all((i, j, k) .≤ size(dT_dt))
        i1, j1, k1 = (i, j, k) .+ 1 # augment indices by 1
        i2, j2, k2 = (i, j, k) .+ 2 # augment indices by 2

        @inbounds begin
            # Average velocityes at cell vertices
            Vxᵢⱼₖ =
                0.25 * (Vx[i1, j1, k1] + Vx[i1, j2, k1] + Vx[i1, j1, k2] + Vx[i1, j2, k2])
            Vyᵢⱼₖ =
                0.25 * (Vy[i1, j1, k1] + Vy[i2, j1, k1] + Vy[i1, j1, k2] + Vy[i2, j1, k2])
            Vzᵢⱼₖ =
                0.25 * (Vz[i1, j1, k1] + Vz[i2, j1, k1] + Vz[i1, j2, k1] + Vz[i2, j2, k1])

            # Cache out local temperature
            Tᵢⱼₖ = T[i1, j1, k1] # this should be moved to shared memory

            # Compute ∂T/∂t = ∇(-k∇T) - V*∇T
            dT_dt[i, j, k] =
                -(
                    (qTx[i1, j, k] - qTx[i, j, k]) * _dx +
                    (qTy[i, j1, k] - qTy[i, j, k]) * _dy +
                    (qTz[i, j, k1] - qTz[i, j, k]) * _dz
                ) - (Vxᵢⱼₖ > 0 ? Tᵢⱼₖ - T[i, j1, k1] : T[i2, j1, k1] - Tᵢⱼₖ) * Vxᵢⱼₖ * _dx -
                (Vyᵢⱼₖ > 0 ? Tᵢⱼₖ - T[i1, j, k1] : T[i1, j2, k1] - Tᵢⱼₖ) * Vyᵢⱼₖ * _dy -
                (Vzᵢⱼₖ > 0 ? Tᵢⱼₖ - T[i1, j1, k] : T[i1, j1, k2] - Tᵢⱼₖ) * Vzᵢⱼₖ * _dz
        end
    end
    return nothing
end

@parallel function advect_T!(dT_dt, qTx, qTy, qTz, _dx, _dy, _dz)
    @all(dT_dt) = -(@d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz)
    return nothing
end

@parallel_indices (i, j, k) function update_T!(T, dT_dt, dt)
    if all((i, j, k) .≤ size(dT_dt))
        @inbounds T[i + 1, j + 1, k + 1] = muladd(
            dT_dt[i, j, k], dt, T[i + 1, j + 1, k + 1]
        )
    end
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
    _di = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, thermal_parameters.κ, _di...
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, thermal.qTz, _di...)
    @hide_communication b_width begin # communication/computation overlap
        @parallel update_T!(thermal.T, thermal.dT_dt, dt)
        update_halo!(thermal.T)
    end
    thermal_bcs!(thermal.T, thermal_bc)

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
    _di = inv.(di)

    # copy thermal array from previous time step
    @copy thermal.Told thermal.T
    # compute flux
    @parallel compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, thermal_parameters.κ, _di...
    )
    # compute upwind advection
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
            _di...,
        )
        update_halo!(thermal.T)
    end
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_bcs!(thermal.T, thermal_bc)

    return nothing
end

# GEOPARAMS VERSION

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::TemperatureBoundaryConditions,
    rheology::NTuple{N,MaterialParams},
    args::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _di = inv.(di)
    ni = size(thermal.T)

    ## SOLVE HEAT DIFFUSION
    # copy thermal array from previous time step
    @copy thermal.Told thermal.T
    # compute flux
    @parallel (@idx ni .- 1) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, rheology, args, _di...
    )
    # compute upwind advection
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, thermal.qTz, _di...)
    # update thermal array
    @hide_communication b_width begin # communication/computation overlap
        @parallel update_T!(thermal.T, thermal.dT_dt, dt)
        update_halo!(thermal.T)
    end
    # apply boundary conditions
    thermal_bcs!(thermal.T, thermal_bc)
    @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# upwind advection 
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::TemperatureBoundaryConditions,
    stokes,
    rheology::NTuple{N,MaterialParams},
    args::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _di = inv.(di)
    ni = size(thermal.T)

    ## SOLVE HEAT DIFFUSION
    # copy thermal array from previous time step
    @copy thermal.Told thermal.T
    # compute upwind advection
    @parallel (@idx ni .- 1) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, rheology, args, _di...
    )
    # update thermal array
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
            _di...,
        )
        update_halo!(thermal.T)
    end
    # apply boundary conditions
    @hide_communication b_width begin # communication/computation overlap
        @parallel update_T!(thermal.T, thermal.dT_dt, dt)
        update_halo!(thermal.T)
    end
    thermal_bcs!(thermal.T, thermal_bc)

    # @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

# upwind advection 
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::TemperatureBoundaryConditions,
    stokes,
    phases,
    rheology::NTuple{N,MaterialParams},
    args::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _di = inv.(di)
    ni = size(thermal.T)

    ## SOLVE HEAT DIFFUSION
    # copy thermal array from previous time step
    @copy thermal.Told thermal.T
    # compute upwind advection
    @parallel (@idx ni .- 1) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, phases, rheology, args, _di...
    )
    # update thermal array
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
            _di...,
        )
        update_halo!(thermal.T)
    end
    # apply boundary conditions
    @hide_communication b_width begin # communication/computation overlap
        @parallel update_T!(thermal.T, thermal.dT_dt, dt)
        update_halo!(thermal.T)
    end
    thermal_bcs!(thermal.T, thermal_bc)

    # @. thermal.ΔT = thermal.T - thermal.Told

    return nothing
end

end
