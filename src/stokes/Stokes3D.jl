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

export solve!, pureshear_bc!

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

@parallel_indices (i, j, k) function compute_∇V!(∇V, Vx, Vy, Vz, _dx, _dy, _dz)
    @inbounds ∇V[i, j, k] =
        _dx * (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) +
        _dy * (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) +
        _dz * (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k])
    return nothing
end

@parallel_indices (i, j, k) function compute_strain_rate!(
    ∇V, εxx, εyy, εzz, εyz, εxz, εxy, Vx, Vy, Vz, _dx, _dy, _dz
)
    @inbounds begin
        # normal components are all located @ cell centers
        if all((i, j, k) .≤ size(εxx))
            ∇Vijk = ∇V[i, j, k] * inv(3)
            # Compute ε_xx
            εxx[i, j, k] = _dx * (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) - ∇Vijk
            # Compute ε_yy
            εyy[i, j, k] = _dy * (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) - ∇Vijk
            # Compute ε_zz
            εzz[i, j, k] = _dz * (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k]) - ∇Vijk
        end
        # Compute ε_yz
        if all((i, j, k) .≤ size(εyz))
            εyz[i, j, k] =
                0.5 * (
                    _dz * (Vy[i + 1, j, k + 1] - Vy[i + 1, j, k]) +
                    _dy * (Vz[i + 1, j + 1, k] - Vz[i + 1, j, k])
                )
        end
        # Compute ε_xz
        if all((i, j, k) .≤ size(εxz))
            εxz[i, j, k] =
                0.5 * (
                    _dz * (Vx[i, j + 1, k + 1] - Vx[i, j + 1, k]) +
                    _dx * (Vz[i + 1, j + 1, k] - Vz[i, j + 1, k])
                )
        end
        # Compute ε_xy
        if all((i, j, k) .≤ size(εxy))
            εxy[i, j, k] =
                0.5 * (
                    _dy * (Vx[i, j + 1, k + 1] - Vx[i, j, k + 1]) +
                    _dx * (Vy[i + 1, j, k + 1] - Vy[i, j, k + 1])
                )
        end
    end
    return nothing
end

@parallel function compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (@all(K) * dt)
    @all(P) = @all(P) + @all(RP) * inv(inv(r * inv(θ_dτ) * @all(η)) + inv(@all(K) * dt))
    return nothing
end

## Compressible - GeoParams
@parallel function compute_P!(P, P_old, RP, ∇V, η, K::Number, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (K * dt)
    @all(P) = @all(P) + @all(RP) * inv(inv(r * inv(θ_dτ) * @all(η)) + inv(K * dt))
    return nothing
end

@parallel_indices (i, j, k) function compute_V!(
    Vx,
    Vy,
    Vz,
    Rx,
    Ry,
    Rz,
    P,
    fx,
    fy,
    fz,
    τxx,
    τyy,
    τzz,
    τyz,
    τxz,
    τxy,
    ητ,
    ηdτ,
    _dx,
    _dy,
    _dz,
)
    #! format: off
    Base.@propagate_inbounds @inline harm_x(x) = 2.0 * inv(inv(x[i + 1, j, k]) + inv(x[i, j, k]))
    Base.@propagate_inbounds @inline harm_y(x) = 2.0 * inv(inv(x[i, j + 1, k]) + inv(x[i, j, k]))
    Base.@propagate_inbounds @inline harm_z(x) = 2.0 * inv(inv(x[i, j, k + 1]) + inv(x[i, j, k]))
    Base.@propagate_inbounds @inline av_x(x)   = 0.5 * (x[i + 1, j, k] + x[i, j, k])
    Base.@propagate_inbounds @inline av_y(x)   = 0.5 * (x[i, j + 1, k] + x[i, j, k])
    Base.@propagate_inbounds @inline av_z(x)   = 0.5 * (x[i, j, k + 1] + x[i, j, k])
    Base.@propagate_inbounds @inline dx(x)     = x[i + 1, j, k] - x[i, j, k]
    Base.@propagate_inbounds @inline dy(x)     = x[i, j + 1, k] - x[i, j, k]
    Base.@propagate_inbounds @inline dz(x)     = x[i, j, k + 1] - x[i, j, k]
    #! format: on

    @inbounds begin
        if all((i, j, k) .< size(Vx) .- 1)
            Rx_ijk =
                _dx * (τxx[i + 1, j, k] - τxx[i, j, k]) +
                _dy * (τxy[i + 1, j + 1, k] - τxy[i + 1, j, k]) +
                _dz * (τxz[i + 1, j, k + 1] - τxz[i + 1, j, k]) - _dx * dx(P) - av_x(fx)
            Vx[i + 1, j + 1, k + 1] += Rx_ijk * ηdτ / av_x(ητ)
            Rx[i, j, k] = Rx_ijk
        end
        if all((i, j, k) .< size(Vy) .- 1)
            Ry_ijk =
                _dx * (τxy[i + 1, j + 1, k] - τxy[i, j + 1, k]) +
                _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
                _dz * (τyz[i, j + 1, k + 1] - τyz[i, j + 1, k]) - _dy * dy(P) - av_y(fy)
            Vy[i + 1, j + 1, k + 1] += Ry_ijk * ηdτ / av_y(ητ)
            Ry[i, j, k] = Ry_ijk
        end
        if all((i, j, k) .< size(Vz) .- 1)
            Rz_ijk =
                _dx * (τxz[i + 1, j, k + 1] - τxz[i, j, k + 1]) +
                _dy * (τyz[i, j + 1, k + 1] - τyz[i, j, k + 1]) +
                _dz * (τzz[i, j, k + 1] - τzz[i, j, k]) - _dz * dz(P) - av_z(fz)
            Vz[i + 1, j + 1, k + 1] += Rz_ijk * ηdτ / av_z(ητ)
            Rz[i, j, k] = Rz_ijk
        end
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_τ!(
    τxx,
    τyy,
    τzz,
    τyz,
    τxz,
    τxy,
    τxx_o,
    τyy_o,
    τzz_o,
    τyz_o,
    τxz_o,
    τxy_o,
    εxx,
    εyy,
    εzz,
    εyz,
    εxz,
    εxy,
    η,
    G,
    dt,
    θ_dτ,
)
    #! format: off
    Base.@propagate_inbounds @inline function harm_xy(x)
        4.0 * inv(
            inv(x[i - 1, j - 1, k]) +
            inv(x[i - 1, j, k]) +
            inv(x[i, j - 1, k]) +
            inv(x[i, j, k])
        )
    end
    Base.@propagate_inbounds @inline function harm_xz(x)
        4.0 * inv(
            inv(x[i, j, k]) +
            inv(x[i - 1, j, k]) +
            inv(x[i, j, k - 1]) +
            inv(x[i - 1, j, k - 1])
        )
    end
    Base.@propagate_inbounds @inline function harm_yz(x)
        4.0 * inv(
            inv(x[i, j, k]) +
            inv(x[i, j - 1, k]) +
            inv(x[i, j, k - 1]) +
            inv(x[i, j - 1, k - 1])
        )
    end
    Base.@propagate_inbounds @inline function av_xy(x)
        0.25 * (x[i - 1, j - 1, k] + x[i - 1, j, k] + x[i, j - 1, k] + x[i, j, k])
    end
    Base.@propagate_inbounds @inline function av_xz(x)
        0.25 * (x[i, j, k] + x[i - 1, j, k] + x[i, j, k - 1] + x[i - 1, j, k - 1])
    end
    Base.@propagate_inbounds @inline function av_yz(x)
        0.25 * (x[i, j, k] + x[i, j - 1, k] + x[i, j, k - 1] + x[i, j - 1, k - 1])
    end
    Base.@propagate_inbounds @inline current(x) = x[i, j, k]
    #! format: on

    @inbounds begin
        if all((i, j, k) .≤ size(τxx))
            _Gdt = inv(current(G) * dt)
            # Compute τ_xx
            τxx[i, j, k] +=
                (
                    -(current(τxx) - current(τxx_o)) * current(η) / (current(G) * dt) -
                    current(τxx) + 2.0 * current(η) * current(εxx)
                ) / (θ_dτ + current(η) * _Gdt + 1.0)
            # Compute τ_yy
            τyy[i, j, k] +=
                (
                    -(current(τyy) - current(τyy_o)) * current(η) / (current(G) * dt) -
                    current(τyy) + 2.0 * current(η) * current(εyy)
                ) / (θ_dτ + current(η) * _Gdt + 1.0)
            # Compute τ_zz
            τzz[i, j, k] +=
                (
                    -(current(τzz) - current(τzz_o)) * current(η) / (current(G) * dt) -
                    current(τzz) + 2.0 * current(η) * current(εzz)
                ) / (θ_dτ + current(η) * _Gdt + 1.0)
        end
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            τxy[i, j, k] +=
                (
                    -(current(τxy) - current(τxy_o)) * harm_xy(η) / (harm_xy(G) * dt) -
                    current(τxy) + 2.0 * harm_xy(η) * current(εxy)
                ) / (θ_dτ + harm_xy(η) / (harm_xy(G) * dt) + 1.0)
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            τxz[i, j, k] +=
                (
                    -(current(τxz) - current(τxz_o)) * harm_xz(η) / (harm_xz(G) * dt) -
                    current(τxz) + 2.0 * harm_xz(η) * current(εxz)
                ) / (θ_dτ + harm_xz(η) / (harm_xz(G) * dt) + 1.0)
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            τyz[i, j, k] +=
                (
                    -(current(τyz) - current(τyz_o)) * harm_yz(η) / (harm_yz(G) * dt) -
                    current(τyz) + 2.0 * harm_yz(η) * current(εyz)
                ) / (θ_dτ + harm_yz(η) / (harm_yz(G) * dt) + 1.0)
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_τ_vertex!(
    τyz, τxz, τxy, τyz_o, τxz_o, τxy_o, εyz, εxz, εxy, η, G, dt, θ_dτ
)
    #! format: off
    @inline Base.@propagate_inbounds function av_xy(x)
        0.25 * (x[i - 1, j - 1, k] + x[i - 1, j, k] + x[i, j - 1, k] + x[i, j, k])
    end
    @inline Base.@propagate_inbounds function av_xz(x)
        0.25 * (x[i, j, k] + x[i - 1, j, k] + x[i, j, k - 1] + x[i - 1, j, k - 1])
    end
    @inline Base.@propagate_inbounds function av_yz(x)
        0.25 * (x[i, j, k] + x[i, j - 1, k] + x[i, j, k - 1] + x[i, j - 1, k - 1])
    end
    @inline Base.@propagate_inbounds current(x) = x[i, j, k]
    #! format: on

    _Gdt = inv(G * dt)
    @inbounds begin
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            τxy[i, j, k] +=
                (
                    -(current(τxy) - current(τxy_o)) * av_xy(η) * _Gdt - current(τxy) +
                    2.0 * av_xy(η) * current(εxy)
                ) * inv(θ_dτ + av_xy(η) * _Gdt + 1.0)
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            τxz[i, j, k] +=
                (
                    -(current(τxz) - current(τxz_o)) * av_xz(η) * _Gdt - current(τxz) +
                    2.0 * av_xz(η) * current(εxz)
                ) * inv(θ_dτ + av_xz(η) * _Gdt + 1.0)
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            τyz[i, j, k] +=
                (
                    -(current(τyz) - current(τyz_o)) * av_yz(η) * _Gdt - current(τyz) +
                    2.0 * av_yz(η) * current(εyz)
                ) * inv(θ_dτ + av_yz(η) * _Gdt + 1.0)
        end
    end
    return nothing
end

# visco-elasto-plastic with GeoParams
@parallel_indices (i, j, k) function compute_τ_gp!(
    τxx,
    τyy,
    τzz,
    τyz,
    τxz,
    τxy,
    τII,
    τxx_o,
    τyy_o,
    τzz_o,
    τyz_o,
    τxz_o,
    τxy_o,
    εxx,
    εyy,
    εzz,
    εyz,
    εxz,
    εxy,
    η,
    η_vep,
    z,
    T,
    rheology,
    dt,
    θ_dτ,
)
    # convinience closures
    @inline gather_yz(A) = _gather_yz(A, i, j, k)
    @inline gather_xz(A) = _gather_xz(A, i, j, k)
    @inline gather_xy(A) = _gather_xy(A, i, j, k)

    @inbounds begin
        # dτ_r = 1.0 / (θ_dτ + η[i, j, k] / (get_G(rheology[1]) * dt) + 1.0) # original
        dτ_r = 1.0 / (θ_dτ / η[i, j, k] + 1.0 / η_vep[i, j, k]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
        # Setup up input for GeoParams.jl
        T_cell =
            0.125 * (
                T[i, j, k] +
                T[i, j + 1, k] +
                T[i + 1, j, k] +
                T[i + 1, j + 1, k] +
                T[i, j, k + 1] +
                T[i, j + 1, k + 1] +
                T[i + 1, j, k + 1] +
                T[i + 1, j + 1, k + 1]
            )
        args = (; dt=dt, P=1e6 * (1 - z[k]), T=T_cell, τII_old=0.0)
        εij_p = (
            εxx[i, j, k] + 1e-25,
            εyy[i, j, k] + 1e-25,
            εzz[i, j, k] + 1e-25,
            gather_yz(εyz) .+ 1e-25,
            gather_xz(εxz) .+ 1e-25,
            gather_xy(εxy) .+ 1e-25,
        )
        τij_p_o = (
            τxx_o[i, j, k],
            τyy_o[i, j, k],
            τzz_o[i, j, k],
            gather_yz(τyz_o),
            gather_xz(τxz_o),
            gather_xy(τxy_o),
        )
        phases = (1, 1, 1, (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)) # for now hard-coded for a single phase
        # update stress and effective viscosity
        τij, τII[i, j, k], ηᵢ = compute_τij(rheology, εij_p, args, τij_p_o, phases)
        τ = ( # caching out improves a wee bit the performance
            τxx[i, j, k],
            τyy[i, j, k],
            τzz[i, j, k],
            τyz[i, j, k],
            τxz[i, j, k],
            τxy[i, j, k],
        )
        dτ_rηᵢ = dτ_r / ηᵢ
        τxx[i, j, k] += dτ_rηᵢ * (-τ[1] + τij[1])
        τyy[i, j, k] += dτ_rηᵢ * (-τ[2] + τij[2])
        τzz[i, j, k] += dτ_rηᵢ * (-τ[3] + τij[3])
        # τyz[i, j, k]  += dτ_rηᵢ * (-τ[4] + τij[4]) 
        # τxz[i, j, k]  += dτ_rηᵢ * (-τ[5] + τij[5]) 
        # τxy[i, j, k]  += dτ_rηᵢ * (-τ[6] + τij[6]) 
        η_vep[i, j, k] = ηᵢ
    end
    return nothing
end

# single phase visco-elasto-plastic flow
@parallel_indices (i, j, k) function compute_τ_nonlinear!(
    τxx,
    τyy,
    τzz,
    τyz, # @ cell centers
    τxz, # @ cell centers
    τxy, # @ cell centers
    τII,
    τxx_old,
    τyy_old,
    τzz_old,
    τyzv_old, # @ vertices
    τxzv_old, # @ vertices
    τxyv_old, # @ vertices
    εxx,
    εyy,
    εzz,
    εyzv, # @ vertices
    εxzv, # @ vertices
    εxyv, # @ vertices
    P,
    η,
    η_vep,
    λ,
    rheology,
    dt,
    θ_dτ,
)
    idx = i, j, k

    # numerics
    ηij = @inbounds η[idx...]
    _Gdt = inv(get_G(rheology[1]) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params(rheology[1])
    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    τ = τxx, τyy, τzz, τyz, τxz, τxy
    τ_old = τxx_old, τyy_old, τzz_old, τyzv_old, τxzv_old, τxyv_old
    ε = εxx, εyy, εzz, εyzv, εxzv, εxyv

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, idx...
    )

    return nothing
end

# multi phase visco-elasto-plastic flow, where phases are defined in the cell center
@parallel_indices (i, j, k) function compute_τ_nonlinear!(
    τxx,
    τyy,
    τzz,
    τyz, # @ cell centers
    τxz, # @ cell centers
    τxy, # @ cell centers
    τII,
    τxx_old,
    τyy_old,
    τzz_old,
    τyzv_old, # @ vertices
    τxzv_old, # @ vertices
    τxyv_old, # @ vertices
    εxx,
    εyy,
    εzz,
    εyzv, # @ vertices
    εxzv, # @ vertices
    εxyv, # @ vertices
    P,
    η,
    η_vep,
    λ,
    phase_center,
    rheology,
    dt,
    θ_dτ,
)
    idx = i, j, k

    # numerics
    ηij = @inbounds η[idx...]
    phase = @inbounds phase_center[idx...]
    _Gdt = inv(get_G(rheology[1]) * dt)
    dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)
    # get plastic paremeters (if any...)
    is_pl, C, sinϕ, η_reg = plastic_params(rheology[1], phase)
    plastic_parameters = (; is_pl, C, sinϕ, η_reg)

    τ = τxx, τyy, τzz, τyz, τxz, τxy
    τ_old = τxx_old, τyy_old, τzz_old, τyzv_old, τxzv_old, τxyv_old
    ε = εxx, εyy, εzz, εyzv, εxzv, εxyv

    _compute_τ_nonlinear!(
        τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, idx...
    )

    return nothing
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

            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            push!(norm_Rx, maximum(abs.(stokes.R.Rx)))
            push!(norm_Ry, maximum(abs.(stokes.R.Ry)))
            push!(norm_Rz, maximum(abs.(stokes.R.Rz)))
            push!(norm_∇V, maximum(abs.(stokes.R.RP)))
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
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            push!(norm_Rx, maximum(abs.(stokes.R.Rx)))
            push!(norm_Ry, maximum(abs.(stokes.R.Ry)))
            push!(norm_Rz, maximum(abs.(stokes.R.Rz)))
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
