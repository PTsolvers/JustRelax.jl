## UTILS

function stress(stokes::StokesArrays{ViscoElastic,A,B,C,D,nDim}) where {A,B,C,D,nDim}
    return stress(stokes.τ), stress(stokes.τ_o)
end

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

## 2D ELASTICITY MODULE

module Elasticity2D

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using JustRelax
using LinearAlgebra
using CUDA
using Printf

# using ..JustRelax: solve!
import JustRelax: stress, strain, elastic_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax: Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic
import JustRelax: compute_maxloc!, solve!

import ..Stokes2D: compute_P!, compute_V!, compute_strain_rate!

export solve!

## 2D ELASTIC KERNELS

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,2}) where {A,B,C,D}
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    @parallel update_τ_o!(τxx_o, τyy_o, τxy_o, τxx, τyy, τxy)
end

@parallel function update_τ_o!(τxx_o, τyy_o, τxy_o, τxx, τyy, τxy)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    return nothing
end

@parallel function compute_∇V!(∇V, Vx, Vy, _dx, _dy)
    @all(∇V) = @d_xi(Vx) * _dx + @d_yi(Vy) * _dy
    return nothing
end

@parallel function compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy)
    @all(εxx) = @d_xi(Vx) * _dx - @all(∇V) / 3.0
    @all(εyy) = @d_yi(Vy) * _dy - @all(∇V) / 3.0
    @all(εxy) = 0.5 * (@d_ya(Vx) * _dy + @d_xa(Vy) * _dx)
    return nothing
end

# Continuity equation

## Incompressible 
@parallel function compute_P!(P, RP, ∇V, η, r, θ_dτ)
    @all(RP) = -@all(∇V)
    @all(P) = @all(P) + @all(RP) * r / θ_dτ * @all(η)
    return nothing
end

## Compressible 
@parallel function compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (@all(K) * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (@all(K) * dt))
    return nothing
end

@parallel_indices (i, j) function compute_V_Res!(Vx, Vy, Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, ητ, ηdτ, _dx, _dy)

    # Again, indices i, j are captured by the closure
    @inline d_xa(A)  = (A[i + 1, j] - A[i, j]) * _dx
    @inline d_ya(A)  = (A[i, j + 1] - A[i, j]) * _dy
    @inline d_xi(A)  = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
    @inline d_yi(A)  = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
    @inline av_xa(A) = (A[i + 1, j] + A[i, j]) * 0.5
    @inline av_ya(A) = (A[i, j + 1] + A[i, j]) * 0.5

    if all( (i, j) .≤ size(Rx))
        @inbounds R = Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
        @inbounds Vx[i+1, j+1] += R * ηdτ / av_xa(ητ)

    end
    if all( (i, j) .≤ size(Ry))
        @inbounds R = Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
        @inbounds Vy[i+1, j+1] += R * ηdτ / av_ya(ητ)
    end
    return nothing
end

# Stress calculation

# viscous
@parallel function compute_τ!(τxx, τyy, τxyv, τxx_o, τyy_o, τxyv_o, εxx, εyy, εxyv, η, θ_dτ)
    @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0 * @all(η) * @all(εxx)) * 1.0 / (θ_dτ + 1.0)
    @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0 * @all(η) * @all(εyy)) * 1.0 / (θ_dτ + 1.0)
    @inn(τxyv) = @inn(τxyv) + (-@inn(τxyv) + 2.0 * @av(η) * @inn(εxyv)) * 1.0 / (θ_dτ + 1.0)

    return nothing
end

# visco-elastic
@parallel function compute_τ!(
    τxx, τyy, τxyv, τxx_o, τyy_o, τxyv_o, εxx, εyy, εxyv, η, G, θ_dτ, dt
)
    @all(τxx) =
        @all(τxx) +
        (
            -(@all(τxx) - @all(τxx_o)) * @all(η) / (@all(G) * dt) - @all(τxx) +
            2 * @all(η) * @all(εxx)
        ) * 1.0 / (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @all(τyy) =
        @all(τyy) +
        (
            -(@all(τyy) - @all(τyy_o)) * @all(η) / (@all(G) * dt) - @all(τyy) +
            2 * @all(η) * @all(εyy)
        ) * 1.0 / (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @inn(τxyv) =
        @inn(τxyv) +
        (
            -(@inn(τxyv) - @inn(τxyv_o)) * @av(η) / (@av(G) * dt) - @inn(τxyv) +
            2 * @av(η) * @inn(εxyv)
        ) * 1.0 / (θ_dτ + @av(η) / (@av(G) * dt) + 1.0)

    return nothing
end

## 2D VISCO-ELASTIC STOKES SOLVER 

# viscous solver
function JustRelax.solve!(
    stokes::StokesArrays{Viscous,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    freeslip,
    ρg,
    η,
    K,
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _dx, _dy = inv.(di)
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    εxx, εyy, εxy = strain(stokes)
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry, RP = stokes.R.Rx, stokes.R.Ry, stokes.R.RP
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    nx, ny = size(P)

    ρgx, ρgy = ρg
    P_old = deepcopy(P)

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)

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

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)

            @parallel compute_∇V!(∇V, Vx, Vy, _dx, _dy)
            @parallel compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy)
            @parallel compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
            @parallel compute_τ!(τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, θ_dτ)
            @parallel compute_V!(Vx, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (1:nx, 1:ny) compute_Res!(
                Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy
            )
            Vmin, Vmax = extrema(Vx)
            Pmin, Pmax = extrema(P)
            push!(norm_Rx, norm(Rx) / (Pmax - Pmin) * lx / sqrt(length(Rx)))
            push!(norm_Ry, norm(Ry) / (Pmax - Pmin) * lx / sqrt(length(Ry)))
            push!(norm_∇V, norm(∇V) / (Vmax - Vmin) * lx / sqrt(length(∇V)))
            err = max(norm_Rx[end], norm_Ry[end], norm_∇V[end])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if (verbose && err > ϵ) || (iter == iterMax)
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
    end

    update_τ_o!(stokes)

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
function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    freeslip,
    ρg,
    η,
    G,
    K,
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _dx, _dy = inv.(di)
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    εxx, εyy, εxy = strain(stokes)
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry, RP = stokes.R.Rx, stokes.R.Ry, stokes.R.RP
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    nx, ny = size(P)

    ρgx, ρgy = ρg
    P_old = deepcopy(P)

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    apply_free_slip!((freeslip_x=true, freeslip_y=true), ητ, ητ)

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
            @parallel compute_∇V!(∇V, Vx, Vy, _dx, _dy)
            @parallel compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy)
            @parallel compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
            @parallel compute_τ!(
                τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
            )
            @parallel (1:nx, 1:ny) compute_V_Res!(Vx, Vy, Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, ητ, ηdτ, _dx, _dy)

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            push!(norm_Rx, maximum(abs.(Rx)))
            push!(norm_Ry, maximum(abs.(Ry)))
            push!(norm_∇V, maximum(abs.(RP)))
            err = max(norm_Rx[end], norm_Ry[end], norm_∇V[end])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if (verbose && err > ϵ) || (iter == iterMax)
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
    end

    update_τ_o!(stokes)

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

end # END OF MODULE

## 3D ELASTICITY MODULE

module Elasticity3D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using JustRelax
using CUDA
using LinearAlgebra
using Printf

import JustRelax:
    stress, strain, elastic_iter_params!, PTArray, Velocity, SymmetricTensor, pureshear_bc!
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
    # unpack
    τ, τ_o = stress(stokes)
    τxx, τyy, τzz, τxy, τxz, τyz = τ
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o = τ_o
    # copy
    @parallel update_τ_o!(
        τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o, τxx, τyy, τzz, τxy, τxz, τyz
    )
end

# Probably WRONG INDICES
@parallel_indices (i, j, k) function compute_∇V!(∇V, Vx, Vy, Vz, _dx, _dy, _dz)
    @inbounds ∇V[i, j, k] =
        _dx * (Vx[i + 1, j, k] - Vx[i, j, k]) +
        _dy * (Vy[i, j + 1, k] - Vy[i, j, k]) +
        _dz * (Vz[i, j, k + 1] - Vz[i, j, k])

    return nothing
end

@parallel_indices (i, j, k) function compute_strain_rate!(
    ∇V, εxx, εyy, εzz, εyz, εxz, εxy, Vx, Vy, Vz, _dx, _dy, _dz
)
    @inline @inbounds next(A) = A[i + 1, j + 1, k + 1]
    @inline @inbounds current_x(x) = x[i, j + 1, k + 1]
    @inline @inbounds current_y(x) = x[i + 1, j, k + 1]
    @inline @inbounds current_z(x) = x[i + 1, j + 1, k]

    # normal components are all located @ cell centers
    if all((i, j, k) .≤ size(εxx))
        # Compute ε_xx
        εxx[i, j, k] = _dx * (next(Vx) - current_x(Vx)) - current_x(∇V) / 3.0
    end
    if all((i, j, k) .≤ size(εyy))
        # Compute ε_yy
        εyy[i, j, k] = _dy * (next(Vy) - current_y(Vy)) - current_y(∇V) / 3.0
    end
    if all((i, j, k) .≤ size(εzz))
        # Compute ε_zz
        εzz[i, j, k] = _dz * (next(Vz) - current_z(Vz)) - current_z(∇V) / 3.0
    end
    # Compute ε_xy
    if all((i, j, k) .≤ size(εxy))
        εxy[i, j, k] =
            0.5 * (_dy * (next(Vx) - current_y(Vx)) + _dx * (next(Vy) - current_x(Vy)))
    end
    # Compute ε_xz
    if all((i, j, k) .≤ size(εxz))
        εxz[i, j, k] =
            0.5 * (_dz * (next(Vx) - current_z(Vx)) + _dx * (next(Vz) - current_x(Vz)))
    end
    # Compute ε_yz
    if all((i, j, k) .≤ size(εyz))
        εyz[i, j, k] =
            0.5 * (_dz * (next(Vy) - current_z(Vy)) + _dy * (next(Vz) - current_y(Vz)))
    end

    return nothing
end

@parallel function compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (@all(K) * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (@all(K) * dt))
    return nothing
end

@parallel_indices (i, j, k) function compute_V!(
    Vx,
    Vy,
    Vz,
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
    nx_1,
    nx_2,
    ny_1,
    ny_2,
    nz_1,
    nz_2,
)
    @inline @inbounds next(A) = A[i + 1, j + 1, k + 1]
    @inline @inbounds next_x(A) = A[i + 1, j, k]
    @inline @inbounds next_y(A) = A[i, j + 1, k]
    @inline @inbounds next_z(A) = A[i, j, k + 1]
    @inline @inbounds current(x) = x[i, j, k]
    @inline @inbounds current_x(x) = x[i, j + 1, k + 1]
    @inline @inbounds current_y(x) = x[i + 1, j, k + 1]
    @inline @inbounds current_z(x) = x[i + 1, j + 1, k]
    @inline @inbounds function harm_xi(x)
        2.0 / (1.0 / x[i, j + 1, k + 1] + 1.0 / x[i + 1, j + 1, k + 1])
    end
    @inline @inbounds function harm_yi(x)
        2.0 / (1.0 / x[i + 1, j, k + 1] + 1.0 / x[i + 1, j + 1, k + 1])
    end
    @inline @inbounds function harm_zi(x)
        2.0 / (1.0 / x[i + 1, j + 1, k] + 1.0 / x[i + 1, j + 1, k + 1])
    end

    if (i ≤ nx_1) && (j ≤ ny_2) && (k ≤ nz_2)
        Vx[i + 1, j + 1, k + 1] +=
            (
                _dx * (next_x(τxx) - current(τxx)) +
                _dy * (next_y(τxy) - current(τxy)) +
                _dz * (next_z(τxz) - current(τxz)) - _dx * (next(P) - current_x(P)) +
                harm_xi(fx)
            ) * ηdτ / harm_xi(ητ)
    end
    if (i ≤ nx_2) && (j ≤ ny_1) && (k ≤ nz_2)
        Vy[i + 1, j + 1, k + 1] +=
            (
                _dx * (next_x(τxy) - current(τxy)) +
                _dy * (next_y(τyy) - current(τyy)) +
                _dz * (next_z(τyz) - current(τyz)) - _dy * (next(P) - current_y(P)) +
                harm_yi(fy)
            ) * ηdτ / harm_yi(ητ)
    end
    if (i ≤ nx_2) && (j ≤ ny_2) && (k ≤ nz_1)
        Vz[i + 1, j + 1, k + 1] +=
            (
                _dx * (next_x(τxz) - current(τxz)) +
                _dy * (next_y(τyz) - current(τyz)) +
                _dz * (next_z(τzz) - current(τzz)) - _dz * (next(P) - current_z(P)) +
                harm_zi(fz)
            ) * ηdτ / harm_zi(ητ)
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
    @inline @inbounds function harm_xyi(x)
        4.0 / (
            1.0 / x[i, j, k + 1] +
            1.0 / x[i + 1, j, k + 1] +
            1.0 / x[i, j + 1, k + 1] +
            1.0 / x[i + 1, j + 1, k + 1]
        )
    end
    @inline @inbounds function harm_xzi(x)
        4.0 / (
            1.0 / x[i, j + 1, k] +
            1.0 / x[i + 1, j + 1, k] +
            1.0 / x[i, j + 1, k + 1] +
            1.0 / x[i + 1, j + 1, k + 1]
        )
    end
    @inline @inbounds function harm_yzi(x)
        4.0 / (
            1.0 / x[i + 1, j, k] +
            1.0 / x[i + 1, j + 1, k] +
            1.0 / x[i + 1, j, k + 1] +
            1.0 / x[i + 1, j + 1, k + 1]
        )
    end
    @inline @inbounds current(x) = x[i, j, k]
    @inline @inbounds current_x(x) = x[i, j + 1, k + 1]
    @inline @inbounds current_y(x) = x[i + 1, j, k + 1]
    @inline @inbounds current_z(x) = x[i + 1, j + 1, k]

    # Compute τ_xx
    if all((i, j, k) .≤ size(τxx))
        τxx[i, j, k] +=
            (
                -(current(τxx) - current(τxx_o)) * current_x(η) / (current_x(G) * dt) -
                current(τxx) + 2.0 * current_x(η) * current(εxx)
            ) / (θ_dτ + current_x(η) / (current_x(G) * dt) + 1.0)
    end
    # Compute τ_yy
    if all((i, j, k) .≤ size(τyy))
        τyy[i, j, k] +=
            (
                -(current(τyy) - current(τyy_o)) * current_y(η) / (current_y(G) * dt) -
                current(τyy) + 2.0 * current_y(η) * current(εyy)
            ) / (θ_dτ + current_y(η) / (current_y(G) * dt) + 1.0)
    end
    # Compute τ_zz
    if all((i, j, k) .≤ size(τzz))
        τzz[i, j, k] +=
            (
                -(current(τzz) - current(τzz_o)) * current_z(η) / (current_z(G) * dt) -
                current(τzz) + 2.0 * current_z(η) * current(εzz)
            ) / (θ_dτ + current_z(η) / (current_z(G) * dt) + 1.0)
    end
    # Compute τ_xy
    if all((i, j, k) .≤ size(τxy))
        τxy[i, j, k] +=
            (
                -(current(τxy) - current(τxy_o)) * harm_xyi(η) / (harm_xyi(G) * dt) -
                current(τxy) + 2.0 * harm_xyi(η) * current(εxy)
            ) / (θ_dτ + harm_xyi(η) / (harm_xyi(G) * dt) + 1.0)
    end
    # Compute τ_xz
    if all((i, j, k) .≤ size(τxz))
        τxz[i, j, k] +=
            (
                -(current(τxz) - current(τxz_o)) * harm_xzi(η) / (harm_xzi(G) * dt) -
                current(τxz) + 2.0 * harm_xzi(η) * current(εxz)
            ) / (θ_dτ + harm_xzi(η) / (harm_xzi(G) * dt) + 1.0)
    end
    # Compute τ_yz
    if all((i, j, k) .≤ size(τyz))
        τyz[i, j, k] +=
            (
                -(current(τyz) - current(τyz_o)) * harm_yzi(η) / (harm_yzi(G) * dt) -
                current(τyz) + 2.0 * harm_yzi(η) * current(εyz)
            ) / (θ_dτ + harm_yzi(η) / (harm_yzi(G) * dt) + 1.0)
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_Res!(
    Rx, Ry, Rz, fx, fy, fz, P, τxx, τyy, τzz, τxy, τxz, τyz, _dx, _dy, _dz
)
    @inline @inbounds function harm_xi(x)
        2.0 / (1.0 / x[i, j + 1, k + 1] + 1.0 / x[i + 1, j + 1, k + 1])
    end
    @inline @inbounds function harm_yi(x)
        2.0 / (1.0 / x[i + 1, j, k + 1] + 1.0 / x[i + 1, j + 1, k + 1])
    end
    @inline @inbounds function harm_zi(x)
        2.0 / (1.0 / x[i + 1, j + 1, k] + 1.0 / x[i + 1, j + 1, k + 1])
    end

    if all((i, j, k) .≤ size(Rx))
        Rx[i, j, k] =
            _dx * (τxx[i + 1, j, k] - τxx[i, j, k]) +
            _dy * (τxy[i, j + 1, k] - τxy[i, j, k]) +
            _dz * (τxz[i, j, k + 1] - τxz[i, j, k]) -
            _dx * (P[i + 1, j + 1, k + 1] - P[i, j + 1, k + 1]) + harm_xi(fx)
    end
    if all((i, j, k) .≤ size(Ry))
        Ry[i, j, k] =
            _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
            _dx * (τxy[i + 1, j, k] - τxy[i, j, k]) +
            _dz * (τyz[i, j, k + 1] - τyz[i, j, k]) -
            _dy * (P[i + 1, j + 1, k + 1] - P[i + 1, j, k + 1]) + harm_yi(fy)
    end
    if all((i, j, k) .≤ size(Rz))
        Rz[i, j, k] =
            _dz * (τzz[i, j, k + 1] - τzz[i, j, k]) +
            _dx * (τxz[i + 1, j, k] - τxz[i, j, k]) +
            _dy * (τyz[i, j + 1, k] - τyz[i, j, k]) -
            _dz * (P[i + 1, j + 1, k + 1] - P[i + 1, j + 1, k]) + harm_zi(fz)
    end

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
    li::NTuple{3,T},
    freeslip,
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

    ## UNPACK

    # phsysics
    lx, ly, lz = li # gravitational forces
    fx, fy, fz = ρg # gravitational forces
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz # velocity
    P, ∇V = stokes.P, stokes.∇V # pressure and velociity divergence
    εxx, εyy, εzz, εxy, εxz, εyz = strain(stokes)
    τ, τ_o = stress(stokes) # stress 
    τxx, τyy, τzz, τxy, τxz, τyz = τ
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o = τ_o
    P_o = deepcopy(P)
    # solver related
    Rx, Ry, Rz, RP = stokes.R.Rx, stokes.R.Ry, stokes.R.Rz, stokes.R.RP
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    # geometry
    _dx, _dy, _dz = @. 1 / di
    lx, ly, lz = li
    nx, ny, nz = size(P)
    nx_1, nx_2, ny_1, ny_2, nz_1, nz_2 = nx - 1, nx - 2, ny - 1, ny - 2, nz - 1, nz - 2

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end
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
            @parallel (1:nx, 1:ny, 1:nz) compute_∇V!(∇V, Vx, Vy, Vz, _dx, _dy, _dz)
            @parallel compute_P!(P, P_o, RP, ∇V, η, K, dt, r, θ_dτ)
            @parallel (1:nx, 1:ny, 1:nz) compute_strain_rate!(
                ∇V, εxx, εyy, εzz, εyz, εxz, εxy, Vx, Vy, Vz, _dx, _dy, _dz
            )
            @parallel (1:nx, 1:ny, 1:nz) compute_τ!(
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
            # @hide_communication b_width begin # communication/computation overlap
            @parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) compute_V!(
                Vx,
                Vy,
                Vz,
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
                nx_1,
                nx_2,
                ny_1,
                ny_2,
                nz_1,
                nz_2,
            )
            #     update_halo!(Vx, Vy, Vz)
            # end

            apply_free_slip!(freeslip, Vx, Vy, Vz)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1

            wtime0 += @elapsed begin
                @parallel (1:(nx - 1), 1:(ny - 1), 1:(nz - 1)) compute_Res!(
                    Rx, Ry, Rz, fx, fy, fz, P, τxx, τyy, τzz, τxy, τxz, τyz, _dx, _dy, _dz
                )
            end

            # Vmin, Vmax = minimum_mpi(Vx), maximum_mpi(Vx)
            # Pmin, Pmax = minimum_mpi(P), maximum_mpi(P)
            # push!(norm_Rx, norm_mpi(Rx) / (Pmax - Pmin) * lx * _sqrt_len_Rx_g)
            # push!(norm_Ry, norm_mpi(Ry) / (Pmax - Pmin) * lx * _sqrt_len_Ry_g)
            # push!(norm_Rz, norm_mpi(Rz) / (Pmax - Pmin) * lx * _sqrt_len_Rz_g)
            # push!(norm_∇V, norm_mpi(RP) / (Vmax - Vmin) * lx * _sqrt_len_∇V_g)

            push!(norm_Rx, maximum(abs.(Rx)))
            push!(norm_Ry, maximum(abs.(Ry)))
            push!(norm_Rz, maximum(abs.(Rz)))
            push!(norm_∇V, maximum(abs.(RP)))
            err = maximum([norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont]])
            push!(
                err_evo1,
                maximum([norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont]]),
            )
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

end # END OF MODULE
