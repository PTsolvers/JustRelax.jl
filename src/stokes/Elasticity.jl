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

## 2D ELASTICITY MODULE

module Elasticity2D

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using JustRelax
using LinearAlgebra
using CUDA
using Printf

# using ..JustRelax: solve!
import JustRelax: stress, elastic_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax: Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic
import JustRelax: compute_maxloc!, solve!

import ..Stokes2D: compute_P!, compute_V!

export solve!

## 2D ELASTIC KERNELS

@parallel function compute_dV_elastic!(
    dVx::AbstractArray{T,2},
    dVy::AbstractArray{T,2},
    P::AbstractArray{T,2},
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    dτ_Rho::AbstractArray{T,2},
    ρg::AbstractArray{T,2},
    dx::T,
    dy::T,
) where {T}
    @all(dVx) = (@d_xi(τxx) / dx + @d_ya(τxy) / dy - @d_xi(P) / dx) * @harm_xi(dτ_Rho)
    @all(dVy) =
        (@d_yi(τyy) / dy + @d_xa(τxy) / dx - @d_yi(P) / dy - @harm_yi(ρg)) *
        @harm_yi(dτ_Rho)
    return nothing
end

@parallel function compute_Res!(
    Rx::AbstractArray{T,2},
    Ry::AbstractArray{T,2},
    P::AbstractArray{T,2},
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    ρg::AbstractArray{T,2},
    dx::T,
    dy::T,
) where {T}
    @all(Rx) = @d_xi(τxx) / dx + @d_ya(τxy) / dy - @d_xi(P) / dx
    @all(Ry) = @d_yi(τyy) / dy + @d_xa(τxy) / dx - @d_yi(P) / dy - @harm_yi(ρg)
    return nothing
end

@parallel function update_τ_o!(
    τxx_o::AbstractArray{T,2},
    τyy_o::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
) where {T}
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    return nothing
end

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,2}) where {A,B,C,D}
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    @parallel update_τ_o!(τxx_o, τyy_o, τxy_o, τxx, τyy, τxy)
end

macro Gr()
    return esc(:(@all(Gdτ) / (G * dt)))
end
macro av_Gr()
    return esc(:(@av(Gdτ) / (G * dt)))
end
macro harm_Gr()
    return esc(:(@harm(Gdτ) / (G * dt)))
end
@parallel function compute_τ!(
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    τxx_o::AbstractArray{T,2},
    τyy_o::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    Vx::AbstractArray{T,2},
    Vy::AbstractArray{T,2},
    η::AbstractArray{T,2},
    G::T,
    dt::T,
    dx::T,
    dy::T,
) where {T}
    @all(τxx) =
        (@all(τxx) + @all(τxx_o) * @Gr() + T(2) * @all(Gdτ) * (@d_xa(Vx) / dx)) /
        (one(T) + @all(Gdτ) / @all(η) + @Gr())
    @all(τyy) =
        (@all(τyy) + @all(τyy_o) * @Gr() + T(2) * @all(Gdτ) * (@d_ya(Vy) / dy)) /
        (one(T) + @all(Gdτ) / @all(η) + @Gr())
    @all(τxy) =
        (
            @all(τxy) +
            @all(τxy_o) * @harm_Gr() +
            T(2) * @harm(Gdτ) * (0.5 * (@d_yi(Vx) / dy + @d_xi(Vy) / dx))
        ) / (one(T) + @harm(Gdτ) / @harm(η) + @harm_Gr())
    return nothing
end

## 2D VISCO-ELASTIC STOKES SOLVER 

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    li::NTuple{2,T},
    max_li,
    freeslip,
    ρg,
    η,
    G,
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    dx, dy = di
    lx, ly = li
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ,
    pt_stokes.dτ_Rho, pt_stokes.ϵ, pt_stokes.Re, pt_stokes.r,
    pt_stokes.Vpdτ

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    # PT numerical coefficients
    @parallel elastic_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li)

    _sqrt_leng_Rx = one(T) / sqrt(length(Rx))
    _sqrt_leng_Ry = one(T) / sqrt(length(Ry))
    _sqrt_leng_∇V = one(T) / sqrt(length(∇V))

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel compute_P!(∇V, P, Vx, Vy, Gdτ, r, dx, dy)
            @parallel compute_τ!(
                τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdτ, Vx, Vy, η, G, dt, dx, dy
            )
            @parallel compute_dV_elastic!(dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρg, dx, dy)
            @parallel compute_V!(Vx, Vy, dVx, dVy)

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            wtime0 += @elapsed begin
                @parallel compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρg, dx, dy)
            end
            Vmin, Vmax = minimum(Vy), maximum(Vy)
            Pmin, Pmax = minimum(P), maximum(P)
            push!(norm_Rx, norm(Rx) / (Pmax - Pmin) * lx * _sqrt_leng_Rx)
            push!(norm_Ry, norm(Ry) / (Pmax - Pmin) * lx * _sqrt_leng_Ry)
            push!(norm_∇V, norm(∇V) / (Vmax - Vmin) * lx * _sqrt_leng_∇V)
            err = maximum([norm_Rx[cont], norm_Ry[cont], norm_∇V[cont]])
            push!(err_evo1, maximum([norm_Rx[cont], norm_Ry[cont], norm_∇V[cont]]))
            push!(err_evo2, iter)
            if verbose && (err < ϵ) || (iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_∇V[cont]
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
    stress, elastic_iter_params!, PTArray, Velocity, SymmetricTensor, pureshear_bc!
import JustRelax:
    Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic, IGG
import JustRelax: compute_maxloc!, solve!

export solve!, pureshear_bc!

@parallel_indices (ix, iy, iz) function update_τ_o!(
    τxx_o::AbstractArray{T,3},
    τyy_o::AbstractArray{T,3},
    τzz_o::AbstractArray{T,3},
    τxy_o::AbstractArray{T,3},
    τxz_o::AbstractArray{T,3},
    τyz_o::AbstractArray{T,3},
    τxx::AbstractArray{T,3},
    τyy::AbstractArray{T,3},
    τzz::AbstractArray{T,3},
    τxy::AbstractArray{T,3},
    τxz::AbstractArray{T,3},
    τyz::AbstractArray{T,3},
) where {T}
    if (ix ≤ size(τxx, 1) && iy ≤ size(τxx, 2) && iz ≤ size(τxx, 3))
        τxx_o[ix, iy, iz] = τxx[ix, iy, iz]
    end
    if (ix ≤ size(τyy, 1) && iy ≤ size(τyy, 2) && iz ≤ size(τyy, 3))
        τyy_o[ix, iy, iz] = τyy[ix, iy, iz]
    end
    if (ix ≤ size(τzz, 1) && iy ≤ size(τzz, 2) && iz ≤ size(τzz, 3))
        τzz_o[ix, iy, iz] = τzz[ix, iy, iz]
    end
    if (ix ≤ size(τxy, 1) && iy ≤ size(τxy, 2) && iz ≤ size(τxy, 3))
        τxy_o[ix, iy, iz] = τxy[ix, iy, iz]
    end
    if (ix ≤ size(τxz, 1) && iy ≤ size(τxz, 2) && iz ≤ size(τxz, 3))
        τxz_o[ix, iy, iz] = τxz[ix, iy, iz]
    end
    if (ix ≤ size(τyz, 1) && iy ≤ size(τyz, 2) && iz ≤ size(τyz, 3))
        τyz_o[ix, iy, iz] = τyz[ix, iy, iz]
    end
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

macro inn_yz_Gdτ(ix, iy, iz)
    return esc(:(Gdτ[$ix, $iy + 1, $iz + 1]))
end
macro inn_xz_Gdτ(ix, iy, iz)
    return esc(:(Gdτ[$ix + 1, $iy, $iz + 1]))
end
macro inn_xy_Gdτ(ix, iy, iz)
    return esc(:(Gdτ[$ix + 1, $iy + 1, $iz]))
end
macro inn_yz_η(ix, iy, iz)
    return esc(:(η[$ix, $iy + 1, $iz + 1]))
end
macro inn_xz_η(ix, iy, iz)
    return esc(:(η[$ix + 1, $iy, $iz + 1]))
end
macro inn_xy_η(ix, iy, iz)
    return esc(:(η[$ix + 1, $iy + 1, $iz]))
end
macro av_xyi_Gdτ(ix, iy, iz)
    return esc(
        :(
            (
                Gdτ[$ix, $iy, $iz + 1] +
                Gdτ[$ix + 1, $iy, $iz + 1] +
                Gdτ[$ix, $iy + 1, $iz + 1] +
                Gdτ[$ix + 1, $iy + 1, $iz + 1]
            ) * 0.25
        ),
    )
end
macro av_xzi_Gdτ(ix, iy, iz)
    return esc(
        :(
            (
                Gdτ[$ix, $iy + 1, $iz] +
                Gdτ[$ix + 1, $iy + 1, $iz] +
                Gdτ[$ix, $iy + 1, $iz + 1] +
                Gdτ[$ix + 1, $iy + 1, $iz + 1]
            ) * 0.25
        ),
    )
end
macro av_yzi_Gdτ(ix, iy, iz)
    return esc(
        :(
            (
                Gdτ[$ix + 1, $iy, $iz] +
                Gdτ[$ix + 1, $iy + 1, $iz] +
                Gdτ[$ix + 1, $iy, $iz + 1] +
                Gdτ[$ix + 1, $iy + 1, $iz + 1]
            ) * 0.25
        ),
    )
end

macro harm_xyi_Gdτ(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / Gdτ[$ix, $iy, $iz + 1] +
                1.0 / Gdτ[$ix + 1, $iy, $iz + 1] +
                1.0 / Gdτ[$ix, $iy + 1, $iz + 1] +
                1.0 / Gdτ[$ix + 1, $iy + 1, $iz + 1]
            ) * 4.0
        ),
    )
end
macro harm_xzi_Gdτ(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / Gdτ[$ix, $iy + 1, $iz] +
                1.0 / Gdτ[$ix + 1, $iy + 1, $iz] +
                1.0 / Gdτ[$ix, $iy + 1, $iz + 1] +
                1.0 / Gdτ[$ix + 1, $iy + 1, $iz + 1]
            ) * 4.0
        ),
    )
end
macro harm_yzi_Gdτ(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / Gdτ[$ix + 1, $iy, $iz] +
                1.0 / Gdτ[$ix + 1, $iy + 1, $iz] +
                1.0 / Gdτ[$ix + 1, $iy, $iz + 1] +
                1.0 / Gdτ[$ix + 1, $iy + 1, $iz + 1]
            ) * 4.0
        ),
    )
end

macro av_xyi_η(ix, iy, iz)
    return esc(
        :(
            (
                η[$ix, $iy, $iz + 1] +
                η[$ix + 1, $iy, $iz + 1] +
                η[$ix, $iy + 1, $iz + 1] +
                η[$ix + 1, $iy + 1, $iz + 1]
            ) * 0.25
        ),
    )
end
macro av_xzi_η(ix, iy, iz)
    return esc(
        :(
            (
                η[$ix, $iy + 1, $iz] +
                η[$ix + 1, $iy + 1, $iz] +
                η[$ix, $iy + 1, $iz + 1] +
                η[$ix + 1, $iy + 1, $iz + 1]
            ) * 0.25
        ),
    )
end
macro av_yzi_η(ix, iy, iz)
    return esc(
        :(
            (
                η[$ix + 1, $iy, $iz] +
                η[$ix + 1, $iy + 1, $iz] +
                η[$ix + 1, $iy, $iz + 1] +
                η[$ix + 1, $iy + 1, $iz + 1]
            ) * 0.25
        ),
    )
end

macro harm_xyi_η(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / η[$ix, $iy, $iz + 1] +
                1.0 / η[$ix + 1, $iy, $iz + 1] +
                1.0 / η[$ix, $iy + 1, $iz + 1] +
                1.0 / η[$ix + 1, $iy + 1, $iz + 1]
            ) * 4.0
        ),
    )
end
macro harm_xzi_η(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / η[$ix, $iy + 1, $iz] +
                1.0 / η[$ix + 1, $iy + 1, $iz] +
                1.0 / η[$ix, $iy + 1, $iz + 1] +
                1.0 / η[$ix + 1, $iy + 1, $iz + 1]
            ) * 4.0
        ),
    )
end
macro harm_yzi_η(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / η[$ix + 1, $iy, $iz] +
                1.0 / η[$ix + 1, $iy + 1, $iz] +
                1.0 / η[$ix + 1, $iy, $iz + 1] +
                1.0 / η[$ix + 1, $iy + 1, $iz + 1]
            ) * 4.0
        ),
    )
end

@parallel_indices (ix, iy, iz) function compute_P_τ!(
    P::AbstractArray{T,3},
    τxx::AbstractArray{T,3},
    τyy::AbstractArray{T,3},
    τzz::AbstractArray{T,3},
    τxy::AbstractArray{T,3},
    τxz::AbstractArray{T,3},
    τyz::AbstractArray{T,3},
    τxx_o::AbstractArray{T,3},
    τyy_o::AbstractArray{T,3},
    τzz_o::AbstractArray{T,3},
    τxy_o::AbstractArray{T,3},
    τxz_o::AbstractArray{T,3},
    τyz_o::AbstractArray{T,3},
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    η::AbstractArray{T,3},
    Gdτ::AbstractArray{T,3},
    r::T,
    G::T,
    dt::M,
    _dx::T,
    _dy::T,
    _dz::T,
) where {T,M}
    # Compute pressure
    if (ix ≤ size(P, 1) && iy ≤ size(P, 2) && iz ≤ size(P, 3))
        P[ix, iy, iz] =
            P[ix, iy, iz] -
            r *
            Gdτ[ix, iy, iz] *
            (
                _dx * (Vx[ix + 1, iy, iz] - Vx[ix, iy, iz]) +
                _dy * (Vy[ix, iy + 1, iz] - Vy[ix, iy, iz]) +
                _dz * (Vz[ix, iy, iz + 1] - Vz[ix, iy, iz])
            )
    end
    # Compute τ_xx
    if (ix ≤ size(τxx, 1) && iy ≤ size(τxx, 2) && iz ≤ size(τxx, 3))
        τxx[ix, iy, iz] =
            (
                τxx[ix, iy, iz] / @inn_yz_Gdτ(ix, iy, iz) +
                τxx_o[ix, iy, iz] / G / dt +
                T(2) * (_dx * (Vx[ix + 1, iy + 1, iz + 1] - Vx[ix, iy + 1, iz + 1]))
            ) / (one(T) / @inn_yz_Gdτ(ix, iy, iz) + one(T) / @inn_yz_η(ix, iy, iz))
    end
    # Compute τ_yy
    if (ix ≤ size(τyy, 1) && iy ≤ size(τyy, 2) && iz ≤ size(τyy, 3))
        τyy[ix, iy, iz] =
            (
                τyy[ix, iy, iz] / @inn_xz_Gdτ(ix, iy, iz) +
                τyy_o[ix, iy, iz] / G / dt +
                T(2) * (_dy * (Vy[ix + 1, iy + 1, iz + 1] - Vy[ix + 1, iy, iz + 1]))
            ) / (one(T) / @inn_xz_Gdτ(ix, iy, iz) + one(T) / @inn_xz_η(ix, iy, iz))
    end
    # Compute τ_zz
    if (ix ≤ size(τzz, 1) && iy ≤ size(τzz, 2) && iz ≤ size(τzz, 3))
        τzz[ix, iy, iz] =
            (
                τzz[ix, iy, iz] / @inn_xy_Gdτ(ix, iy, iz) +
                τzz_o[ix, iy, iz] / G / dt +
                T(2) * (_dz * (Vz[ix + 1, iy + 1, iz + 1] - Vz[ix + 1, iy + 1, iz]))
            ) / (one(T) / @inn_xy_Gdτ(ix, iy, iz) + one(T) / @inn_xy_η(ix, iy, iz))
    end
    # Compute τ_xy
    if (ix ≤ size(τxy, 1) && iy ≤ size(τxy, 2) && iz ≤ size(τxy, 3))
        τxy[ix, iy, iz] =
            (
                τxy[ix, iy, iz] / @harm_xyi_Gdτ(ix, iy, iz) +
                τxy_o[ix, iy, iz] / G / dt +
                T(2) * (
                    0.5 * (
                        _dy * (Vx[ix + 1, iy + 1, iz + 1] - Vx[ix + 1, iy, iz + 1]) +
                        _dx * (Vy[ix + 1, iy + 1, iz + 1] - Vy[ix, iy + 1, iz + 1])
                    )
                )
            ) / (one(T) / @harm_xyi_Gdτ(ix, iy, iz) + one(T) / @harm_xyi_η(ix, iy, iz))
    end
    # Compute τ_xz
    if (ix ≤ size(τxz, 1) && iy ≤ size(τxz, 2) && iz ≤ size(τxz, 3))
        τxz[ix, iy, iz] =
            (
                τxz[ix, iy, iz] / @harm_xzi_Gdτ(ix, iy, iz) +
                τxz_o[ix, iy, iz] / G / dt +
                T(2) * (
                    0.5 * (
                        _dz * (Vx[ix + 1, iy + 1, iz + 1] - Vx[ix + 1, iy + 1, iz]) +
                        _dx * (Vz[ix + 1, iy + 1, iz + 1] - Vz[ix, iy + 1, iz + 1])
                    )
                )
            ) / (one(T) / @harm_xzi_Gdτ(ix, iy, iz) + one(T) / @harm_xzi_η(ix, iy, iz))
    end
    # Compute τ_yz
    if (ix ≤ size(τyz, 1) && iy ≤ size(τyz, 2) && iz ≤ size(τyz, 3))
        τyz[ix, iy, iz] =
            (
                τyz[ix, iy, iz] / @harm_yzi_Gdτ(ix, iy, iz) +
                τyz_o[ix, iy, iz] / G / dt +
                T(2) * (
                    0.5 * (
                        _dz * (Vy[ix + 1, iy + 1, iz + 1] - Vy[ix + 1, iy + 1, iz]) +
                        _dy * (Vz[ix + 1, iy + 1, iz + 1] - Vz[ix + 1, iy, iz + 1])
                    )
                )
            ) / (one(T) / @harm_yzi_Gdτ(ix, iy, iz) + one(T) / @harm_yzi_η(ix, iy, iz))
    end
    return nothing
end

macro av_xi_dτ_Rho(ix, iy, iz)
    return esc(:((dτ_Rho[$ix, $iy + 1, $iz + 1] + dτ_Rho[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end
macro av_yi_dτ_Rho(ix, iy, iz)
    return esc(:((dτ_Rho[$ix + 1, $iy, $iz + 1] + dτ_Rho[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end
macro av_zi_dτ_Rho(ix, iy, iz)
    return esc(:((dτ_Rho[$ix + 1, $iy + 1, $iz] + dτ_Rho[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end
macro av_xi_ρg(ix, iy, iz)
    return esc(:((fx[$ix, $iy + 1, $iz + 1] + fx[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end
macro av_yi_ρg(ix, iy, iz)
    return esc(:((fy[$ix + 1, $iy, $iz + 1] + fy[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end
macro av_zi_ρg(ix, iy, iz)
    return esc(:((fz[$ix + 1, $iy + 1, $iz] + fz[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end

macro harm_xi_dτ_Rho(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / dτ_Rho[$ix, $iy + 1, $iz + 1] +
                1.0 / dτ_Rho[$ix + 1, $iy + 1, $iz + 1]
            ) * 2.0
        ),
    )
end
macro harm_yi_dτ_Rho(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / dτ_Rho[$ix + 1, $iy, $iz + 1] +
                1.0 / dτ_Rho[$ix + 1, $iy + 1, $iz + 1]
            ) * 2.0
        ),
    )
end
macro harm_zi_dτ_Rho(ix, iy, iz)
    return esc(
        :(
            1.0 / (
                1.0 / dτ_Rho[$ix + 1, $iy + 1, $iz] +
                1.0 / dτ_Rho[$ix + 1, $iy + 1, $iz + 1]
            ) * 2.0
        ),
    )
end
macro harm_xi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fx[$ix, $iy + 1, $iz + 1] + 1.0 / fx[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end
macro harm_yi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fy[$ix + 1, $iy, $iz + 1] + 1.0 / fy[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end
macro harm_zi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fz[$ix + 1, $iy + 1, $iz] + 1.0 / fz[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end

@parallel_indices (ix, iy, iz) function compute_V!(
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    P::AbstractArray{T,3},
    fx::AbstractArray{T,3},
    fy::AbstractArray{T,3},
    fz::AbstractArray{T,3},
    τxx::AbstractArray{T,3},
    τyy::AbstractArray{T,3},
    τzz::AbstractArray{T,3},
    τxy::AbstractArray{T,3},
    τxz::AbstractArray{T,3},
    τyz::AbstractArray{T,3},
    dτ_Rho::AbstractArray{T,3},
    _dx::T,
    _dy::T,
    _dz::T,
    nx_1::N,
    nx_2::N,
    ny_1::N,
    ny_2::N,
    nz_1::N,
    nz_2::N,
) where {T,N}
    if (ix ≤ nx_1) && (iy ≤ ny_2) && (iz ≤ nz_2)
        Vx[ix + 1, iy + 1, iz + 1] =
            Vx[ix + 1, iy + 1, iz + 1] +
            (
                _dx * (τxx[ix + 1, iy, iz] - τxx[ix, iy, iz]) +
                _dy * (τxy[ix, iy + 1, iz] - τxy[ix, iy, iz]) +
                _dz * (τxz[ix, iy, iz + 1] - τxz[ix, iy, iz]) -
                _dx * (P[ix + 1, iy + 1, iz + 1] - P[ix, iy + 1, iz + 1]) +
                @harm_xi_ρg(ix, iy, iz)
            ) * @harm_xi_dτ_Rho(ix, iy, iz)
    end
    if (ix ≤ nx_2) && (iy ≤ ny_1) && (iz ≤ nz_2)
        Vy[ix + 1, iy + 1, iz + 1] =
            Vy[ix + 1, iy + 1, iz + 1] +
            (
                _dy * (τyy[ix, iy + 1, iz] - τyy[ix, iy, iz]) +
                _dx * (τxy[ix + 1, iy, iz] - τxy[ix, iy, iz]) +
                _dz * (τyz[ix, iy, iz + 1] - τyz[ix, iy, iz]) -
                _dy * (P[ix + 1, iy + 1, iz + 1] - P[ix + 1, iy, iz + 1]) +
                @harm_yi_ρg(ix, iy, iz)
            ) * @harm_yi_dτ_Rho(ix, iy, iz)
    end
    if (ix ≤ nx_2) && (iy ≤ ny_2) && (iz ≤ nz_1)
        Vz[ix + 1, iy + 1, iz + 1] =
            Vz[ix + 1, iy + 1, iz + 1] +
            (
                _dz * (τzz[ix, iy, iz + 1] - τzz[ix, iy, iz]) +
                _dx * (τxz[ix + 1, iy, iz] - τxz[ix, iy, iz]) +
                _dy * (τyz[ix, iy + 1, iz] - τyz[ix, iy, iz]) -
                _dz * (P[ix + 1, iy + 1, iz + 1] - P[ix + 1, iy + 1, iz]) +
                @harm_zi_ρg(ix, iy, iz)
            ) * @harm_zi_dτ_Rho(ix, iy, iz)
    end

    return nothing
end

macro av_xi_ρg(ix, iy, iz)
    return esc(:((fx[$ix, $iy + 1, $iz + 1] + fx[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end
macro av_yi_ρg(ix, iy, iz)
    return esc(:((fy[$ix + 1, $iy, $iz + 1] + fy[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end
macro av_zi_ρg(ix, iy, iz)
    return esc(:((fz[$ix + 1, $iy + 1, $iz] + fz[$ix + 1, $iy + 1, $iz + 1]) * 0.5))
end

macro harm_xi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fx[$ix, $iy + 1, $iz + 1] + 1.0 / fx[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end
macro harm_yi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fy[$ix + 1, $iy, $iz + 1] + 1.0 / fy[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end
macro harm_zi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fz[$ix + 1, $iy + 1, $iz] + 1.0 / fz[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end

@parallel_indices (ix, iy, iz) function compute_Res!(
    ∇V::AbstractArray{T,3},
    Rx::AbstractArray{T,3},
    Ry::AbstractArray{T,3},
    Rz::AbstractArray{T,3},
    fx::AbstractArray{T,3},
    fy::AbstractArray{T,3},
    fz::AbstractArray{T,3},
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    P::AbstractArray{T,3},
    τxx::AbstractArray{T,3},
    τyy::AbstractArray{T,3},
    τzz::AbstractArray{T,3},
    τxy::AbstractArray{T,3},
    τxz::AbstractArray{T,3},
    τyz::AbstractArray{T,3},
    _dx::T,
    _dy::T,
    _dz::T,
) where {T}
    if (ix ≤ size(∇V, 1)) && (iy ≤ size(∇V, 2)) && (iz ≤ size(∇V, 3))
        ∇V[ix, iy, iz] =
            _dx * (Vx[ix + 1, iy, iz] - Vx[ix, iy, iz]) +
            _dy * (Vy[ix, iy + 1, iz] - Vy[ix, iy, iz]) +
            _dz * (Vz[ix, iy, iz + 1] - Vz[ix, iy, iz])
    end
    if (ix ≤ size(Rx, 1)) && (iy ≤ size(Rx, 2)) && (iz ≤ size(Rx, 3))
        Rx[ix, iy, iz] =
            _dx * (τxx[ix + 1, iy, iz] - τxx[ix, iy, iz]) +
            _dy * (τxy[ix, iy + 1, iz] - τxy[ix, iy, iz]) +
            _dz * (τxz[ix, iy, iz + 1] - τxz[ix, iy, iz]) -
            _dx * (P[ix + 1, iy + 1, iz + 1] - P[ix, iy + 1, iz + 1]) +
            @harm_xi_ρg(ix, iy, iz)
    end
    if (ix ≤ size(Ry, 1)) && (iy ≤ size(Ry, 2)) && (iz ≤ size(Ry, 3))
        Ry[ix, iy, iz] =
            _dy * (τyy[ix, iy + 1, iz] - τyy[ix, iy, iz]) +
            _dx * (τxy[ix + 1, iy, iz] - τxy[ix, iy, iz]) +
            _dz * (τyz[ix, iy, iz + 1] - τyz[ix, iy, iz]) -
            _dy * (P[ix + 1, iy + 1, iz + 1] - P[ix + 1, iy, iz + 1]) +
            @harm_yi_ρg(ix, iy, iz)
    end
    if (ix ≤ size(Rz, 1)) && (iy ≤ size(Rz, 2)) && (iz ≤ size(Rz, 3))
        Rz[ix, iy, iz] =
            _dz * (τzz[ix, iy, iz + 1] - τzz[ix, iy, iz]) +
            _dx * (τxz[ix + 1, iy, iz] - τxz[ix, iy, iz]) +
            _dy * (τyz[ix, iy + 1, iz] - τyz[ix, iy, iz]) -
            _dz * (P[ix + 1, iy + 1, iz + 1] - P[ix + 1, iy + 1, iz]) +
            @harm_zi_ρg(ix, iy, iz)
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
        -εbg * ((ix - 1) * dx - 0.5 * lx) for ix in 1:size(Vx, 1), iy in 1:size(Vx, 2),
        iz in 1:size(Vx, 3)
    ])
    return stokes.V.Vz .= PTArray([
        εbg * ((iz - 1) * dz - 0.5 * lz) for ix in 1:size(Vz, 1), iy in 1:size(Vz, 2),
        iz in 1:size(Vz, 3)
    ])
end

## 3D VISCO-ELASTIC STOKES SOLVER 

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,3},
    pt_stokes::PTStokesCoeffs,
    ni::NTuple{3,Integer},
    di::NTuple{3,T},
    li::NTuple{3,T},
    max_li,
    freeslip,
    ρg,
    η,
    G,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(1, 1, 1),
    verbose=true,
) where {A,B,C,D,T}

    ## UNPACK
    # geometry
    dx, dy, dz = di
    _dx, _dy, _dz = @. 1 / di
    lx, ly, lz = li
    nx, ny, nz = ni
    nx_1, nx_2, ny_1, ny_2, nz_1, nz_2 = nx - 1, nx - 2, ny - 1, ny - 2, nz - 1, nz - 2
    # phsysics
    fx, fy, fz = ρg # gravitational forces
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz # velocity
    P, ∇V = stokes.P, stokes.∇V  # pressure and velociity divergence
    τ, τ_o = stress(stokes) # stress 
    τxx, τyy, τzz, τxy, τxz, τyz = τ
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o = τ_o
    # solver related
    Rx, Ry, Rz = stokes.R.Rx, stokes.R.Ry, stokes.R.Rz
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ,
    pt_stokes.dτ_Rho, pt_stokes.ϵ, pt_stokes.Re, pt_stokes.r,
    pt_stokes.Vpdτ

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end
    @parallel (1:size(ητ, 2), 1:size(ητ, 3)) free_slip_x!(ητ)
    @parallel (1:size(ητ, 1), 1:size(ητ, 3)) free_slip_y!(ητ)
    @parallel (1:size(ητ, 1), 1:size(ητ, 2)) free_slip_z!(ητ)

    # PT numerical coefficients
    @parallel elastic_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li)

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

    # residual lengths
    _sqrt_len_Rx_g =
        1.0 / √(
            ((nx - 2 - 1) * igg.dims[1] + 2) *
            ((ny - 2 - 2) * igg.dims[2] + 2) *
            ((nz - 2 - 2) * igg.dims[3] + 2),
        )
    _sqrt_len_Ry_g =
        1.0 / √(
            ((nx - 2 - 2) * igg.dims[1] + 2) *
            ((ny - 2 - 1) * igg.dims[2] + 2) *
            ((nz - 2 - 2) * igg.dims[3] + 2),
        )
    _sqrt_len_Rz_g =
        1.0 / √(
            ((nx - 2 - 2) * igg.dims[1] + 2) *
            ((ny - 2 - 2) * igg.dims[2] + 2) *
            ((nz - 2 - 1) * igg.dims[3] + 2),
        )
    _sqrt_len_∇V_g =
        1.0 / √(
            ((nx - 2) * igg.dims[1] + 2) *
            ((ny - 2) * igg.dims[2] + 2) *
            ((nz - 2) * igg.dims[3] + 2),
        )

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel compute_P_τ!(
                P,
                τxx,
                τyy,
                τzz,
                τxy,
                τxz,
                τyz,
                τxx_o,
                τyy_o,
                τzz_o,
                τxy_o,
                τxz_o,
                τyz_o,
                Vx,
                Vy,
                Vz,
                η,
                Gdτ,
                r,
                G,
                dt,
                _dx,
                _dy,
                _dz,
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
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
                    τxy,
                    τxz,
                    τyz,
                    dτ_Rho,
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
                update_halo!(Vx, Vy, Vz)
            end
            apply_free_slip!(freeslip, Vx, Vy, Vz)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1

            wtime0 += @elapsed begin
                @parallel compute_Res!(
                    ∇V,
                    Rx,
                    Ry,
                    Rz,
                    fx,
                    fy,
                    fz,
                    Vx,
                    Vy,
                    Vz,
                    P,
                    τxx,
                    τyy,
                    τzz,
                    τxy,
                    τxz,
                    τyz,
                    _dx,
                    _dy,
                    _dz,
                )
            end

            Vmin, Vmax = minimum_mpi(Vx), maximum_mpi(Vx)
            Pmin, Pmax = minimum_mpi(P), maximum_mpi(P)
            push!(norm_Rx, norm_mpi(Rx) / (Pmax - Pmin) * lx * _sqrt_len_Rx_g)
            push!(norm_Ry, norm_mpi(Ry) / (Pmax - Pmin) * lx * _sqrt_len_Ry_g)
            push!(norm_Rz, norm_mpi(Rz) / (Pmax - Pmin) * lx * _sqrt_len_Rz_g)
            push!(norm_∇V, norm_mpi(∇V) / (Vmax - Vmin) * lx * _sqrt_len_∇V_g)
            err = maximum([norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont]])
            if isnan(err)
                error("NaN")
            end
            push!(
                err_evo1,
                maximum([norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont]]),
            )
            push!(err_evo2, iter)
            if igg.me == 0 && (verbose && (err < ϵ) || (iter == iterMax))
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
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations finished after $iter iterations")
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
