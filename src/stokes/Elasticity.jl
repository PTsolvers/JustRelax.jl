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

# @parallel function compute_∇V!(∇V, Vx, Vy, _dx, _dy)
#     @all(∇V) = @d_xa(Vx) * _dx + @d_ya(Vy) * _dy
#     return nothing
# end

# @parallel function compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy)
#     @all(εxx) =        @d_xa(Vx) * _dx - @all(∇V) / 3.0
#     @all(εyy) =        @d_ya(Vy) * _dy - @all(∇V) / 3.0
#     @inn(εxy) = 0.5 * (@d_yi(Vx) * _dy + @d_xi(Vy) * _dx)
#     return nothing
# end

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

@parallel function compute_V!(Vx, Vy, P, τxx, τyy, τxyv, ηdτ, ρgx, ρgy, ητ, _dx, _dy)
    @inn(Vx) =
        @inn(Vx) +
        (-@d_xa(P) * _dx + @d_xa(τxx) * _dx + @d_yi(τxyv) * _dy - @av_xa(ρgx)) * ηdτ /
        @harm_xa(ητ)
    @inn(Vy) =
        @inn(Vy) +
        (-@d_ya(P) * _dy + @d_ya(τyy) * _dy + @d_xi(τxyv) * _dx - @av_ya(ρgy)) * ηdτ /
        @harm_ya(ητ)

    return nothing
end

# Stress calculation

# viscous
@parallel function compute_τ!(τxx, τyy, τxyv, τxx_o, τyy_o, τxyv_o, εxx, εyy, εxyv, η, θ_dτ)
    @all(τxx) =
        @all(τxx) +
        (-(@all(τxx) - @all(τxx_o)) - @all(τxx) + 2 * @all(η) * @all(εxx)) * 1.0 /
        (θ_dτ + 1.0)
    @all(τyy) =
        @all(τyy) +
        (-(@all(τyy) - @all(τyy_o)) - @all(τyy) + 2 * @all(η) * @all(εyy)) * 1.0 /
        (θ_dτ + 1.0)
    @inn(τxyv) =
        @inn(τxyv) +
        (-(@inn(τxyv) - @inn(τxyv_o)) - @inn(τxyv) + 2 * @harm(η) * @inn(εxyv)) * 1.0 /
        (θ_dτ + 1.0)

    return nothing
end

# visco-elastic
@parallel function compute_τ!(
    τxx, τyy, τxyv, τxx_o, τyy_o, τxyv_o, εxx, εyy, εxyv, η, G, θ_dτ, dt
)
    @all(τxx) =
        @all(τxx) +
        (-(@all(τxx) - @all(τxx_o)) - @all(τxx) + 2 * @all(η) * @all(εxx)) * 1.0 /
        (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @all(τyy) =
        @all(τyy) +
        (-(@all(τyy) - @all(τyy_o)) - @all(τyy) + 2 * @all(η) * @all(εyy)) * 1.0 /
        (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @inn(τxyv) =
        @inn(τxyv) +
        (-(@inn(τxyv) - @inn(τxyv_o)) - @inn(τxyv) + 2 * @harm(η) * @inn(εxyv)) * 1.0 /
        (θ_dτ + @harm(η) / (@harm(G) * dt) + 1.0)

    return nothing
end

# @parallel function compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy)
#     @all(Rx) = -@d_xa(P) * _dx + @d_xa(τxx) * _dx + @d_yi(τxy) * _dy - @av_xa(ρgx)
#     @all(Ry) = -@d_ya(P) * _dy + @d_ya(τyy) * _dy + @d_xi(τxy) * _dx - @av_ya(ρgy)

#     return nothing
# end

# Faster then macros version above (by c. 40%...)
@parallel_indices (i, j) function compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy)

    # Again, indices i, j are captured by the closure
    @inbounds @inline d_xa(A)  = (A[i+1, j  ] - A[i  , j  ]) * _dx 
    @inbounds @inline d_ya(A)  = (A[i  , j+1] - A[i  , j  ]) * _dy
    @inbounds @inline d_xi(A)  = (A[i+1, j+1] - A[i  , j+1]) * _dx
    @inbounds @inline d_yi(A)  = (A[i+1, j+1] - A[i+1, j  ]) * _dy
    @inbounds @inline av_xa(A) = (A[i+1, j  ] + A[i  , j  ]) * 0.5
    @inbounds @inline av_ya(A) = (A[i  , j+1] + A[i  , j  ]) * 0.5

    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        @inbounds Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        @inbounds Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
    end
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

            # push!(norm_Rx, maximum(abs.(Rx)))
            # push!(norm_Ry, maximum(abs.(Ry)))
            # push!(norm_∇V, maximum(abs.(RP)))

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
            @parallel compute_V!(Vx, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy)

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (1:nx, 1:ny) compute_Res!(
                Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy
            )

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

@parallel_indices function update_τ_o!(
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

macro inn_yz_Gdτ(i, j, k)
    return esc(:(Gdτ[$i, $j + 1, $k + 1]))
end
macro inn_xz_Gdτ(i, j, k)
    return esc(:(Gdτ[$i + 1, $j, $k + 1]))
end
macro inn_xy_Gdτ(i, j, k)
    return esc(:(Gdτ[$i + 1, $j + 1, $k]))
end
macro inn_yz_η(i, j, k)
    return esc(:(η[$i, $j + 1, $k + 1]))
end
macro inn_xz_η(i, j, k)
    return esc(:(η[$i + 1, $j, $k + 1]))
end
macro inn_xy_η(i, j, k)
    return esc(:(η[$i + 1, $j + 1, $k]))
end
macro av_xyi_Gdτ(i, j, k)
    return esc(
        :(
            (
                Gdτ[$i, $j, $k + 1] +
                Gdτ[$i + 1, $j, $k + 1] +
                Gdτ[$i, $j + 1, $k + 1] +
                Gdτ[$i + 1, $j + 1, $k + 1]
            ) * 0.25
        ),
    )
end
macro av_xzi_Gdτ(i, j, k)
    return esc(
        :(
            (
                Gdτ[$i, $j + 1, $k] +
                Gdτ[$i + 1, $j + 1, $k] +
                Gdτ[$i, $j + 1, $k + 1] +
                Gdτ[$i + 1, $j + 1, $k + 1]
            ) * 0.25
        ),
    )
end
macro av_yzi_Gdτ(i, j, k)
    return esc(
        :(
            (
                Gdτ[$i + 1, $j, $k] +
                Gdτ[$i + 1, $j + 1, $k] +
                Gdτ[$i + 1, $j, $k + 1] +
                Gdτ[$i + 1, $j + 1, $k + 1]
            ) * 0.25
        ),
    )
end

macro harm_xyi_G(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / G[$i, $j, $k + 1] +
                1.0 / G[$i + 1, $j, $k + 1] +
                1.0 / G[$i, $j + 1, $k + 1] +
                1.0 / G[$i + 1, $j + 1, $k + 1]
            ) * 4.0
        ),
    )
end
macro harm_xzi_G(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / G[$i, $j + 1, $k] +
                1.0 / G[$i + 1, $j + 1, $k] +
                1.0 / G[$i, $j + 1, $k + 1] +
                1.0 / G[$i + 1, $j + 1, $k + 1]
            ) * 4.0
        ),
    )
end
macro harm_yzi_G(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / G[$i + 1, $j, $k] +
                1.0 / G[$i + 1, $j + 1, $k] +
                1.0 / G[$i + 1, $j, $k + 1] +
                1.0 / G[$i + 1, $j + 1, $k + 1]
            ) * 4.0
        ),
    )
end

macro av_xyi_η(i, j, k)
    return esc(
        :(
            (
                η[$i, $j, $k + 1] +
                η[$i + 1, $j, $k + 1] +
                η[$i, $j + 1, $k + 1] +
                η[$i + 1, $j + 1, $k + 1]
            ) * 0.25
        ),
    )
end
macro av_xzi_η(i, j, k)
    return esc(
        :(
            (
                η[$i, $j + 1, $k] +
                η[$i + 1, $j + 1, $k] +
                η[$i, $j + 1, $k + 1] +
                η[$i + 1, $j + 1, $k + 1]
            ) * 0.25
        ),
    )
end
macro av_yzi_η(i, j, k)
    return esc(
        :(
            (
                η[$i + 1, $j, $k] +
                η[$i + 1, $j + 1, $k] +
                η[$i + 1, $j, $k + 1] +
                η[$i + 1, $j + 1, $k + 1]
            ) * 0.25
        ),
    )
end

macro harm_xyi_η(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / η[$i, $j, $k + 1] +
                1.0 / η[$i + 1, $j, $k + 1] +
                1.0 / η[$i, $j + 1, $k + 1] +
                1.0 / η[$i + 1, $j + 1, $k + 1]
            ) * 4.0
        ),
    )
end
macro harm_xzi_η(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / η[$i, $j + 1, $k] +
                1.0 / η[$i + 1, $j + 1, $k] +
                1.0 / η[$i, $j + 1, $k + 1] +
                1.0 / η[$i + 1, $j + 1, $k + 1]
            ) * 4.0
        ),
    )
end
macro harm_yzi_η(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / η[$i + 1, $j, $k] +
                1.0 / η[$i + 1, $j + 1, $k] +
                1.0 / η[$i + 1, $j, $k + 1] +
                1.0 / η[$i + 1, $j + 1, $k + 1]
            ) * 4.0
        ),
    )
end

@parallel_indices (i, j, k) function compute_τ!(
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
    εxx::AbstractArray{T,3},
    εyy::AbstractArray{T,3},
    εzz::AbstractArray{T,3},
    εyz::AbstractArray{T,3},
    εxz::AbstractArray{T,3},
    εxy::AbstractArray{T,3},
    G::T,
    dt::M,
    θ_dτ
) where {T,M}
    # Compute τ_xx
    if (i ≤ size(τxx, 1) && j ≤ size(τxx, 2) && k ≤ size(τxx, 3))
        τxx[i, j, k] =
            -((τxx[i, j, k] - τxx_o[i, j, k]) - τxx[i, j, k] + 2.0 * εxx[i, j, k]) /
            (θ_dτ + η[i, j, k] / (G[i, j, k] * dt) + 1.0)
    end
    # Compute τ_yy
    if (i ≤ size(τyy, 1) && j ≤ size(τyy, 2) && k ≤ size(τyy, 3))
        τyy[i, j, k] =
            -((τyy[i, j, k] - τyy_o[i, j, k]) - τyy[i, j, k] + 2.0 * εyy[i, j, k]) /
            (θ_dτ + η[i, j, k] / (G[i, j, k] * dt) + 1.0)
    end
    # Compute τ_zz
    if (i ≤ size(τzz, 1) && j ≤ size(τzz, 2) && k ≤ size(τzz, 3))
        τzz[i, j, k] =
            -((τzz[i, j, k] - τzz_o[i, j, k]) - τzz[i, j, k] + 2.0 * εzz[i, j, k]) /
            (θ_dτ + η[i, j, k] / (G[i, j, k] * dt) + 1.0)
    end
    # Compute τ_xy
    if (i ≤ size(τxy, 1) && j ≤ size(τxy, 2) && k ≤ size(τxy, 3))
        τxy[i, j, k] =
            -((τxy[i, j, k] - τxy_o[i, j, k]) - τxy[i, j, k] + 2.0 * εxy[i, j, k]) /
            (θ_dτ + @harm_xyi_η / (@harm_xyi_G * dt) + 1.0)
    end
    # Compute τ_xz
    if (i ≤ size(τxz, 1) && j ≤ size(τxz, 2) && k ≤ size(τxz, 3))
        τxz[i, j, k] =
            -((τxz[i, j, k] - τxz_o[i, j, k]) - τxz[i, j, k] + 2.0 * εxz[i, j, k]) /
            (θ_dτ + @harm_xzi_η / (@harm_xzi_G * dt) + 1.0)
    end
    # Compute τ_yz
    if (i ≤ size(τyz, 1) && j ≤ size(τyz, 2) && k ≤ size(τyz, 3))
        τyz[i, j, k] =
            -((τyz[i, j, k] - τyz_o[i, j, k]) - τyz[i, j, k] + 2.0 * εyz[i, j, k]) /
            (θ_dτ + @harm_yzi_η / (@harm_yzi_G * dt) + 1.0)
    end
    return nothing
end

@parallel function compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (@all(K) * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (@all(K) * dt))
    return nothing
end

@parallel (i, j, k) function compute_∇V!(∇V, Vx, Vy, Vz, _dx, _dy, _dz)

    ∇V[i, j, k] = 
        _dx * (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) +
        _dy * (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) + 
        _dz * (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k])

    return nothing
end

@parallel_indices (i, j, k) function compute_strain_rate!(
    ∇V::AbstractArray{T,3},
    εxx::AbstractArray{T,3},
    εyy::AbstractArray{T,3},
    εzz::AbstractArray{T,3},
    εyz::AbstractArray{T,3},
    εxz::AbstractArray{T,3},
    εxy::AbstractArray{T,3},
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    _dx::T,
    _dy::T,
    _dz::T,
) where {T}
    
    @inline next(A) = A[i + 1, j + 1, k + 1]

    # normal components are all located @ cell centers
    if (i ≤ size(εxx, 1) && j ≤ size(εxx, 2) && k ≤ size(εxx, 3))
        # Compute ε_xx
        εxx[i, j, k] =
            _dx * (next(Vx) - Vx[i    , j + 1, k + 1]) + ∇V[i, j, k] / 3.0
        # Compute ε_yy
        εyy[i, j, k] =
            _dy * (next(Vy) - Vy[i + 1, j    , k + 1]) + ∇V[i, j, k] / 3.0
        # Compute ε_zz
        εzz[i, j, k] =
            _dz * (next(Vz) - Vz[i + 1, j + 1, k    ]) + ∇V[i, j, k] / 3.0
    end
    # Compute ε_xy
    if (i ≤ size(εxy, 1) && j ≤ size(εxy, 2) && k ≤ size(εxy, 3))
        εxy[i, j, k] =
            0.5 * (
                _dy * (next(Vx) - Vx[i + 1, j, k + 1]) +
                _dx * (next(Vy) - Vy[i, j + 1, k + 1])
            )
    end
    # Compute ε_xz
    if (i ≤ size(εxz, 1) && j ≤ size(εxz, 2) && k ≤ size(εxz, 3))
        εxz[i, j, k] =
            0.5 * (
                _dz * (next(Vx) - Vx[i + 1, j + 1, k]) +
                _dx * (next(Vz) - Vz[i, j + 1, k + 1])
            )
    end
    # Compute ε_yz
    if (i ≤ size(εyz, 1) && j ≤ size(εyz, 2) && k ≤ size(εyz, 3))
        εyz[i, j, k] =
            0.5 * (
                _dz * (next(Vy) - Vy[i + 1, j + 1, k]) +
                _dy * (next(Vz) - Vz[i + 1, j, k + 1])
            )
    end

    return nothing
end

macro av_xi_dτ_Rho(i, j, k)
    return esc(:((dτ_Rho[$i, $j + 1, $k + 1] + dτ_Rho[$i + 1, $j + 1, $k + 1]) * 0.5))
end
macro av_yi_dτ_Rho(i, j, k)
    return esc(:((dτ_Rho[$i + 1, $j, $k + 1] + dτ_Rho[$i + 1, $j + 1, $k + 1]) * 0.5))
end
macro av_zi_dτ_Rho(i, j, k)
    return esc(:((dτ_Rho[$i + 1, $j + 1, $k] + dτ_Rho[$i + 1, $j + 1, $k + 1]) * 0.5))
end
macro av_xi_ρg(i, j, k)
    return esc(:((fx[$i, $j + 1, $k + 1] + fx[$i + 1, $j + 1, $k + 1]) * 0.5))
end
macro av_yi_ρg(i, j, k)
    return esc(:((fy[$i + 1, $j, $k + 1] + fy[$i + 1, $j + 1, $k + 1]) * 0.5))
end
macro av_zi_ρg(i, j, k)
    return esc(:((fz[$i + 1, $j + 1, $k] + fz[$i + 1, $j + 1, $k + 1]) * 0.5))
end

macro harm_xi_ητ(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / ητ[$i, $j + 1, $k + 1] +
                1.0 / ητ[$i + 1, $j + 1, $k + 1]
            ) * 2.0
        ),
    )
end
macro harm_yi_ητ(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / ητ[$i + 1, $j, $k + 1] +
                1.0 / ητ[$i + 1, $j + 1, $k + 1]
            ) * 2.0
        ),
    )
end
macro harm_zi_ητ(i, j, k)
    return esc(
        :(
            1.0 / (
                1.0 / ητ[$i + 1, $j + 1, $k] +
                1.0 / ητ[$i + 1, $j + 1, $k + 1]
            ) * 2.0
        ),
    )
end
macro harm_xi_ρg(i, j, k)
    return esc(
        :(
            1.0 / (1.0 / fx[$i, $j + 1, $k + 1] + 1.0 / fx[$i + 1, $j + 1, $k + 1]) *
            2.0
        ),
    )
end
macro harm_yi_ρg(i, j, k)
    return esc(
        :(
            1.0 / (1.0 / fy[$i + 1, $j, $k + 1] + 1.0 / fy[$i + 1, $j + 1, $k + 1]) *
            2.0
        ),
    )
end
macro harm_zi_ρg(i, j, k)
    return esc(
        :(
            1.0 / (1.0 / fz[$i + 1, $j + 1, $k] + 1.0 / fz[$i + 1, $j + 1, $k + 1]) *
            2.0
        ),
    )
end

@parallel_indices (i, j, k) function compute_V!(
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
    ητ::AbstractArray{T,3},
    ηdτ,
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
    if (i ≤ nx_1) && (j ≤ ny_2) && (k ≤ nz_2)
        Vx[i + 1, j + 1, k + 1] =
            Vx[i + 1, j + 1, k + 1] +
            (
                _dx * (τxx[i + 1, j, k] - τxx[i, j, k]) +
                _dy * (τxy[i, j + 1, k] - τxy[i, j, k]) +
                _dz * (τxz[i, j, k + 1] - τxz[i, j, k]) -
                _dx * (P[i + 1, j + 1, k + 1] - P[i, j + 1, k + 1]) +
                @harm_xi_ρg(i, j, k)
            ) * ηdτ / @harm_xi_ητ(ητ)
    end
    if (i ≤ nx_2) && (j ≤ ny_1) && (k ≤ nz_2)
        Vy[i + 1, j + 1, k + 1] =
            Vy[i + 1, j + 1, k + 1] +
            (
                _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
                _dx * (τxy[i + 1, j, k] - τxy[i, j, k]) +
                _dz * (τyz[i, j, k + 1] - τyz[i, j, k]) -
                _dy * (P[i + 1, j + 1, k + 1] - P[i + 1, j, k + 1]) +
                @harm_yi_ρg(i, j, k)
            ) * ηdτ / @harm_yi_ητ(ητ)
    end
    if (i ≤ nx_2) && (j ≤ ny_2) && (k ≤ nz_1)
        Vz[i + 1, j + 1, k + 1] =
            Vz[i + 1, j + 1, k + 1] +
            (
                _dz * (τzz[i, j, k + 1] - τzz[i, j, k]) +
                _dx * (τxz[i + 1, j, k] - τxz[i, j, k]) +
                _dy * (τyz[i, j + 1, k] - τyz[i, j, k]) -
                _dz * (P[i + 1, j + 1, k + 1] - P[i + 1, j + 1, k]) +
                @harm_zi_ρg(i, j, k)
            ) * ηdτ / @harm_zi_ητ(ητ)
    end

    return nothing
end

macro av_xi_ρg(i, j, k)
    return esc(:((fx[$i, $j + 1, $k + 1] + fx[$i + 1, $j + 1, $k + 1]) * 0.5))
end
macro av_yi_ρg(i, j, k)
    return esc(:((fy[$i + 1, $j, $k + 1] + fy[$i + 1, $j + 1, $k + 1]) * 0.5))
end
macro av_zi_ρg(i, j, k)
    return esc(:((fz[$i + 1, $j + 1, $k] + fz[$i + 1, $j + 1, $k + 1]) * 0.5))
end

macro harm_xi_ρg(i, j, k)
    return esc(
        :(
            1.0 / (1.0 / fx[$i, $j + 1, $k + 1] + 1.0 / fx[$i + 1, $j + 1, $k + 1]) *
            2.0
        ),
    )
end
macro harm_yi_ρg(i, j, k)
    return esc(
        :(
            1.0 / (1.0 / fy[$i + 1, $j, $k + 1] + 1.0 / fy[$i + 1, $j + 1, $k + 1]) *
            2.0
        ),
    )
end
macro harm_zi_ρg(i, j, k)
    return esc(
        :(
            1.0 / (1.0 / fz[$i + 1, $j + 1, $k] + 1.0 / fz[$i + 1, $j + 1, $k + 1]) *
            2.0
        ),
    )
end

@parallel_indices (i, j, k) function compute_Res!(
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
    if (i ≤ size(∇V, 1)) && (j ≤ size(∇V, 2)) && (k ≤ size(∇V, 3))
        ∇V[i, j, k] =
            _dx * (Vx[i + 1, j, k] - Vx[i, j, k]) +
            _dy * (Vy[i, j + 1, k] - Vy[i, j, k]) +
            _dz * (Vz[i, j, k + 1] - Vz[i, j, k])
    end
    if (i ≤ size(Rx, 1)) && (j ≤ size(Rx, 2)) && (k ≤ size(Rx, 3))
        Rx[i, j, k] =
            _dx * (τxx[i + 1, j, k] - τxx[i, j, k]) +
            _dy * (τxy[i, j + 1, k] - τxy[i, j, k]) +
            _dz * (τxz[i, j, k + 1] - τxz[i, j, k]) -
            _dx * (P[i + 1, j + 1, k + 1] - P[i, j + 1, k + 1]) +
            @harm_xi_ρg(i, j, k)
    end
    if (i ≤ size(Ry, 1)) && (j ≤ size(Ry, 2)) && (k ≤ size(Ry, 3))
        Ry[i, j, k] =
            _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
            _dx * (τxy[i + 1, j, k] - τxy[i, j, k]) +
            _dz * (τyz[i, j, k + 1] - τyz[i, j, k]) -
            _dy * (P[i + 1, j + 1, k + 1] - P[i + 1, j, k + 1]) +
            @harm_yi_ρg(i, j, k)
    end
    if (i ≤ size(Rz, 1)) && (j ≤ size(Rz, 2)) && (k ≤ size(Rz, 3))
        Rz[i, j, k] =
            _dz * (τzz[i, j, k + 1] - τzz[i, j, k]) +
            _dx * (τxz[i + 1, j, k] - τxz[i, j, k]) +
            _dy * (τyz[i, j + 1, k] - τyz[i, j, k]) -
            _dz * (P[i + 1, j + 1, k + 1] - P[i + 1, j + 1, k]) +
            @harm_zi_ρg(i, j, k)
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
    εxx, εyy, εzz, εxy, εxz, εyz = strain(stokes)
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
            @parallel compute_strain_rate!(
                εxx, εyy, εzz, εyz, εxz, εxy, Vx, Vy, Vz, _dx, _dy, _dz
            )
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
                εxx,
                εyy,
                εzz,
                εyz,
                εxz,
                εxy,
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
