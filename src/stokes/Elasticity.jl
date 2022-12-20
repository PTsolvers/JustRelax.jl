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
        (- @all(τxx) + 2 * @all(η) * @all(εxx)) * 1.0 /
        (θ_dτ + 1.0)
    @all(τyy) =
        @all(τyy) +
        (- @all(τyy) + 2 * @all(η) * @all(εyy)) * 1.0 /
        (θ_dτ + 1.0)
    @inn(τxyv) =
        @inn(τxyv) +
        (- @inn(τxyv) + 2 * @harm(η) * @inn(εxyv)) * 1.0 /
        (θ_dτ + 1.0)

    return nothing
end

# visco-elastic
@parallel function compute_τ!(
    τxx, τyy, τxyv, τxx_o, τyy_o, τxyv_o, εxx, εyy, εxyv, η, G, θ_dτ, dt
)
    @all(τxx) =
        @all(τxx) +
        (-(@all(τxx) - @all(τxx_o)) *  @all(η) / (@all(G) * dt) - @all(τxx) + 2 * @all(η) * @all(εxx)) * 1.0 /
        (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @all(τyy) =
        @all(τyy) +
        (-(@all(τyy) - @all(τyy_o)) *  @all(η) / (@all(G) * dt) - @all(τyy) + 2 * @all(η) * @all(εyy)) * 1.0 /
        (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @inn(τxyv) =
        @inn(τxyv) +
        (-(@inn(τxyv) - @inn(τxyv_o)) * @harm(η) / (@harm(G) * dt)- @inn(τxyv) + 2 * @harm(η) * @inn(εxyv)) * 1.0 /
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

@parallel function update_τ_o!(
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

@parallel_indices (i, j, k) function compute_τ!(τxx, τyy, τzz, τxy, τxz, τyz, τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o, εxx, εyy, εzz, εyz, εxz, εxy, η, G, dt, θ_dτ)

    @inline function harm_xyi(A)
        4.0 / (
            1.0 / A[i    , j    , k + 1] +
            1.0 / A[i + 1, j    , k + 1] +
            1.0 / A[i    , j + 1, k + 1] +
            1.0 / A[i + 1, j + 1, k + 1]
        )
    end
    @inline function harm_xzi(A)
        4.0 / (
            1.0 / A[i    , j + 1, k    ] +
            1.0 / A[i + 1, j + 1, k    ] +
            1.0 / A[i    , j + 1, k + 1] +
            1.0 / A[i + 1, j + 1, k + 1]
        )
    end
    @inline function harm_yzi(A)
        4.0 / (
            1.0 / A[i + 1, j    , k    ] +
            1.0 / A[i + 1, j + 1, k    ] +
            1.0 / A[i + 1, j    , k + 1] +
            1.0 / A[i + 1, j + 1, k + 1]
        )
    end

    @inline current_x(A) = A[i    , j + 1, k + 1]
    @inline current_y(A) = A[i + 1, j    , k + 1]
    @inline current_z(A) = A[i + 1, j + 1, k    ]

    # Compute τ_xx
    if all((i, j, k) .≤ size(τxx))
        τxx[i, j, k] +=
            (-(τxx[i, j, k] - τxx_o[i, j, k]) * current_x(η) / (current_x(G) * dt) - τxx[i, j, k] + 2.0 * current_x(η) * εxx[i, j, k]) /
            (θ_dτ + current_x(η) / (current_x(G) * dt) + 1.0)
    end
    if all((i, j, k) .≤ size(τyy))
        # Compute τ_yy
        τyy[i, j, k] +=
            (-(τyy[i, j, k] - τyy_o[i, j, k]) * current_y(η) / (current_y(G) * dt) - τyy[i, j, k] + 2.0 * current_y(η) * εyy[i, j, k]) /
            (θ_dτ + current_y(η) / (current_y(G) * dt) + 1.0)
    end
    if all((i, j, k) .≤ size(τzz))
        # Compute τ_zz
        τzz[i, j, k] +=
            (-(τzz[i, j, k] - τzz_o[i, j, k]) * current_z(η) / (current_z(G) * dt) - τzz[i, j, k] + 2.0 * current_z(η) * εzz[i, j, k]) /
            (θ_dτ + current_z(η) / (current_z(G) * dt) + 1.0)
    end
    # Compute τ_xy
    if all((i, j, k) .≤ size(τxy))
        τxy[i, j, k] +=
            (-(τxy[i, j, k] - τxy_o[i, j, k]) * harm_xyi(η) / (harm_xyi(G) * dt) - τxy[i, j, k] + 2.0 * harm_xyi(η) * εxy[i, j, k]) /
            (θ_dτ + harm_xyi(η) / (harm_xyi(G) * dt) + 1.0)
    end
    # Compute τ_xz
    if all((i, j, k) .≤ size(τxz))
        τxz[i, j, k] +=
            (-(τxz[i, j, k] - τxz_o[i, j, k]) * harm_xzi(η) / (harm_xzi(G) * dt) - τxz[i, j, k] + 2.0 * harm_xzi(η) * εxz[i, j, k]) /
            (θ_dτ + harm_xzi(η) / (harm_xzi(G) * dt) + 1.0)
    end
    # Compute τ_yz
    if all((i, j, k) .≤ size(τyz))
        τyz[i, j, k] +=
            (-(τyz[i, j, k] - τyz_o[i, j, k]) * harm_yzi(η) / (harm_yzi(G) * dt) - τyz[i, j, k] + 2.0 * harm_yzi(η) * εyz[i, j, k]) /
            (θ_dτ + harm_yzi(η) / (harm_yzi(G) * dt) + 1.0)
    end
    return nothing
end

@parallel function compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (@all(K) * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (@all(K) * dt))
    return nothing
end

@parallel_indices (i, j, k) function compute_∇V!(∇V, Vx, Vy, Vz, _dx, _dy, _dz)
    ∇V[i, j, k] = 
        _dx * (Vx[i + 1, j    , k    ] - Vx[i, j, k]) +
        _dy * (Vy[i    , j + 1, k    ] - Vy[i, j, k]) +
        _dz * (Vz[i    , j    , k + 1] - Vz[i, j, k])

    return nothing
end

@parallel_indices (i, j, k) function compute_strain_rate!(εxx, εyy, εzz, εyz, εxz, εxy, ∇V, Vx, Vy, Vz, _dx, _dy, _dz)
    
    @inline next(A)       = A[i + 1, j + 1, k + 1]
    @inline previous_x(A) = A[i    , j + 1, k + 1]
    @inline previous_y(A) = A[i + 1, j    , k + 1]
    @inline previous_z(A) = A[i + 1, j + 1, k    ]

    # Normal components
    if all((i, j, k) .≤ size(εxx)) 
        εxx[i, j, k] = _dx * (next(Vx) - previous_x(Vx)) - previous_x(∇V) / 3.0
    end
    # Compute ε_yy
    if all((i, j, k) .≤ size(εyy)) 
        εyy[i, j, k] = _dy * (next(Vy) - previous_y(Vy)) - previous_y(∇V) / 3.0
    end
    # Compute ε_zz
    if all((i, j, k) .≤ size(εzz)) 
        εzz[i, j, k] = _dz * (next(Vz) - previous_z(Vz)) - previous_z(∇V) / 3.0
    end
    # Shear components
    if all((i, j, k) .≤ size(εxy)) 
        εxy[i, j, k] =
            0.5 * (_dy * (next(Vx) - previous_y(Vx)) + _dx * (next(Vy) - previous_x(Vy)))
    end
    if all((i, j, k) .≤ size(εxz)) 
        εxz[i, j, k] =
            0.5 * (_dz * (next(Vx) - previous_z(Vx)) + _dx * (next(Vz) - previous_x(Vz)))
    end
    if all((i, j, k) .≤ size(εyz)) 
        εyz[i, j, k] =
            0.5 * (_dz * (next(Vy) - previous_z(Vy)) +  _dy * (next(Vz) - previous_y(Vz)))
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_V!(Vx, Vy, Vz, P, fx, fy, fz, τxx, τyy, τzz, τxy, τxz, τyz, ητ, ηdτ, _dx, _dy, _dz, nx_1, nx_2, ny_1, ny_2, nz_1, nz_2)
    
    @inline harm_xi(A) = 2.0 / (1.0 / A[i    , j + 1, k + 1] + 1.0 / A[i + 1, j + 1, k + 1])
    @inline harm_yi(A) = 2.0 / (1.0 / A[i + 1, j    , k + 1] + 1.0 / A[i + 1, j + 1, k + 1])
    @inline harm_zi(A) = 2.0 / (1.0 / A[i + 1, j + 1, k    ] + 1.0 / A[i + 1, j + 1, k + 1])
    @inline previous(A) = A[i, j, k]
    @inline next(A)    = A[i + 1, j + 1, k + 1]
    @inline next_x(A)  = A[i + 1, j    , k    ]
    @inline next_y(A)  = A[i    , j + 1, k    ]
    @inline next_z(A)  = A[i    , j    , k + 1]

    if (i ≤ nx_1) && (j ≤ ny_2) && (k ≤ nz_2)
        Vx[i + 1, j + 1, k + 1] =
                Vx[i + 1, j + 1, k + 1] +
            (
                _dx * (next_x(τxx) - previous(τxx)) +
                _dy * (next_y(τxy) - previous(τxy)) +
                _dz * (next_z(τxz) - previous(τxz)) -
                _dx * (next(P) - P[i, j + 1, k + 1]) +
                harm_xi(fx)
            ) * ηdτ / harm_xi(ητ)
    end
    if (i ≤ nx_2) && (j ≤ ny_1) && (k ≤ nz_2)
        Vy[i + 1, j + 1, k + 1] =
                Vy[i + 1, j + 1, k + 1] +
            (
                _dx * (next_x(τxy) - previous(τxy)) +
                _dy * (next_y(τyy) - previous(τyy)) +
                _dz * (next_z(τyz) - previous(τyz)) -
                _dy * (next(P) - P[i + 1, j, k + 1]) +
                harm_yi(fy)
            ) * ηdτ / harm_yi(ητ)
    end
    if (i ≤ nx_2) && (j ≤ ny_2) && (k ≤ nz_1)
        Vz[i + 1, j + 1, k + 1] =
                Vz[i + 1, j + 1, k + 1] +
            (
                _dx * (next_x(τxz) - previous(τxz)) +
                _dy * (next_y(τyz) - previous(τyz)) +
                _dz * (next_z(τzz) - previous(τzz)) -
                _dz * (next(P) - P[i + 1, j + 1, k]) +
                harm_zi(fz)
            ) * ηdτ / harm_zi(ητ)
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_Res!(Rx, Ry, Rz, fx, fy, fz, P, τxx, τyy, τzz, τxy, τxz, τyz, _dx, _dy, _dz)

    @inline harm_xi(A) = 2.0 / (1.0 / A[i    , j + 1, k + 1] + 1.0 / A[i + 1, j + 1, k + 1])
    @inline harm_yi(A) = 2.0 / (1.0 / A[i + 1, j    , k + 1] + 1.0 / A[i + 1, j + 1, k + 1])
    @inline harm_zi(A) = 2.0 / (1.0 / A[i + 1, j + 1, k    ] + 1.0 / A[i + 1, j + 1, k + 1])
    @inline current(A) = A[i    , j    , k    ]
    @inline next(A)    = A[i + 1, j + 1, k + 1]
    @inline next_x(A)  = A[i + 1, j    , k    ]
    @inline next_y(A)  = A[i    , j + 1, k    ]
    @inline next_z(A)  = A[i    , j    , k + 1]

    if all((i, j, k) .≤ size(Rx))
        Rx[i, j, k] =
            _dx * (next_x(τxx) - current(τxx)) +
            _dy * (next_y(τxy) - current(τxy)) +
            _dz * (next_z(τxz) - current(τxz)) -
            _dx * (next(P) - P[i, j + 1, k + 1]) +
            harm_xi(fx)
    end
    if all((i, j, k) .≤ size(Ry))
        Ry[i, j, k] =
            _dx * (next_x(τxy) - current(τxy)) +
            _dy * (next_y(τyy) - current(τyy)) +
            _dz * (next_z(τyz) - current(τyz)) -
            _dy * (next(P) - P[i + 1, j, k + 1]) +
            harm_yi(fy)
    end
    if all((i, j, k) .≤ size(Rz))
        Rz[i, j, k] =
            _dx * (next_x(τxz) - current(τxz)) +
            _dy * (next_y(τyz) - current(τyz)) +
            _dz * (next_z(τzz) - current(τzz)) -
            _dz * (next(P) - P[i + 1, j + 1, k]) +
            harm_zi(fz)
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
    li,
    freeslip,
    ρg,
    η,
    G,
    K,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T}

    # phsysics
    lx, ly, lz = li # gravitational forces
    fx, fy, fz = ρg # gravitational forces
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz # velocity
    P, ∇V = stokes.P, stokes.∇V # pressure and velociity divergence
    εxx, εyy, εzz, εxy, εxz, εyz = strain(stokes)
    τ, τ_o = stress(stokes) # stress 
    τxx, τyy, τzz, τxy, τxz, τyz = τ
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o = τ_o
    # solver related
    Rx, Ry, Rz, RP = stokes.R.Rx, stokes.R.Ry, stokes.R.Rz, stokes.R.RP
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    P_old = deepcopy(P)
    # geometry
    _dx, _dy, _dz = @. 1 / di
    nx, ny, nz = size(P)
    nx_1, nx_2, ny_1, ny_2, nz_1, nz_2 = nx - 1, nx - 2, ny - 1, ny - 2, nz - 1, nz - 2
    
    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end
    apply_free_slip!((freeslip_x=true, freeslip_y=true, freeslip_z=true), ητ, ητ, ητ)
    # @parallel (1:size(ητ, 2), 1:size(ητ, 3)) free_slip_x!(ητ)
    # @parallel (1:size(ητ, 1), 1:size(ητ, 3)) free_slip_y!(ητ)
    # @parallel (1:size(ητ, 1), 1:size(ητ, 2)) free_slip_z!(ητ)

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

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy, Vz)

            @parallel (1:nx, 1:ny, 1:nz) compute_∇V!(∇V, Vx, Vy, Vz, _dx, _dy, _dz)
            @parallel compute_strain_rate!(
                εxx, εyy, εzz, εyz, εxz, εxy, ∇V, Vx, Vy, Vz, _dx, _dy, _dz
            )
            @parallel compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
            @parallel compute_τ!(
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
                η,
                G,
                dt,
                θ_dτ
            )

            # @hide_communication b_width begin # communication/computation overlap
                @parallel (1:nx_1, 1:ny_1, 1:nz_1) compute_V!(
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
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1

            wtime0 += @elapsed begin
                @parallel (1:nx_1, 1:ny_1, 1:nz_1) compute_Res!(
                    Rx,
                    Ry,
                    Rz,
                    fx,
                    fy,
                    fz,
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

            # Vmin, Vmax = minimum_mpi(Vx), maximum_mpi(Vx)
            # Pmin, Pmax = minimum_mpi(P), maximum_mpi(P)
            # push!(norm_Rx, norm_mpi(Rx) / (Pmax - Pmin) * lx * _sqrt_len_Rx_g)
            # push!(norm_Ry, norm_mpi(Ry) / (Pmax - Pmin) * lx * _sqrt_len_Ry_g)
            # push!(norm_Rz, norm_mpi(Rz) / (Pmax - Pmin) * lx * _sqrt_len_Rz_g)
            
            # push!(norm_∇V, norm_mpi(∇V) / (Vmax - Vmin) * lx * _sqrt_len_∇V_g)
            # Vmin, Vmax = extrema(Vx)
            # Pmin, Pmax = extrema(P)
            push!(norm_Rx, maximum(abs.(Rx)))
            push!(norm_Ry, maximum(abs.(Ry)))
            push!(norm_Rz, maximum(abs.(Rz)))
            push!(norm_∇V, maximum(abs.(RP)))
            # push!(norm_Rx, norm(Rx) / (Pmax - Pmin) * lx * _sqrt_len_Rx_g)
            # push!(norm_Ry, norm(Ry) / (Pmax - Pmin) * lx * _sqrt_len_Ry_g)
            # push!(norm_Rz, norm(Rz) / (Pmax - Pmin) * lx * _sqrt_len_Rz_g)
            # push!(norm_∇V, norm(RP) / (Vmax - Vmin) * lx * _sqrt_len_∇V_g)

            err = maximum([norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont]])
            
            push!(
                err_evo1,
                maximum([norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont]]),
            )
            push!(err_evo2, iter)
            if igg.me == 0 && ((verbose && (err > ϵ)) || (iter == iterMax))
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

            if isnan(err)
                error("NaN")
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
