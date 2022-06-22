## UTILS

stress(stokes::StokesArrays{Viscous,A,B,C,D,nDim}) where {A,B,C,D,nDim} = stress(stokes.τ)
stress(τ::SymmetricTensor{<:AbstractMatrix{T}}) where {T} = (τ.xx, τ.yy, τ.xy)
function stress(τ::SymmetricTensor{<:AbstractArray{T,3}}) where {T}
    return (τ.xx, τ.yy, τ.zz, τ.xy, τ.xz, τ.yz)
end

@parallel function compute_iter_params!(
    dτ_Rho::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    Musτ::AbstractArray{T,2},
    Vpdτ::Real,
    Re::Real,
    r::Real,
    max_lxy::Real,
) where {T}
    @all(dτ_Rho) = Vpdτ * max_lxy / Re / @all(Musτ)
    @all(Gdτ) = Vpdτ^2 / @all(dτ_Rho) / (r + 2.0)
    return nothing
end

## DIMENSION AGNOSTIC KERNELS
@parallel function compute_maxloc!(A::AbstractArray, B::AbstractArray)
    @inn(A) = @maxloc(B)
    return nothing
end

## 2D STOKES MODULE

module Stokes2D

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using JustRelax
using LinearAlgebra
using CUDA
using Printf

import JustRelax: stress, compute_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax: Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, Viscous
import JustRelax: compute_maxloc!, solve!, pureshear_bc!

export compute_P!, compute_V!, solve!

## 2D KERNELS

@parallel function compute_P!(
    ∇V::AbstractArray{T,2},
    P::AbstractArray{T,2},
    Vx::AbstractArray{T,2},
    Vy::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    r::T,
    dx::T,
    dy::T,
) where {T}
    @all(∇V) = @d_xa(Vx) / dx + @d_ya(Vy) / dy
    @all(P) = @all(P) - r * @all(Gdτ) * @all(∇V)
    return nothing
end

@parallel function compute_τ!(
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    Vx::AbstractArray{T,2},
    Vy::AbstractArray{T,2},
    η::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    dx::T,
    dy::T,
) where {T}
    @all(τxx) = (@all(τxx) + 2.0 * @all(Gdτ) * @d_xa(Vx) / dx) / (@all(Gdτ) / @all(η) + 1.0)
    @all(τyy) = (@all(τyy) + 2.0 * @all(Gdτ) * @d_ya(Vy) / dy) / (@all(Gdτ) / @all(η) + 1.0)
    @all(τxy) =
        (@all(τxy) + 2.0 * @harm(Gdτ) * (0.5 * (@d_yi(Vx) / dy + @d_xi(Vy) / dx))) /
        (@harm(Gdτ) / @harm(η) + 1.0)
    return nothing
end

@parallel function compute_dV!(
    Rx::AbstractArray{T,2},
    Ry::AbstractArray{T,2},
    dVx::AbstractArray{T,2},
    dVy::AbstractArray{T,2},
    P::AbstractArray{T,2},
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    dτ_Rho::AbstractArray{T,2},
    ρg::Nothing,
    dx::T,
    dy::T,
) where {T}
    @all(Rx) = @d_xi(τxx) / dx + @d_ya(τxy) / dy - @d_xi(P) / dx
    @all(Ry) = @d_yi(τyy) / dy + @d_xa(τxy) / dx - @d_yi(P) / dy
    @all(dVx) = @harm_xi(dτ_Rho) * @all(Rx)
    @all(dVy) = @harm_yi(dτ_Rho) * @all(Ry)
    return nothing
end

@parallel function compute_dV!(
    Rx::AbstractArray{T,2},
    Ry::AbstractArray{T,2},
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
    @all(Rx) = @d_xi(τxx) / dx + @d_ya(τxy) / dy - @d_xi(P) / dx
    @all(Ry) = @d_yi(τyy) / dy + @d_xa(τxy) / dx - @d_yi(P) / dy - @harm_yi(ρg)
    @all(dVx) = @harm_xi(dτ_Rho) * @all(Rx)
    @all(dVy) = @harm_yi(dτ_Rho) * @all(Ry)
    return nothing
end

@parallel function compute_V!(
    Vx::AbstractArray{T,2},
    Vy::AbstractArray{T,2},
    dVx::AbstractArray{T,2},
    dVy::AbstractArray{T,2},
) where {T}
    @inn(Vx) = @inn(Vx) + @all(dVx)
    @inn(Vy) = @inn(Vy) + @all(dVy)
    return nothing
end

## VISCOUS STOKES SOLVER 

function solve!(
    stokes::StokesArrays{Viscous,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    li::NTuple{2,T},
    max_li,
    freeslip,
    ρg,
    η;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    dx, dy = di
    lx, ly = li
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    τxx, τyy, τxy = stress(stokes)
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ,
    pt_stokes.dτ_Rho, pt_stokes.ϵ, pt_stokes.Re, pt_stokes.r,
    pt_stokes.Vpdτ

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    # PT numerical coefficients
    @parallel compute_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, Re, r, max_li)
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
    while err > ϵ && iter <= iterMax
        wtime0 += @elapsed begin
            @parallel compute_P!(∇V, P, Vx, Vy, Gdτ, r, dx, dy)
            @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, η, Gdτ, dx, dy)
            @parallel compute_dV!(Rx, Ry, dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρg, dx, dy)
            @parallel compute_V!(Vx, Vy, dVx, dVy)

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            Vmin, Vmax = minimum(Vx), maximum(Vx)
            Pmin, Pmax = minimum(P), maximum(P)
            push!(norm_Rx, norm(Rx) / (Pmax - Pmin) * lx / sqrt(length(Rx)))
            push!(norm_Ry, norm(Ry) / (Pmax - Pmin) * lx / sqrt(length(Ry)))
            push!(norm_∇V, norm(∇V) / (Vmax - Vmin) * lx / sqrt(length(∇V)))
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

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

end
