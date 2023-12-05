# Pseudo-transient solver for two phase flow
# The formulation form the Book Introduction to numerical geodynamic modelling by Taras Gerya is used
# DOI:  https://doi.org/10.1017/9781316534243
# The structure of the code is based on the ParallelStencil miniapp HydroMech2D.jl 8ab11d5cdd3bc28e64e389db3a674fc8c16e66e6

<<<<<<< HEAD
=======
## DIMENSION AGNOSTIC KERNELS
@parallel function compute_maxloc!(A::AbstractArray, B::AbstractArray)
    @inn(A) = @maxloc(B)
    return nothing
end

>>>>>>> e73c9771331d58652e55511c9a8877816900db59
#=
@parallel function hyd_pressure!(
    Pt::AbstractArray{T,2},
    ρsg::AbstractArray{T,2},
)
    @all(Pt) = a
    return nothing
end
=#

@parallel function compute_phys_params!(
    η_φ::AbstractArray{T,2},
    k_ηf::AbstractArray{T,2},
    ρtg::AbstractArray{T,2},
    φ::AbstractArray{T,2},
    ρsg::AbstractArray{T,2},
    ρm::AbstractArray{T,2},
    ρfg::Real,
    η_φbg::Real,
    φ_bg::Real,
    k_ηf0::Real,
    n::Real,
) where{T}
    @all(η_φ) = @all(φ) / (η_φbg *φ_bg) # compaction viscosity 1/η
    @all(k_ηf) = (φ_bg ^ n) / (k_ηf0 * @all(φ) ^ n) # permeability 1/k_ηf
    @all(ρtg) = ρfg * @all(φ) + (1 - @all(φ)) * @all(ρsg) #- @all(ρm)
    return nothing
end

@parallel function compute_iter_params!(
    Re_F::AbstractArray{T,2},
    Rhodτ_M::AbstractArray{T,2},
    Rhodτ_F::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    Betadτ::AbstractArray{T,2},
    k_ηfτ::AbstractArray{T,2},
    ηs::AbstractArray{T,2},
    k_ηf::AbstractArray{T,2},
    η_φ::AbstractArray{T,2},
    Vpdτ::Real,
    Re_M::Real,
    r::Real,
    min_li::Real,
) where {T}
    @all(Re_F) = π + sqrt(π^2 + min_li^2 * @all(k_ηf) * @all(η_φ))
    @all(Rhodτ_M) = Vpdτ * min_li / Re_M / @all(ηs) # 1/ρ
    @all(Rhodτ_F) = @all(Re_F) / (Vpdτ * min_li * @all(η_φ))
    @all(Gdτ) = Vpdτ^2 / @all(Rhodτ_M) / (r + 2.0)
    @all(Betadτ) = 1.0 / (@all(Rhodτ_F) *  Vpdτ^2)
    @all(k_ηfτ) = 1.0 / (@all(Rhodτ_F) + @all(k_ηf)) # used to make qD update rule implicite
    return nothing
end

@parallel function update_φ!(
    φ::AbstractArray{T,2},
    η_φ::AbstractArray{T,2},
    dt::Real,
) where{T}
    @all(φ) = dt * @all(φ) * @all(η_φ)
    return nothing
end

#=
@parallel function upwind!(

)
end
=#

## 2D TPF MODULE

module TwoPhaseFlow2D

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using JustRelax
using LinearAlgebra
using CUDA
using Printf

import JustRelax: stress,strain, compute_iter_params!, compute_phys_params!, PTArray, Velocity, SymmetricTensor
import JustRelax: Residual, TPF_Pressure, P_Residual, TPFArrays, PTTPFCoeffs, PTTPFParams, AbstractStokesModel, Viscous
import JustRelax: compute_maxloc!, solve!

import ..Stokes2D: compute_strain_rate!, compute_τ!, compute_dV!, compute_V!

export solve!

## 2D KERNELS

## Solver functions

@parallel function compute_RP!(
    RPt::AbstractArray{T,2},
    RPe::AbstractArray{T,2},
    dPt::AbstractArray{T,2},
    dPe::AbstractArray{T,2},
    Pe::AbstractArray{T,2},
    εxx::AbstractArray{T,2},
    εyy::AbstractArray{T,2},
    qDx::AbstractArray{T,2},
    qDy::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    η_φ::AbstractArray{T,2},
    Betadτ::AbstractArray{T,2},
    r::T,
    _dx::T,
    _dy::T,
) where {T}
    @all(RPt) = @all(εxx) + @all(εyy) + @all(Pe) * @all(η_φ)
    @all(RPe) = @d_xa(qDx) * _dx + @d_ya(qDy) * _dy - @all(Pe) * @all(η_φ)
    @all(dPt) = - r * @all(Gdτ) * @all(RPt)
    @all(dPe) = 1.0 / (@all(Betadτ) + @all(η_φ)) * @all(RPe) # used to make Pe update rule implicite
    return nothing
end

@parallel function compute_P!(
    Pt::AbstractArray{T,2},
    Pe::AbstractArray{T,2},
    dPt::AbstractArray{T,2},
    dPe::AbstractArray{T,2},
) where{T}
    @all(Pt) = @all(Pt) + @all(dPt)
    @all(Pe) = @all(Pe) + @all(dPe)
    return nothing
end

#= In Stokes.jl av_yi(ρg) used????
@parallel function compute_dV!(
    Rx::AbstractArray{T,2},
    Ry::AbstractArray{T,2},
    dVx::AbstractArray{T,2},
    dVy::AbstractArray{T,2},
    Pt::AbstractArray{T,2},
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    Rhodτ_M::AbstractArray{T,2},
    ρg::AbstractArray{T,2},
    _dx::T,
    _dy::T,
) where {T}
    @all(Rx) = @d_xi(τxx) * _dx + @d_ya(τxy) * _dy - @d_xi(Pt) * _dx
    @all(Ry) = @d_yi(τyy) * _dy + @d_xa(τxy) * _dx - @d_yi(Pt) * _dy - @harm_yi(ρg)
    @all(dVx) = @harm_xi(Rhodτ_M) * @all(Rx)
    @all(dVy) = @harm_yi(Rhodτ_M) * @all(Ry)
    return nothing
end
=#

@parallel function compute_dqD!(
    RqDx::AbstractArray{T,2},
    RqDy::AbstractArray{T,2},
    dqDx::AbstractArray{T,2},
    dqDy::AbstractArray{T,2},
    qDx::AbstractArray{T,2},
    qDy::AbstractArray{T,2},
    Pe::AbstractArray{T,2},
    Pt::AbstractArray{T,2},
    k_ηf::AbstractArray{T,2},
    k_ηfτ::AbstractArray{T,2},
    ρsg::AbstractArray{T,2},
    ρm::AbstractArray{T,2},
    ρfg::T,
    _dx::T,
    _dy::T,
) where {T}
    @all(RqDx) = - @inn(qDx) * @harm_xi(k_ηf) + @d_xi(Pe) * _dx -  @d_xi(Pt) * _dx
    @all(RqDy) = - @inn(qDy) * @harm_yi(k_ηf) + @d_yi(Pe) * _dy -  @d_yi(Pt) * _dy - ρfg #+ @harm_yi(ρm)#+ (@harm_yi(ρsg)-ρfg) + @harm_yi(ρm)
    @all(dqDx) = @harm_xi(k_ηfτ) * @all(RqDx)
    @all(dqDy) = @harm_yi(k_ηfτ) * @all(RqDy)
    return nothing
end

## VISCOUS STOKES SOLVER 

function solve!(
    TPF::TPFArrays{Viscous,A,B,C,D,E,F,2},
    TPF_Coeffs::PTTPFCoeffs{Viscous,T,2},
    TPF_Params::PTTPFParams{Viscous,G,T},
    di::NTuple{2,T},
    min_li,
    freeslip,
    ρsg,
    ρfg,
    ηs,
    nt;
    #η_φ,
    #k_ηf;
    iterMax=1e5,
    nout=500,
    verbose=true,
) where {A,B,C,D,E,F,G,T}

    # unpack
    dx, dy = di
    _dx, _dy = inv.(di)

    # TPFArrays
    Vx, Vy, qDx, qDy =  TPF.V.Vx, TPF.V.Vy, TPF.qD.Vx, TPF.qD.Vy
    dVx, dVy, dqDx, dqDy = TPF.dV.Vx, TPF.dV.Vy, TPF.dqD.Vx, TPF.dqD.Vy
    Rx, Ry, RqDx, RqDy = TPF.R.Rx, TPF.R.Ry, TPF.RqD.Rx, TPF.RqD.Ry
    Pt, Pe = TPF.P.Pt, TPF.P.Pe
    dPt, dPe = TPF.dP.Pt, TPF.dP.Pe
    RPt, RPe = TPF.RP.RPt, TPF.RP.RPe
    τxx, τyy, τxy = stress(TPF)
    εxx, εyy, εxy = strain(stokes)
    φ, η_φ, k_ηf, k_ηfτ, ρtg = TPF.φ, TPF.η_φ, TPF.k_ηf, TPF.k_ηfτ, TPF.ρtg
    #k_ηfτ = TPF.k_ηfτ

    # PTTPFCoeff
    ϵ = TPF_Coeffs.ϵ
    Re_M, Re_F = TPF_Coeffs.Re_M, TPF_Coeffs.Re_F
    r, Vpdτ = TPF_Coeffs.r, TPF_Coeffs.Vpdτ
    Rhodτ_M, Rhodτ_F = TPF_Coeffs.Rhodτ_M, TPF_Coeffs.Rhodτ_F
    Gdτ, Betadτ = TPF_Coeffs.Gdτ, TPF_Coeffs.Betadτ

    # PTTPFParams
    n = TPF_Params.n
    k_ηf0, φ_bg, η_φbg = TPF_Params.kr, TPF_Params.φr, TPF_Params.ηr
    ρm = TPF_Params.ρm

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_RPe = Float64[]
    norm_RPt = Float64[]
    norm_RqDx = Float64[]
    norm_RqDy = Float64[]


    # solver loop
    wtime0 = Float64[]
    while err > ϵ && iter <= iterMax
        tmp_t = @elapsed begin
            @parallel compute_phys_params!(η_φ, k_ηf, ρtg, φ, ρsg, ρm, ρfg, η_φbg, φ_bg, k_ηf0, n)
            @parallel compute_iter_params!(
                Re_F,
                Rhodτ_M,
                Rhodτ_F,
                Gdτ,
                Betadτ,
                k_ηfτ,
                ηs,
                k_ηf,
                η_φ,
                Vpdτ,
                Re_M,
                r,
                min_li
            )
            @parallel compute_strain_rate!(εxx, εyy, εxy, Vx, Vy, _dx, _dy)
            @parallel compute_RP!(RPt, RPe, dPt, dPe, Pe, εxx, εyy, qDx, qDy, Gdτ, η_φ, Betadτ, r, _dx, _dy)
            @parallel compute_P!(Pt, Pe, dPt, dPe)
            @parallel (1:size(Pe, 1)) zero_y!(Pe)
            # Add boundary condition for Pe, zero on top and bottom
            @parallel compute_τ!(τxx, τyy, τxy, Vx, Vy, ηs, Gdτ, _dx, _dy)
            @parallel compute_dV!(Rx, Ry, dVx, dVy, Pt, τxx, τyy, τxy, Rhodτ_M, ρtg, _dx, _dy)
            @parallel compute_dqD!(RqDx, RqDy, dqDx, dqDy, qDx, qDy, Pe, Pt, k_ηf, k_ηfτ, ρsg, ρm, ρfg, _dx, _dy)
            @parallel compute_V!(Vx, Vy, dVx, dVy) # Solid velocity
            @parallel compute_V!(qDx, qDy, dqDx, dqDy) # Fluid velocity

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)
            apply_free_slip!(freeslip, qDx, qDy)
        end

        push!(wtime0, tmp_t)

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            push!(norm_Rx, norm(Rx) / length(Rx))
            push!(norm_Ry, norm(Ry) / length(Ry))
            push!(norm_RPe, norm(RPe[:,2:end-1]) / length(RPe))
            push!(norm_RPt, norm(RPt) / length(RPt))
            push!(norm_RqDx, norm(RqDx) / length(RqDx))
            push!(norm_RqDy, norm(RqDy) / length(RqDy))
            err = maximum([norm_Rx[cont], norm_Ry[cont], norm_RPe[cont], #=norm_RPt[cont],=# norm_RqDx[cont],norm_RqDy[cont]])
            push!(err_evo1, maximum([norm_Rx[cont], norm_Ry[cont], norm_RPe[cont], #=norm_RPt[cont],=# norm_RqDx[cont],norm_RqDy[cont]]))
            push!(err_evo2, iter)
            if verbose || (err < ϵ) || (iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_RPe=%1.3e, norm_RPt=%1.3e, norm_RqDx=%1.3e, norm_RqDy=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_RPe[cont],
                    norm_RPt[cont],
                    norm_RqDx[cont],
                    norm_RqDy[cont]
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
        norm_RPe=norm_RPe,
        norm_RPt=norm_RPt,
        norm_RqDx=norm_RqDx,
        norm_RqDy=norm_RqDy,
        wtime=wtime0,
    )
end

end # END OF MODULE
