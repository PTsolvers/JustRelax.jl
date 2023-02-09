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

using ..JustRelax
using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using GeoParams, LinearAlgebra, Printf

import JustRelax: stress, strain, elastic_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax: Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic
import JustRelax: compute_maxloc!, solve!, @tuple

import ..Stokes2D: compute_P!, compute_V!, compute_strain_rate!

export solve!

include("StressRotation.jl")

## 2D ELASTIC KERNELS

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,2}) where {A,B,C,D}
    # τ, τ_o = stress(stokes)
    τxx, τyy, τxy, τxy_c = stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, stokes.τ.xy_c
    τxx_o, τyy_o, τxy_o, τxy_o_c = stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy, stokes.τ_o.xy_c
    @parallel  update_τ_o!(τxx_o, τyy_o, τxy_o, τxy_o_c, τxx, τyy, τxy, τxy_c)
    return nothing
end

@parallel function update_τ_o!(τxx_o, τyy_o, τxy_o, τxy_o_c, τxx, τyy, τxy, τxy_c)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    @all(τxy_o_c) = @all(τxy_c)
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

## Compressible - GeoParams
@parallel function compute_P!(P, P_old, RP, ∇V, η, K::Number, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (K * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (K * dt))
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
        return
end

@parallel_indices (i, j) function compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy)
    # Again, indices i, j are captured by the closure
    @inbounds @inline d_xa(A) = (A[i + 1, j] - A[i, j]) * _dx
    @inbounds @inline d_ya(A) = (A[i, j + 1] - A[i, j]) * _dy
    @inbounds @inline d_xi(A) = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
    @inbounds @inline d_yi(A) = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
    @inbounds @inline av_xa(A) = (A[i + 1, j] + A[i, j]) * 0.5
    @inbounds @inline av_ya(A) = (A[i, j + 1] + A[i, j]) * 0.5

    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        @inbounds Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        @inbounds Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
    end
    return nothing
end

## Compressible - GeoParams
@parallel function compute_P!(P, P_old, RP, ∇V, η, K::Number, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (K * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (K * dt))
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

@parallel_indices (i, j) function compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy)
    # Again, indices i, j are captured by the closure
    @inbounds @inline d_xa(A) = (A[i + 1, j] - A[i, j]) * _dx
    @inbounds @inline d_ya(A) = (A[i, j + 1] - A[i, j]) * _dy
    @inbounds @inline d_xi(A) = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
    @inbounds @inline d_yi(A) = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
    @inbounds @inline av_xa(A) = (A[i + 1, j] + A[i, j]) * 0.5
    @inbounds @inline av_ya(A) = (A[i, j + 1] + A[i, j]) * 0.5

    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        @inbounds Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        @inbounds Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
    end
    return nothing
end

@parallel_indices (i, j) function compute_V_Res!(
    Vx, Vy, Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, ητ, ηdτ, _dx, _dy
)

    # Again, indices i, j are captured by the closure
    @inline d_xa(A) = (A[i + 1, j] - A[i, j]) * _dx
    @inline d_ya(A) = (A[i, j + 1] - A[i, j]) * _dy
    @inline d_xi(A) = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
    @inline d_yi(A) = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
    @inline av_xa(A) = (A[i + 1, j] + A[i, j]) * 0.5
    @inline av_ya(A) = (A[i, j + 1] + A[i, j]) * 0.5

    if all((i, j) .≤ size(Rx))
        @inbounds R = Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
        @inbounds Vx[i + 1, j + 1] += R * ηdτ / av_xa(ητ)
    end
    if all((i, j) .≤ size(Ry))
        @inbounds R = Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
        @inbounds Vy[i + 1, j + 1] += R * ηdτ / av_ya(ητ)
    end
    return nothing
end

# Stress calculation

# viscous
@parallel function compute_τ!(τxx, τyy, τxy, εxx, εyy, εxy, η, θ_dτ)
    @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0 * @all(η) * @all(εxx)) * 1.0 / (θ_dτ + 1.0)
    @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0 * @all(η) * @all(εyy)) * 1.0 / (θ_dτ + 1.0)
    @inn(τxy) = @inn(τxy) + (-@inn(τxy) + 2.0 * @av(η) * @inn(εxy)) * 1.0 / (θ_dτ + 1.0)
    return nothing
end

# visco-elastic
@parallel function compute_τ!(
    τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
)
    @all(τxx) =
        @all(τxx) +
        (
            -(@all(τxx) - @all(τxx_o)) * @all(η) / (@all(G) * dt) - @all(τxx) +
            2.0 * @all(η) * @all(εxx)
        ) * 1.0 / (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @all(τyy) =
        @all(τyy) +
        (
            -(@all(τyy) - @all(τyy_o)) * @all(η) / (@all(G) * dt) - @all(τyy) +
            2.0 * @all(η) * @all(εyy)
        ) * 1.0 / (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @inn(τxy) =
        @inn(τxy) +
        (
            -(@inn(τxy) - @inn(τxy_o)) * @av(η) / (@av(G) * dt) - @inn(τxy) +
            2.0 * @av(η) * @inn(εxy)
        ) * 1.0 / (θ_dτ + @av(η) / (@av(G) * dt) + 1.0)

    return nothing
end

# visco-elasto-plastic with GeoParams
@parallel_indices (i, j) function compute_τ_gp!(
    τxx,
    τyy,
    τxy,
    τII,
    τxx_o,
    τyy_o,
    τxyv_o,
    εxx,
    εyy,
    εxyv,
    η,
    η_vep,
    z,
    T,
    MatParam,
    dt,
    θ_dτ
)
    # convinience closure
    @inline gather(A) = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1] 
    @inline av(T)     = (T[i, j] + T[i + 1, j] + T[i, j + 1] + T[i + 1, j + 1]) * 0.25

    @inline begin
        # # numerics
        # dτ_r                = 1.0 / (θ_dτ + η[i, j] / (get_G(MatParam[1]) * dt) + 1.0) # original
        @inbounds dτ_r                = 1.0 / (θ_dτ / η[i, j] + 1.0 / η_vep[i, j]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
        # # Setup up input for GeoParams.jl
        args                = (; dt=dt, P = 1e6 * (1 - z[j]) , T=av(T), τII_old=0.0)
        # args                = (; dt=dt, P=P[i, j] , T=av(T), τII_old=0.0)
        εij_p               = εxx[i, j]+1e-25, εyy[i, j]+1e-25, gather(εxyv).+1e-25
        τij_p_o             = τxx_o[i,j], τyy_o[i,j], gather(τxyv_o)
        phases              = (1, 1, (1,1,1,1)) # for now hard-coded for a single phase
        # update stress and effective viscosity
        τij, τII[i, j], ηᵢ  = compute_τij(MatParam, εij_p, args, τij_p_o, phases)
        # ηᵢ                  = clamp(ηᵢ, 1e0, 1e6)
        τxx[i, j]          += dτ_r * (-(τxx[i,j]) + τij[1] ) / ηᵢ # NOTE: from GP Tij = 2*η_vep * εij
        τyy[i, j]          += dτ_r * (-(τyy[i,j]) + τij[2] ) / ηᵢ 
        τxy[i, j]          += dτ_r * (-(τxy[i,j]) + τij[3] ) / ηᵢ 
        η_vep[i, j]         = ηᵢ
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
    τ, _ = stress(stokes)
    τxx, τyy, τxy = τ
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
            @parallel compute_τ!(τxx, τyy, τxy, εxx, εyy, εxy, η, θ_dτ)
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
    flow_bcs,
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
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)

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
            @parallel compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)
            @parallel compute_strain_rate!(
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                stokes.∇V,
                stokes.V.Vx,
                stokes.V.Vy,
                _di...,
            )
            @parallel compute_P!(stokes.P, P_old, stokes.R.RP, stokes.∇V, η, K, dt, r, θ_dτ)
            @parallel compute_τ!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                stokes.τ_o.xx,
                stokes.τ_o.yy,
                stokes.τ_o.xy,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                η,
                G,
                θ_dτ,
                dt,
            )
            @parallel compute_V!(
                stokes.V.Vx,
                stokes.V.Vy,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ηdτ,
                ρg[1],
                ρg[2],
                ητ,
                _di...,
            )
            # free slip boundary conditions
            flow_bcs!(flow_bcs, stokes.V.Vx, stokes.V.Vy, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (1:nx, 1:ny) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ρg[1],
                ρg[2],
                _di...,
            )
            errs = maximum.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if (verbose) || (iter == iterMax)
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

    if -Inf < dt < Inf 
        update_τ_o!(stokes)
        @parallel (1:nx, 1:ny) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
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

# GeoParams: general (visco-elasto-plastic) solver

tupleize(v::MaterialParams) = (v,)
tupleize(v::Tuple) = v

@parallel function center2vertex!(vertex, center)
    @inn(vertex) = @av(center)
    return nothing
end

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    thermal::ThermalArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    η_vep,
    MatParam::MaterialParams,
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)
    z = LinRange(di[2]*0.5, 1.0-di[2]*0.5, ny)
    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    apply_free_slip!((freeslip_x=true, freeslip_y=true), ητ, ητ)

    Kb = get_Kb(MatParam)

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
            @parallel compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)
            @parallel compute_P!(
                stokes.P,
                P_old,
                stokes.R.RP,
                stokes.∇V,
                η,
                Kb,
                dt,
                r,
                θ_dτ,
            )
            @parallel compute_strain_rate!(
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                stokes.∇V,
                stokes.V.Vx,
                stokes.V.Vy,
                _di...,
            )
            @parallel (1:nx, 1:ny) compute_τ_gp!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy_c,
                stokes.τ.II,
                stokes.τ_o.xx,
                stokes.τ_o.yy,
                stokes.τ_o.xy,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                η,
                η_vep,
                z,
                thermal.T,
                tupleize(MatParam), # needs to be a tuple
                dt,
                θ_dτ,
            )
            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @parallel compute_V!(
                stokes.V.Vx,
                stokes.V.Vy,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ηdτ,
                ρg[1],
                ρg[2],
                ητ,
                _di...,
            )
            # apply boundary conditions boundary conditions
            # apply_free_slip!(freeslip, stokes.V.Vx, stokes.V.Vy)
            flow_bcs!(flow_bcs, stokes.V.Vx, stokes.V.Vy, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (1:nx, 1:ny) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ρg[1],
                ρg[2],
                _di...,
            )
            errs = maximum.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum(errs)
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

    if -Inf < dt < Inf 
        update_τ_o!(stokes)
        @parallel (1:nx, 1:ny) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
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
using GeoParams

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
            ∇Vᵢⱼₖ = ∇V[i, j, k] / 3.0
            # Compute ε_xx
            εxx[i, j, k] = _dx * (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) - ∇Vᵢⱼₖ
            # Compute ε_yy
            εyy[i, j, k] = _dy * (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) - ∇Vᵢⱼₖ
            # Compute ε_zz
            εzz[i, j, k] = _dz * (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k]) - ∇Vᵢⱼₖ
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
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (@all(K) * dt))
    return nothing
end

## Compressible - GeoParams
@parallel function compute_P!(P, P_old, RP, ∇V, η, K::Number, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (K * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (K * dt))
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
    Base.@propagate_inbounds @inline harm_x(x) = 2.0 / (1.0 / x[i + 1, j, k] + 1.0 / x[i, j, k])
    Base.@propagate_inbounds @inline harm_y(x) = 2.0 / (1.0 / x[i, j + 1, k] + 1.0 / x[i, j, k])
    Base.@propagate_inbounds @inline harm_z(x) = 2.0 / (1.0 / x[i, j, k + 1] + 1.0 / x[i, j, k])
    Base.@propagate_inbounds @inline av_x(x)   = 0.5 * (x[i + 1, j, k] + x[i, j, k])
    Base.@propagate_inbounds @inline av_y(x)   = 0.5 * (x[i, j + 1, k] + x[i, j, k])
    Base.@propagate_inbounds @inline av_z(x)   = 0.5 * (x[i, j, k + 1] + x[i, j, k])
    Base.@propagate_inbounds @inline dx(x)     = x[i + 1, j, k] - x[i, j, k]
    Base.@propagate_inbounds @inline dy(x)     = x[i, j + 1, k] - x[i, j, k]
    Base.@propagate_inbounds @inline dz(x)     = x[i, j, k + 1] - x[i, j, k]

    @inbounds begin
        if all((i, j, k) .< size(Vx) .- 1)
            Rx_ijk =
                _dx * (τxx[i + 1, j    , k    ] - τxx[i    , j, k]) +
                _dy * (τxy[i + 1, j + 1, k    ] - τxy[i + 1, j, k]) +
                _dz * (τxz[i + 1, j    , k + 1] - τxz[i + 1, j, k]) -
                _dx * dx(P) + av_x(fx)
            Vx[i + 1, j + 1, k + 1] += Rx_ijk * ηdτ / av_x(ητ)
            Rx[i, j, k] = Rx_ijk
        end
        if all((i, j, k) .< size(Vy) .- 1)
            Ry_ijk =
                _dx * (τxy[i + 1, j + 1, k    ] - τxy[i, j + 1, k]) +
                _dy * (τyy[i    , j + 1, k    ] - τyy[i, j    , k]) +
                _dz * (τyz[i    , j + 1, k + 1] - τyz[i, j + 1, k]) -
                _dy * dy(P) + av_y(fy)
            Vy[i + 1, j + 1, k + 1] += Ry_ijk * ηdτ / av_y(ητ)
            Ry[i, j, k] = Ry_ijk
        end
        if all((i, j, k) .< size(Vz) .- 1)
            Rz_ijk =
                _dx * (τxz[i + 1, j    , k + 1] - τxz[i, j, k + 1]) +
                _dy * (τyz[i    , j + 1, k + 1] - τyz[i, j, k + 1]) +
                _dz * (τzz[i    , j    , k + 1] - τzz[i, j, k    ]) - 
                _dz * dz(P) + av_z(fz)
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
    Base.@propagate_inbounds @inline function harm_xy(x)
        4.0 / (
            1.0 / x[i - 1, j - 1, k] +
            1.0 / x[i - 1, j, k] +
            1.0 / x[i, j - 1, k] +
            1.0 / x[i, j, k]
        )
    end
    Base.@propagate_inbounds @inline function harm_xz(x)
        4.0 / (
            1.0 / x[i, j, k] +
            1.0 / x[i - 1, j, k] +
            1.0 / x[i, j, k - 1] +
            1.0 / x[i - 1, j, k - 1]
        )
    end
    Base.@propagate_inbounds @inline function harm_yz(x)
        4.0 / (
            1.0 / x[i, j, k] +
            1.0 / x[i, j - 1, k] +
            1.0 / x[i, j, k - 1] +
            1.0 / x[i, j - 1, k - 1]
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

    @inbounds begin
        if all((i, j, k) .≤ size(τxx))
            # Compute τ_xx
            τxx[i, j, k] +=
                (
                    -(current(τxx) - current(τxx_o)) * current(η) / (current(G) * dt) -
                    current(τxx) + 2.0 * current(η) * current(εxx)
                ) / (θ_dτ + current(η) / (current(G) * dt) + 1.0)
            # Compute τ_yy
            τyy[i, j, k] +=
                (
                    -(current(τyy) - current(τyy_o)) * current(η) / (current(G) * dt) -
                    current(τyy) + 2.0 * current(η) * current(εyy)
                ) / (θ_dτ + current(η) / (current(G) * dt) + 1.0)
            # Compute τ_zz
            τzz[i, j, k] +=
                (
                    -(current(τzz) - current(τzz_o)) * current(η) / (current(G) * dt) -
                    current(τzz) + 2.0 * current(η) * current(εzz)
                ) / (θ_dτ + current(η) / (current(G) * dt) + 1.0)
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
    τyz,
    τxz,
    τxy,
    τyz_o,
    τxz_o,
    τxy_o,
    εyz,
    εxz,
    εxy,
    η,
    G,
    dt,
    θ_dτ,
)
    
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

    @inbounds begin
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            τxy[i, j, k] +=
                (
                    -(current(τxy) - current(τxy_o)) * av_xy(η) / (G * dt) -
                    current(τxy) + 2.0 * av_xy(η) * current(εxy)
                ) / (θ_dτ + av_xy(η) / (G * dt) + 1.0)
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            τxz[i, j, k] +=
                (
                    -(current(τxz) - current(τxz_o)) * av_xz(η) / (G * dt) -
                    current(τxz) + 2.0 * av_xz(η) * current(εxz)
                ) / (θ_dτ + av_xz(η) / (G * dt) + 1.0)
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            τyz[i, j, k] +=
                (
                    -(current(τyz) - current(τyz_o)) * av_yz(η) / (G * dt) -
                    current(τyz) + 2.0 * av_yz(η) * current(εyz)
                ) / (θ_dτ + av_yz(η) / (G * dt) + 1.0)
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
    MatParam, 
    dt, 
    θ_dτ,
)
    # convinience closures
    @inline gather_yz(A) =  A[i, j, k], A[i    , j + 1, k], A[i, j    , k + 1], A[i    , j + 1, k + 1]
    @inline gather_xz(A) =  A[i, j, k], A[i + 1, j    , k], A[i, j    , k + 1], A[i + 1, j    , k + 1]
    @inline gather_xy(A) =  A[i, j, k], A[i + 1, j    , k], A[i, j + 1, k    ], A[i + 1, j + 1, k    ]

    @inbounds begin
        # dτ_r = 1.0 / (θ_dτ + η[i, j, k] / (get_G(MatParam[1]) * dt) + 1.0) # original
        dτ_r  = 1.0 / (θ_dτ / η[i, j, k] + 1.0 / η_vep[i, j, k]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
        # Setup up input for GeoParams.jl
        T_cell = 0.125 * (
            T[i, j, k  ] + T[i, j+1, k  ] + T[i+1, j, k  ] + T[i+1, j+1, k  ] +
            T[i, j, k+1] + T[i, j+1, k+1] + T[i+1, j, k+1] + T[i+1, j+1, k+1]
        )
        args  = (; dt=dt, P = 1e6 * (1 - z[k]), T=T_cell, τII_old=0.0)
        εij_p = (
            εxx[i, j, k]+1e-25, 
            εyy[i, j, k]+1e-25, 
            εzz[i, j, k]+1e-25, 
            gather_yz(εyz).+1e-25, 
            gather_xz(εxz).+1e-25, 
            gather_xy(εxy).+1e-25
        )
        τij_p_o = (
            τxx_o[i, j, k],
            τyy_o[i, j, k],
            τzz_o[i, j, k],
            gather_yz(τyz_o), 
            gather_xz(τxz_o), 
            gather_xy(τxy_o)
        )
        phases = (1, 1, 1, (1,1,1,1), (1,1,1,1), (1,1,1,1)) # for now hard-coded for a single phase
        # update stress and effective viscosity
        τij, τII[i, j, k], ηᵢ = compute_τij(MatParam, εij_p, args, τij_p_o, phases)
        τ = ( # caching out improves a wee bit the performance
            τxx[i, j, k],
            τyy[i, j, k],
            τzz[i, j, k],
            τyz[i, j, k],
            τxz[i, j, k],
            τxy[i, j, k], 
        )
        dτ_rηᵢ = dτ_r/ηᵢ
        τxx[i, j, k]  += dτ_rηᵢ * (-τ[1] + τij[1]) # NOTE: from GP Tij = 2*η_vep * εij
        τyy[i, j, k]  += dτ_rηᵢ * (-τ[2] + τij[2]) 
        τzz[i, j, k]  += dτ_rηᵢ * (-τ[3] + τij[3]) 
        # τyz[i, j, k]  += dτ_rηᵢ * (-τ[4] + τij[4]) 
        # τxz[i, j, k]  += dτ_rηᵢ * (-τ[5] + τij[5]) 
        # τxy[i, j, k]  += dτ_rηᵢ * (-τ[6] + τij[6]) 
        η_vep[i, j, k] = ηᵢ
    end
    return
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
4444444444444444444324    
    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    nx, ny, nz = size(stokes.P)

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
            @parallel (1:nx, 1:ny, 1:nz) compute_∇V!(
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
            @parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) compute_strain_rate!(
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
            @parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) compute_τ!(
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
                # (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) 
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

            apply_free_slip!(freeslip, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
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

# @parallel function center2vertex!(vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy)
#     @inn_yz(vertex_yz) = @av_yza(center_yz)
#     @inn_xz(vertex_xz) = @av_xza(center_xz)
#     @inn_xy(vertex_xy) = @av_xya(center_xy)
#     return nothing
# end

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,3},
    thermal::ThermalArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    freeslip,
    ρg,
    η,
    η_vep,
    MatParam::MaterialParams,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T}

    ## UNPACK

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    nx, ny, nz = size(stokes.P)
    z = LinRange(di[3]*0.5, 1.0-di[3]*0.5, nz)

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

    Kb = get_Kb(MatParam)
    G  = get_G(MatParam)
    @parallel assign!(stokes.P0, stokes.P)

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (1:nx, 1:ny, 1:nz) compute_∇V!(
                stokes.∇V, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz, _di...
            )
            @parallel compute_P!(
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
            @parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) compute_strain_rate!(
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
            @parallel (1:nx, 1:ny, 1:nz) compute_τ_gp!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.zz,
                stokes.τ.yz_c, 
                stokes.τ.xz_c, 
                stokes.τ.xy_c,
                stokes.τ.II,
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
                η_vep,
                z,
                thermal.T,
                tupleize(MatParam), # needs to be a tuple
                dt,
                pt_stokes.θ_dτ,
            )
            # @parallel center2vertex!(
            #     stokes.τ.yz, 
            #     stokes.τ.xz, 
            #     stokes.τ.xy, 
            #     stokes.τ.yz_c, 
            #     stokes.τ.xz_c, 
            #     stokes.τ.xy_c
            # )
            # ------------------------------
            # @parallel (1:size(stokes.τ.xz,2), 1:size(stokes.τ.xz,3)) zero_shear_stress_lateral!(stokes.τ.xz)
            # @parallel (1:size(stokes.τ.xy,2), 1:size(stokes.τ.xy,3)) zero_shear_stress_lateral!(stokes.τ.xy)
            # @parallel (1:size(stokes.τ.xz,1), 1:size(stokes.τ.xz,2)) zero_shear_stress_front!(stokes.τ.xy)
            # @parallel (1:size(stokes.τ.yz,1), 1:size(stokes.τ.yz,2)) zero_shear_stress_front!(stokes.τ.yz)
            # @parallel (1:size(stokes.τ.xz,1), 1:size(stokes.τ.xz,2)) zero_shear_stress_top!(stokes.τ.xz)
            # @parallel (1:size(stokes.τ.yz,1), 1:size(stokes.τ.yz,2)) zero_shear_stress_top!(stokes.τ.yz)

            # @parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) compute_τ!(
            #     stokes.τ.xx,
            #     stokes.τ.yy,
            #     stokes.τ.zz,
            #     stokes.τ.yz,
            #     stokes.τ.xz,
            #     stokes.τ.xy,
            #     stokes.τ_o.xx,
            #     stokes.τ_o.yy,
            #     stokes.τ_o.zz,
            #     stokes.τ_o.yz,
            #     stokes.τ_o.xz,
            #     stokes.τ_o.xy,
            #     stokes.ε.xx,
            #     stokes.ε.yy,
            #     stokes.ε.zz,
            #     stokes.ε.yz,
            #     stokes.ε.xz,
            #     stokes.ε.xy,
            #     η,
            #     @fill(Inf, size(stokes.P)),
            #     dt,
            #     pt_stokes.θ_dτ,
            # )

            @parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) compute_τ_vertex!(
                stokes.τ.yz,
                stokes.τ.xz,
                stokes.τ.xy,
                stokes.τ_o.yz,
                stokes.τ_o.xz,
                stokes.τ_o.xy,
                stokes.ε.yz,
                stokes.ε.xz,
                stokes.ε.xy,
                η_vep,
                G,
                dt,
                pt_stokes.θ_dτ,
            )

            @hide_communication b_width begin # communication/computation overlap
                # (1:(nx + 1), 1:(ny + 1), 1:(nz + 1))
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

            apply_free_slip!(freeslip, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
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
            if igg.me == 0 && (verbose  || iter == iterMax)
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