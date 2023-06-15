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

module Stokes2D

using ImplicitGlobalGrid
using ..JustRelax
using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using GeoParams, LinearAlgebra, Printf

import JustRelax: stress, strain, elastic_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax:
    Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic, IGG
import JustRelax: compute_maxloc!, solve!, @tuple

# import ..Stokes2D: compute_P!, compute_V!, compute_strain_rate!

export solve!, compute_ρg!

include("StressRotation.jl")

# Helper kernels

Base.@propagate_inbounds @inline  _d_xa(A<:AbstractArray{T,2}, i, j, _dx) where T = (A[i + 1, j] - A[i, j]) * _dx
Base.@propagate_inbounds @inline  _d_ya(A<:AbstractArray{T,2}, i, j, _dy) where T = (A[i, j + 1] - A[i, j]) * _dy
Base.@propagate_inbounds @inline  _d_xi(A<:AbstractArray{T,2}, i, j, _dx) where T = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
Base.@propagate_inbounds @inline  _d_yi(A<:AbstractArray{T,2}, i, j, _dy) where T = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
Base.@propagate_inbounds @inline _av_xa(A<:AbstractArray{T,2}, i, j) where T = (A[i + 1, j] + A[i, j]) * 0.5
Base.@propagate_inbounds @inline _av_ya(A<:AbstractArray{T,2}, i, j) where T = (A[i, j + 1] + A[i, j]) * 0.5

## 2D ELASTIC KERNELS

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,2}) where {A,B,C,D}
    τxx, τyy, τxy, τxy_c = stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, stokes.τ.xy_c
    τxx_o, τyy_o, τxy_o, τxy_o_c = stokes.τ_o.xx,
    stokes.τ_o.yy, stokes.τ_o.xy,
    stokes.τ_o.xy_c
    @parallel update_τ_o!(τxx_o, τyy_o, τxy_o, τxy_o_c, τxx, τyy, τxy, τxy_c)
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

@parallel_indices (i, j) function compute_P!(
    P, P_old, RP, ∇V, η, rheology::NTuple{N,MaterialParams}, phase, dt, r, θ_dτ
) where {N}
    @inbounds begin
        _Kdt = inv(get_Kb(rheology, phase[i, j]) * dt)
        RP[i, j] = RP_ij = -∇V[i, j] - (P[i, j] - P_old[i, j]) * _Kdt
        P[i, j] += RP_ij * inv(inv(r / θ_dτ * η[i, j]) + _Kdt)
    end
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
    # Closures
    @inline  d_xa(A) = _d_xa(A, i, j, _dx)
    @inline  d_ya(A) = _d_ya(A, i, j, _dy)
    @inline  d_xi(A) = _d_xi(A, i, j, _dx)
    @inline  d_yi(A) = _d_yi(A, i, j, _dy)
    @inline av_xa(A) = _av_xa(A,i,j)
    @inline av_ya(A) = _av_ya(A,i,j)

    @inbounds begin
        if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
            Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
        end
        if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
            Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
        end
    end
    return nothing
end

@parallel_indices (i, j) function compute_V_Res!(
    Vx, Vy, Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, ητ, ηdτ, _dx, _dy
)

    # Closures
    @inline  d_xa(A) = _d_xa(A, i, j, _dx)
    @inline  d_ya(A) = _d_ya(A, i, j, _dy)
    @inline  d_xi(A) = _d_xi(A, i, j, _dx)
    @inline  d_yi(A) = _d_yi(A, i, j, _dy)
    @inline av_xa(A) = _av_xa(A,i,j)
    @inline av_ya(A) = _av_ya(A,i,j)

    @inbounds begin
        if all((i, j) .≤ size(Rx))
            R = Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
            Vx[i + 1, j + 1] += R * ηdτ / av_xa(ητ)
        end
        if all((i, j) .≤ size(Ry))
            R = Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
            Vy[i + 1, j + 1] += R * ηdτ / av_ya(ητ)
        end
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
        ) * inv(θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @all(τyy) =
        @all(τyy) +
        (
            -(@all(τyy) - @all(τyy_o)) * @all(η) / (@all(G) * dt) - @all(τyy) +
            2.0 * @all(η) * @all(εyy)
        ) * inv(θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @inn(τxy) =
        @inn(τxy) +
        (
            -(@inn(τxy) - @inn(τxy_o)) * @av(η) / (@av(G) * dt) - @inn(τxy) +
            2.0 * @av(η) * @inn(εxy)
        ) * inv(θ_dτ + @av(η) / (@av(G) * dt) + 1.0)

    return nothing
end

# visco-elasto-plastic with GeoParams - with multiple phases
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
    phase_v,
    phase_c,
    args_η,
    rheology,
    dt,
    θ_dτ,
)
    #! format: off
    # convinience closure
    Base.@propagate_inbounds @inline function gather(A)
        A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1]
    end
    Base.@propagate_inbounds @inline function av(T)
        (T[i, j] + T[i + 1, j] + T[i, j + 1] + T[i + 1, j + 1]) * 0.25
    end
    #! format: on

    @inbounds begin
        k = keys(args_η)
        v = getindex.(values(args_η), i, j)
        # # numerics
        # dτ_r                = 1.0 / (θ_dτ + η[i, j] / (get_G(rheology[1]) * dt) + 1.0) # original
        dτ_r = 1.0 / (θ_dτ / η[i, j] + 1.0 / η_vep[i, j]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
        # # Setup up input for GeoParams.jl
        args = (; zip(k, v)..., dt=dt, T=av(T), τII_old=0.0)
        εij_p = εxx[i, j] + 1e-25, εyy[i, j] + 1e-25, gather(εxyv) .+ 1e-25
        τij_p_o = τxx_o[i, j], τyy_o[i, j], gather(τxyv_o)
        phases = phase_c[i, j], phase_c[i, j], gather(phase_v) # for now hard-coded for a single phase
        # update stress and effective viscosity
        τij, τII[i, j], ηᵢ = compute_τij(rheology, εij_p, args, τij_p_o, phases)
        τxx[i, j] += dτ_r * (-(τxx[i, j]) + τij[1]) / ηᵢ # NOTE: from GP Tij = 2*η_vep * εij
        τyy[i, j] += dτ_r * (-(τyy[i, j]) + τij[2]) / ηᵢ
        τxy[i, j] += dτ_r * (-(τxy[i, j]) + τij[3]) / ηᵢ
        η_vep[i, j] = ηᵢ
    end

    return nothing
end

#update density depending on melt fraction and number of phases
@parallel_indices (i, j) function compute_ρg!(ρg, ϕ, rheology, args)
    i1, j1 = i + 1, j + 1
    i2 = i + 2
    @inline av(T) = 0.25 * (T[i1, j] + T[i2, j] + T[i1, j1] + T[i2, j1])

    ρg[i, j] =
        compute_density_ratio(
            (1 - ϕ[i, j], ϕ[i, j], 0.0), rheology, (; T=av(args.T), P=args.P[i, j])
        ) * compute_gravity(rheology[1])
    return nothing
end

## 2D VISCO-ELASTIC STOKES SOLVER 

# viscous solver
function JustRelax.solve!(
    stokes::StokesArrays{Viscous,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs::FlowBoundaryConditions,
    ρg,
    η,
    K,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 1),
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
    ni = nx, ny = size(P)

    ρgx, ρgy = ρg
    P_old = deepcopy(P)

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end

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
            # apply_free_slip!(freeslip, Vx, Vy)
            @parallel (@idx ni) compute_∇V!(∇V, Vx, Vy, _dx, _dy)
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy
            )
            @parallel compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
            @parallel (@idx ni .+ 1) compute_τ!(τxx, τyy, τxy, εxx, εyy, εxy, η, θ_dτ)

            @hide_communication b_width begin
                @parallel compute_V!(Vx, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy)
                update_halo!(Vx, Vy)
            end
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy)
            Vmin, Vmax = extrema(Vx)
            Pmin, Pmax = extrema(P)
            push!(norm_Rx, norm(Rx) / (Pmax - Pmin) * lx / sqrt(length(Rx)))
            push!(norm_Ry, norm(Ry) / (Pmax - Pmin) * lx / sqrt(length(Ry)))
            push!(norm_∇V, norm(∇V) / (Vmax - Vmin) * lx / sqrt(length(∇V)))
            err = max(norm_Rx[end], norm_Ry[end], norm_∇V[end])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
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
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 1),
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    ni = nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end

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
            @parallel (@idx ni) compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                stokes.∇V,
                stokes.V.Vx,
                stokes.V.Vy,
                _di...,
            )
            @parallel compute_P!(stokes.P, P_old, stokes.R.RP, stokes.∇V, η, K, dt, r, θ_dτ)
            @parallel (@idx ni .+ 1) compute_τ!(
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
            @hide_communication b_width begin # communication/computation overlap
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
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
            # free slip boundary conditions
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
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
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            # isnan(err) && error("NaN(s)")   #not working yet
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    if -Inf < dt < Inf
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
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
    rheology::MaterialParams,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 1),
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    ni = nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)
    z = LinRange(di[2] * 0.5, 1.0 - di[2] * 0.5, ny)

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end

    Kb = get_Kb(rheology)

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
            @parallel (@idx ni) compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)
            @parallel compute_P!(
                stokes.P, P_old, stokes.R.RP, stokes.∇V, η, Kb, dt, r, θ_dτ
            )
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                stokes.∇V,
                stokes.V.Vx,
                stokes.V.Vy,
                _di...,
            )
            @parallel (@idx ni) compute_τ_gp!(
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
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )
            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @hide_communication b_width begin # communication/computation overlap
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
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
            # apply boundary conditions boundary conditions
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
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
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    if -Inf < dt < Inf
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
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

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    thermal::ThermalArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ϕ,
    ρg,
    η,
    η_vep,
    phase_v,
    phase_c,
    args_η,
    rheology::NTuple{N,MaterialParams},
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 1),
    verbose=true,
) where {A,B,C,D,N,T}

    # unpack
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    ni = nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)
    z = LinRange(di[2] * 0.5, 1.0 - di[2] * 0.5, ny)
    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end

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
            @parallel (@idx ni) compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)

            @parallel (@idx ni) compute_P!(
                stokes.P, P_old, stokes.R.RP, stokes.∇V, η, rheology, phase_c, dt, r, θ_dτ
            )
            @parallel (@idx ni) compute_strain_rate!(
                @tuple(stokes.ε)..., stokes.∇V, @tuple(stokes.V)..., _di...
            )
            @parallel (@idx ni) compute_ρg!(ρg[2], ϕ, rheology, (T=thermal.T, P=stokes.P))
            @parallel (@idx ni) compute_τ_gp!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy_c,
                stokes.τ.II,
                @tuple(stokes.τ_o)...,
                @tuple(stokes.ε)...,
                η,
                η_vep,
                z,
                thermal.T,
                phase_v,
                phase_c,
                args_η,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )
            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @hide_communication b_width begin # communication/computation overlap
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
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
            # apply boundary conditions boundary conditions
            # apply_free_slip!(freeslip, stokes.V.Vx, stokes.V.Vy)
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
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
            if igg.me == 0 && (verbose || iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    if -Inf < dt < Inf
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
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
