## DIMENSION AGNOSTIC KERNELS

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

## 2D STOKES MODULE

module Stokes2D

using ImplicitGlobalGrid
using ..JustRelax
using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using GeoParams, LinearAlgebra, Printf

import JustRelax: elastic_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax:
    Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic, IGG
import JustRelax: compute_maxloc!, solve!
import JustRelax: mean_mpi, norm_mpi, maximum_mpi, minimum_mpi

export solve!

include("StressRotation.jl")
include("PressureKernels.jl")
include("VelocityKernels.jl")
include("StressKernels.jl")

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

# Stress kernels

# # viscous
# @parallel function compute_τ!(τxx, τyy, τxy, εxx, εyy, εxy, η, θ_dτ)
#     @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0 * @all(η) * @all(εxx)) * inv(θ_dτ + 1.0)
#     @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0 * @all(η) * @all(εyy)) * inv(θ_dτ + 1.0)
#     @inn(τxy) = @inn(τxy) + (-@inn(τxy) + 2.0 * @av(η) * @inn(εxy)) * inv(θ_dτ + 1.0)
#     return nothing
# end

# # visco-elastic
# @parallel function compute_τ!(
#     τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
# )
#     @all(τxx) =
#         @all(τxx) +
#         (
#             -(@all(τxx) - @all(τxx_o)) * @all(η) / (@all(G) * dt) - @all(τxx) +
#             2.0 * @all(η) * @all(εxx)
#         ) * inv(θ_dτ + @all(η) * inv(@all(G) * dt) + 1.0)
#     @all(τyy) =
#         @all(τyy) +
#         (
#             -(@all(τyy) - @all(τyy_o)) * @all(η) / (@all(G) * dt) - @all(τyy) +
#             2.0 * @all(η) * @all(εyy)
#         ) * inv(θ_dτ + @all(η) * inv(@all(G) * dt) + 1.0)
#     @inn(τxy) =
#         @inn(τxy) +
#         (
#             -(@inn(τxy) - @inn(τxy_o)) * @av(η) / (@av(G) * dt) - @inn(τxy) +
#             2.0 * @av(η) * @inn(εxy)
#         ) * inv(θ_dτ + @av(η) * inv(@av(G) * dt) + 1.0)

#     return nothing
# end

# # visco-elasto-plastic with GeoParams - with single phases
# @parallel_indices (i, j) function compute_τ_gp!(
#     τxx,
#     τyy,
#     τxy,
#     τII,
#     τxx_o,
#     τyy_o,
#     τxyv_o,
#     εxx,
#     εyy,
#     εxyv,
#     η,
#     η_vep,
#     T,
#     args_η,
#     rheology,
#     dt,
#     θ_dτ,
# )
#     #! format: off
#     # convinience closure
#     Base.@propagate_inbounds @inline gather(A) = _gather(A, i, j)
#     Base.@propagate_inbounds @inline function av(T)
#         (T[i, j] + T[i + 1, j] + T[i, j + 1] + T[i + 1, j + 1]) * 0.25
#     end
#     #! format: on

#     @inbounds begin
#         k = keys(args_η)
#         v = getindex.(values(args_η), i, j)
#         # numerics
#         # dτ_r                = 1.0 / (θ_dτ + η[i, j] / (get_G(rheology[1]) * dt) + 1.0) # original
#         dτ_r = 1.0 / (θ_dτ / η[i, j] + 1.0 / η_vep[i, j]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
#         # # Setup up input for GeoParams.jl
#         args = (; zip(k, v)..., dt=dt, T=av(T), τII_old=0.0)
#         εij_p = εxx[i, j] + 1e-25, εyy[i, j] + 1e-25, gather(εxyv) .+ 1e-25
#         τij_p_o = τxx_o[i, j], τyy_o[i, j], gather(τxyv_o)
#         phases = 1, 1, (1, 1, 1, 1) # there is only one phase...
#         # update stress and effective viscosity
#         τij, τII[i, j], ηᵢ = compute_τij(rheology, εij_p, args, τij_p_o, phases)
#         τxx[i, j] += dτ_r * (-(τxx[i, j]) + τij[1]) / ηᵢ # NOTE: from GP Tij = 2*η_vep * εij
#         τyy[i, j] += dτ_r * (-(τyy[i, j]) + τij[2]) / ηᵢ
#         τxy[i, j] += dτ_r * (-(τxy[i, j]) + τij[3]) / ηᵢ
#         η_vep[i, j] = ηᵢ
#     end

#     return nothing
# end

# # visco-elasto-plastic with GeoParams - with multiple phases
# @parallel_indices (i, j) function compute_τ_gp!(
#     τxx,
#     τyy,
#     τxy,
#     τII,
#     τxx_o,
#     τyy_o,
#     τxyv_o,
#     εxx,
#     εyy,
#     εxyv,
#     η,
#     η_vep,
#     T,
#     phase_v,
#     phase_c,
#     args_η,
#     rheology,
#     dt,
#     θ_dτ,
# )
#     #! format: off
#     # convinience closure
#     Base.@propagate_inbounds @inline gather(A) = _gather(A, i, j)
#     Base.@propagate_inbounds @inline function av(T)
#         (T[i, j] + T[i + 1, j] + T[i, j + 1] + T[i + 1, j + 1]) * 0.25
#     end
#     #! format: on

#     @inbounds begin
#         k = keys(args_η)
#         v = getindex.(values(args_η), i, j)
#         # # numerics
#         # dτ_r                = 1.0 / (θ_dτ + η[i, j] / (get_G(rheology[1]) * dt) + 1.0) # original
#         dτ_r = 1.0 / (θ_dτ / η[i, j] + 1.0 / η_vep[i, j]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
#         # # Setup up input for GeoParams.jl
#         args = (; zip(k, v)..., dt=dt, T=av(T), τII_old=0.0)
#         εij_p = εxx[i, j] + 1e-25, εyy[i, j] + 1e-25, gather(εxyv) .+ 1e-25
#         τij_p_o = τxx_o[i, j], τyy_o[i, j], gather(τxyv_o)
#         phases = phase_c[i, j], phase_c[i, j], gather(phase_v) # for now hard-coded for a single phase
#         # update stress and effective viscosity
#         τij, τII[i, j], ηᵢ = compute_τij(rheology, εij_p, args, τij_p_o, phases)
#         τxx[i, j] += dτ_r * (-(τxx[i, j]) + τij[1]) / ηᵢ # NOTE: from GP Tij = 2*η_vep * εij
#         τyy[i, j] += dτ_r * (-(τyy[i, j]) + τij[2]) / ηᵢ
#         τxy[i, j] += dτ_r * (-(τxy[i, j]) + τij[3]) / ηᵢ
#         η_vep[i, j] = ηᵢ
#     end

#     return nothing
# end

# # single phase visco-elasto-plastic flow
# @parallel_indices (i, j) function compute_τ_nonlinear!(
#     τxx,
#     τyy,
#     τxy,
#     τII,
#     τxx_old,
#     τyy_old,
#     τxyv_old,
#     εxx,
#     εyy,
#     εxyv,
#     P,
#     η,
#     η_vep,
#     λ,
#     rheology,
#     dt,
#     θ_dτ,
# )
#     idx = i, j

#     # numerics
#     ηij = η[i, j]
#     _Gdt = inv(get_G(rheology[1]) * dt)
#     dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

#     # get plastic paremeters (if any...)
#     is_pl, C, sinϕ, η_reg = plastic_params(rheology[1])
#     plastic_parameters = (; is_pl, C, sinϕ, η_reg)

#     τ = τxx, τyy, τxy
#     τ_old = τxx_old, τyy_old, τxyv_old
#     ε = εxx, εyy, εxyv

#     _compute_τ_nonlinear!(
#         τ, τII, τ_old, ε, P, ηij, η_vep, λ, dτ_r, _Gdt, plastic_parameters, idx...
#     )

#     return nothing
# end

# # multi phase visco-elasto-plastic flow, where phases are defined in the cell center
# @parallel_indices (i, j) function compute_τ_nonlinear!(
#     τxx,
#     τyy,
#     τxy,
#     τII,
#     τxx_old,
#     τyy_old,
#     τxyv_old,
#     εxx,
#     εyy,
#     εxyv,
#     P,
#     η,
#     η_vep,
#     λ,
#     phase_ratios::PhaseRatio,
#     rheology,
#     dt,
#     θ_dτ,
# )
#     idx = i, j

#     # numerics
#     ηij = @inbounds η[i, j]
#     phase = @inbounds phase_ratios[i, j]
#     G = fn_ratio(get_G, MatParam, phase)
#     _Gdt = inv(G * dt)
#     dτ_r = compute_dτ_r(θ_dτ, ηij, _Gdt)

#     # get plastic paremeters (if any...)
#     is_pl, C, sinϕ, η_reg = plastic_params(rheology, phase)

#     plastic_parameters = (; is_pl, C, sinϕ, η_reg)

#     τ = τxx, τyy, τxy
#     τ_old = τxx_old, τyy_old, τxyv_old
#     ε = εxx, εyy, εxyv

#     _compute_τ_nonlinear!(
#         τ,
#         τII,
#         τ_old,
#         ε,
#         P,
#         ηij,
#         η_vep,
#         phase_ratios,
#         λ,
#         dτ_r,
#         _Gdt,
#         plastic_parameters,
#         idx...,
#     )

#     return nothing
# end

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
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η)
    update_halo!(ητ)
    # end

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
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.RP, stokes.∇V, η, K, dt, r, θ_dτ
            )
            @parallel (@idx ni .+ 1) compute_τ!(
                @stress(stokes)..., @strain(stokes)..., η, θ_dτ
            )

            @hide_communication b_width begin
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
                    ητ,
                    _dx,
                    _dy,
                )
                update_halo!(@velocity(stokes)...)
            end
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )
            Vmin, Vmax = extrema(stokes.V.Vx)
            Pmin, Pmax = extrema(stokes.P)
            push!(
                norm_Rx,
                norm_mpi(stokes.R.Rx) / (Pmax - Pmin) * lx / sqrt(length(stokes.R.Rx)),
            )
            push!(
                norm_Ry,
                norm_mpi(stokes.R.Ry) / (Pmax - Pmin) * lx / sqrt(length(stokes.R.Ry)),
            )
            push!(
                norm_∇V, norm_mpi(stokes.∇V) / (Vmax - Vmin) * lx / sqrt(length(stokes.∇V))
            )
            err = maximum_mpi(norm_Rx[end], norm_Ry[end], norm_∇V[end])
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
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window=(1, 1, 1))
    update_halo!(ητ)
    # end

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
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, η, K, dt, r, θ_dτ
            )
            @parallel (@idx ni .+ 1) compute_τ!(
                @stress(stokes)...,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                η,
                G,
                θ_dτ,
                dt,
            )
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
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
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )
            errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
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

    if !isinf(dt) # if dt is inf, then we are in the non-elastic case
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@velocity(stokes), @tensor(stokes.τ_o), _di, dt)
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

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    η_vep,
    rheology::MaterialParams,
    args,
    dt,
    igg::IGG;
    viscosity_cutoff=(1e16, 1e24),
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 0),
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _di = inv.(di)
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η)
    update_halo!(ητ)
    # end

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
    λ = @zeros(ni...)
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel compute_P!(
                stokes.P, stokes.P0, stokes.R.RP, stokes.∇V, η, Kb, dt, r, θ_dτ
            )

            @parallel (@idx ni) compute_ρg!(ρg[2], rheology, args)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            ν = 0.01
            @parallel (@idx ni) compute_viscosity!(
                η, ν, @strain(stokes)..., args, rheology, viscosity_cutoff
            )
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
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )

            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
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
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
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

    if !isinf(dt) # if dt is inf, then we are in the non-elastic case
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@velocity(stokes), @tensor(stokes.τ_o), _di, dt)
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

## With phase ratios 

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    η_vep,
    phase_ratios::PhaseRatio,
    rheology,
    args,
    dt,
    igg::IGG;
    viscosity_cutoff=(1e16, 1e24),
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 0),
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _di = inv.(di)
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η)
    update_halo!(ητ)
    # end

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]
    sizehint!(norm_Rx, Int(iterMax))
    sizehint!(norm_Ry, Int(iterMax))
    sizehint!(norm_∇V, Int(iterMax))
    sizehint!(err_evo1, Int(iterMax))
    sizehint!(err_evo2, Int(iterMax))

    # solver loop
    wtime0 = 0.0
    λ = @zeros(ni...)
    # while iter < 2 
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)

            @parallel (@idx ni) compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                rheology,
                phase_ratios.center,
                dt,
                r,
                θ_dτ,
            )

            @parallel (@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, args)

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            ν = 0.05
            @parallel (@idx ni) compute_viscosity!(
                η,
                ν,
                phase_ratios.center,
                @strain(stokes)...,
                args,
                rheology,
                viscosity_cutoff,
            )
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
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )

            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
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
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )
            errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
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

    if !isinf(dt) # if dt is inf, then we are in the non-elastic case
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@velocity(stokes), @tensor(stokes.τ_o), _di, dt)
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
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)
    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η)
    update_halo!(ητ)
    # end

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
            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)
            @parallel (@idx ni) compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                rheology,
                phase_c,
                dt,
                r,
                θ_dτ,
            )
            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
            @parallel (@idx ni) compute_ρg!(ρg[2], ϕ, rheology, (T=thermal.Tc, P=stokes.P))
            @parallel (@idx ni) compute_τ_gp!(
                @tensor_center(stokes.τ)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                η,
                η_vep,
                thermal.T,
                phase_v,
                phase_c,
                args_η,
                rheology, # needs to be a tuple
                dt,
                θ_dτ,
            )
            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
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
                stokes.R.Rx, stokes.R.Ry, stokes.P, @stress(stokes)..., ρg..., _di...
            )
            errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
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

    if !isinf(dt) # if dt is inf, then we are in the non-elastic case 
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@velocity(stokes), @tensor(stokes.τ_o), _di, dt)
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
