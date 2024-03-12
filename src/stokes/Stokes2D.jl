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
using CUDA, AMDGPU
using ParallelStencil
# using ParallelStencil.FiniteDifferences2D
using GeoParams, LinearAlgebra, Printf

import JustRelax: elastic_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax:
    Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic, IGG
import JustRelax: compute_maxloc!, solve!
import JustRelax: mean_mpi, norm_mpi, maximum_mpi, minimum_mpi, backend

@eval @init_parallel_stencil($backend, Float64, 2)

include("../rheology/GeoParams.jl")
include("StressRotation.jl")
include("StressKernels.jl")
include("PressureKernels.jl")
include("VelocityKernels.jl")
include("StressKernels.jl")

export solve!

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,2}) where {A,B,C,D}
    τxx, τyy, τxy, τxy_c = stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, stokes.τ.xy_c
    τxx_o, τyy_o, τxy_o, τxy_o_c = stokes.τ_o.xx,
    stokes.τ_o.yy, stokes.τ_o.xy,
    stokes.τ_o.xy_c

    @parallel (@idx size(τxy)) multi_copy!(
        (τxx_o, τyy_o, τxy_o, τxy_o_c), (τxx, τyy, τxy, τxy_c)
    )
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
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    # @hide_communication b_width begin # communication/computation overlap
    compute_maxloc!(ητ, η; window=(1, 1))
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
                # apply boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
            end
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
    compute_maxloc!(ητ, η; window=(1, 1))
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
            @parallel (@idx ni) compute_τ!(
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
                # free slip boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
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
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
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
    compute_maxloc!(ητ, η; window=(1, 1))
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

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # solver loop
    wtime0 = 0.0
    λ = @zeros(ni...)
    θ = @zeros(ni...)
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

            ν = 1e-2
            @parallel (@idx ni) compute_viscosity!(
                η, ν, @strain(stokes)..., args, rheology, viscosity_cutoff
            )
            compute_maxloc!(ητ, η; window=(1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                stokes.τ.II,
                @tensor(stokes.τ_o),
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                stokes.P,
                θ,
                η,
                η_vep,
                λ,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )

            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            update_halo!(stokes.τ.xy)

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
                # apply boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
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

    stokes.P .= θ

    # accumulate plastic strain tensor
    @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)

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
    compute_maxloc!(ητ, η; window=(1, 1))
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
    @copy stokes.P0 stokes.P
    wtime0 = 0.0
    θ = @zeros(ni...)
    λ = @zeros(ni...)
    η0 = deepcopy(η)
    do_visc = true
    # GC.enable(false)

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            compute_maxloc!(ητ, η; window=(1, 1))
            update_halo!(ητ)

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...)

            @parallel (@idx ni) compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                ητ,
                rheology,
                phase_ratios.center,
                dt,
                r,
                θ_dτ,
            )

            if rem(iter, 5) == 0
                @parallel (@idx ni) compute_ρg!(ρg[2], phase_ratios.center, rheology, args)
            end

            @parallel (@idx ni .+ 1) compute_strain_rate!(
                @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )

            if rem(iter, nout) == 0
                @copy η0 η
            end
            if do_visc
                ν = 1e-2
                @parallel (@idx ni) compute_viscosity!(
                    η,
                    ν,
                    phase_ratios.center,
                    @strain(stokes)...,
                    args,
                    rheology,
                    viscosity_cutoff,
                )
            end

            @parallel (@idx ni) compute_τ_nonlinear!(
                @tensor_center(stokes.τ),
                stokes.τ.II,
                @tensor_center(stokes.τ_o),
                @strain(stokes),
                @tensor_center(stokes.ε_pl),
                stokes.EII_pl,
                stokes.P,
                θ,
                η,
                η_vep,
                λ,
                phase_ratios.center,
                tupleize(rheology), # needs to be a tuple
                dt,
                θ_dτ,
            )

            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            update_halo!(stokes.τ.xy)

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)..., θ, @stress(stokes)..., ηdτ, ρg..., ητ, _di...
                )
                # apply boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            er_η = norm_mpi(@.(log10(η) - log10(η0)))
            er_η < 1e-3 && (do_visc = false)
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx, stokes.R.Ry, θ, @stress(stokes)..., ρg..., _di...
            )
            # errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
            errs = (
                norm_mpi(stokes.R.Rx) / length(stokes.R.Rx),
                norm_mpi(stokes.R.Ry) / length(stokes.R.Ry),
                norm_mpi(stokes.R.RP) / length(stokes.R.RP),
            )
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

    stokes.P .= θ

    # accumulate plastic strain tensor
    @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)

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
    compute_maxloc!(ητ, η; window=(1, 1))
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
            @parallel (@idx ni) compute_ρg!(
                ρg[end], ϕ, rheology, (T=thermal.Tc, P=stokes.P)
            )
            @parallel (@idx ni) compute_τ_gp!(
                @tensor_center(stokes.τ),
                stokes.τ.II,
                @tensor(stokes.τ_o),
                @strain(stokes),
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
                # apply boundary conditions boundary conditions
                flow_bcs!(stokes, flow_bcs)
                update_halo!(stokes.V.Vx, stokes.V.Vy)
            end
            # apply boundary conditions boundary conditions
            # flow_bcs!(stokes, flow_bcs)
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
