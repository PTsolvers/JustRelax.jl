module JustRelax2D

using JustRelax: JustRelax
using CUDA
using JustPIC, JustPIC._2D
using StaticArrays
using CellArrays
using ParallelStencil, ParallelStencil.FiniteDifferences2D
using ImplicitGlobalGrid
using GeoParams, LinearAlgebra, Printf
using MPI

import JustRelax.JustRelax2D as JR2D

import JustRelax:
    IGG, BackendTrait, CPUBackendTrait, CUDABackendTrait, backend, CPUBackend, Geometry

import JustRelax:
    AbstractBoundaryConditions,
    TemperatureBoundaryConditions,
    AbstractFlowBoundaryConditions,
    DisplacementBoundaryConditions,
    VelocityBoundaryConditions,
    apply_dirichlet,
    apply_dirichlet!

import JustRelax: normal_stress, shear_stress, shear_vorticity, unwrap

import JustPIC._2D: numphases, nphases

__init__() = @init_parallel_stencil(CUDA, Float64, 2)

include("../../common.jl")
include("../../stokes/Stokes2D.jl")
include("../../variational_stokes/Stokes2D.jl")

# Types

function JR2D.StokesArrays(::Type{CUDABackend}, ni::NTuple{N,Integer}) where {N}
    return StokesArrays(ni)
end

function JR2D.ThermalArrays(::Type{CUDABackend}, ni::NTuple{N,Number}) where {N}
    return ThermalArrays(ni...)
end

function JR2D.ThermalArrays(::Type{CUDABackend}, ni::Vararg{Number,N}) where {N}
    return ThermalArrays(ni...)
end

function JR2D.WENO5(::Type{CUDABackend}, method::Val{T}, ni::NTuple{N,Integer}) where {N,T}
    return WENO5(method, tuple(ni...))
end

function JR2D.RockRatio(::Type{CUDABackend}, ni::NTuple{N,Integer}) where {N}
    return RockRatio(ni...)
end

function JR2D.RockRatio(::Type{CUDABackend}, ni::Vararg{Integer,N}) where {N}
    return RockRatio(ni...)
end

function JR2D.PTThermalCoeffs(
    ::Type{CUDABackend}, K, ρCp, dt, di::NTuple, li::NTuple; ϵ=1e-8, CFL=0.9 / √3
)
    return PTThermalCoeffs(K, ρCp, dt, di, li; ϵ=ϵ, CFL=CFL)
end

function JR2D.PTThermalCoeffs(
    ::Type{CUDABackend},
    rheology,
    phase_ratios,
    args,
    dt,
    ni,
    di::NTuple{nDim,T},
    li::NTuple{nDim,Any};
    ϵ=1e-8,
    CFL=0.9 / √3,
) where {nDim,T}
    return PTThermalCoeffs(rheology, phase_ratios, args, dt, ni, di, li; ϵ=ϵ, CFL=CFL)
end

function JR2D.PTThermalCoeffs(
    ::Type{CUDABackend},
    rheology::MaterialParams,
    args,
    dt,
    ni,
    di::NTuple,
    li::NTuple;
    ϵ=1e-8,
    CFL=0.9 / √3,
)
    return PTThermalCoeffs(rheology, args, dt, ni, di, li; ϵ=ϵ, CFL=CFL)
end

function JR2D.update_thermal_coeffs!(
    pt_thermal::JustRelax.PTThermalCoeffs{T,<:CuArray}, rheology, phase_ratios, args, dt
) where {T}
    ni = size(pt_thermal.dτ_ρ)
    @parallel (@idx ni) compute_pt_thermal_arrays!(
        pt_thermal.θr_dτ,
        pt_thermal.dτ_ρ,
        rheology,
        phase_ratios.center,
        args,
        pt_thermal.max_lxyz,
        pt_thermal.Vpdτ,
        inv(dt),
    )
    return nothing
end

function JR2D.update_thermal_coeffs!(
    pt_thermal::JustRelax.PTThermalCoeffs{T,<:CuArray}, rheology, args, dt
) where {T}
    ni = size(pt_thermal.dτ_ρ)
    @parallel (@idx ni) compute_pt_thermal_arrays!(
        pt_thermal.θr_dτ,
        pt_thermal.dτ_ρ,
        rheology,
        args,
        pt_thermal.max_lxyz,
        pt_thermal.Vpdτ,
        inv(dt),
    )
    return nothing
end

function JR2D.update_thermal_coeffs!(
    pt_thermal::JustRelax.PTThermalCoeffs{T,<:CuArray}, rheology, ::Nothing, args, dt
) where {T}
    ni = size(pt_thermal.dτ_ρ)
    @parallel (@idx ni) compute_pt_thermal_arrays!(
        pt_thermal.θr_dτ,
        pt_thermal.dτ_ρ,
        rheology,
        args,
        pt_thermal.max_lxyz,
        pt_thermal.Vpdτ,
        inv(dt),
    )
    return nothing
end

# Boundary conditions
function JR2D.flow_bcs!(
    ::CUDABackendTrait, stokes::JustRelax.StokesArrays, bcs::VelocityBoundaryConditions
)
    return _flow_bcs!(bcs, @velocity(stokes))
end

function flow_bcs!(
    ::CUDABackendTrait, stokes::JustRelax.StokesArrays, bcs::VelocityBoundaryConditions
)
    return _flow_bcs!(bcs, @velocity(stokes))
end

function JR2D.flow_bcs!(
    ::CUDABackendTrait, stokes::JustRelax.StokesArrays, bcs::DisplacementBoundaryConditions
)
    return _flow_bcs!(bcs, @displacement(stokes))
end

function flow_bcs!(
    ::CUDABackendTrait, stokes::JustRelax.StokesArrays, bcs::DisplacementBoundaryConditions
)
    return _flow_bcs!(bcs, @displacement(stokes))
end

function JR2D.thermal_bcs!(::CUDABackendTrait, thermal::JustRelax.ThermalArrays, bcs)
    return thermal_bcs!(thermal.T, bcs)
end

function thermal_bcs!(::CUDABackendTrait, thermal::JustRelax.ThermalArrays, bcs)
    return thermal_bcs!(thermal.T, bcs)
end

# Rheology

## viscosity
function JR2D.compute_viscosity!(::CUDABackendTrait, stokes, ν, args, rheology, cutoff)
    return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
end

function JR2D.compute_viscosity!(
    ::CUDABackendTrait, stokes, ν, phase_ratios, args, rheology, cutoff
)
    return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, cutoff)
end

function JR2D.compute_viscosity!(
    ::CUDABackendTrait, stokes, ν, phase_ratios, args, rheology, air_phase, cutoff
)
    return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, air_phase, cutoff)
end

function JR2D.compute_viscosity!(η, ν, εII::CuArray, args, rheology, cutoff)
    return compute_viscosity!(η, ν, εII, args, rheology, cutoff)
end

function compute_viscosity!(::CUDABackendTrait, stokes, ν, args, rheology, cutoff)
    return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
end

function compute_viscosity!(
    ::CUDABackendTrait, stokes, ν, phase_ratios, args, rheology, air_phase, cutoff
)
    return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, air_phase, cutoff)
end

function compute_viscosity!(η, ν, εII::CuArray, args, rheology, cutoff)
    return compute_viscosity!(η, ν, εII, args, rheology, cutoff)
end

## Stress
function JR2D.tensor_invariant!(::CUDABackendTrait, A::JustRelax.SymmetricTensor)
    return _tensor_invariant!(A)
end

## Buoyancy forces
function JR2D.compute_ρg!(ρg::Union{CuArray,NTuple{N,CuArray}}, rheology, args) where {N}
    return compute_ρg!(ρg, rheology, args)
end

function JR2D.compute_ρg!(
    ρg::Union{CuArray,NTuple{N,CuArray}}, phase_ratios::JustPIC.PhaseRatios, rheology, args
) where {N}
    return compute_ρg!(ρg, phase_ratios, rheology, args)
end

## Melt fraction
function JR2D.compute_melt_fraction!(ϕ::CuArray, rheology, args)
    return compute_melt_fraction!(ϕ, rheology, args)
end

function JR2D.compute_melt_fraction!(
    ϕ::CuArray, phase_ratios::JustPIC.PhaseRatios, rheology, args
)
    return compute_melt_fraction!(ϕ, phase_ratios, rheology, args)
end

# Interpolations
function JR2D.temperature2center!(::CUDABackendTrait, thermal::JustRelax.ThermalArrays)
    return _temperature2center!(thermal)
end

function temperature2center!(::CUDABackendTrait, thermal::JustRelax.ThermalArrays)
    return _temperature2center!(thermal)
end

function JR2D.vertex2center!(center::T, vertex::T) where {T<:CuArray}
    return vertex2center!(center, vertex)
end

function JR2D.center2vertex!(vertex::T, center::T) where {T<:CuArray}
    return center2vertex!(vertex, center)
end

function JR2D.center2vertex!(
    vertex_yz::T, vertex_xz::T, vertex_xy::T, center_yz::T, center_xz::T, center_xy::T
) where {T<:CuArray}
    center2vertex!(vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy)
    return nothing
end

function JR2D.velocity2center!(Vx_c::T, Vy_c::T, Vx::T, Vy::T) where {T<:CuArray}
    velocity2center!(Vx_c, Vy_c, Vx, Vy)
    return nothing
end

function JR2D.velocity2vertex!(Vx_v::CuArray, Vy_v::CuArray, Vx::CuArray, Vy::CuArray)
    velocity2vertex!(Vx_v, Vy_v, Vx, Vy)
    return nothing
end

function JR2D.velocity2displacement!(::CUDABackendTrait, stokes::JustRelax.StokesArrays, dt)
    return _velocity2displacement!(stokes, dt)
end

function velocity2displacement!(::CUDABackendTrait, stokes::JustRelax.StokesArrays, dt)
    return _velocity2displacement!(stokes, dt)
end

function JR2D.displacement2velocity!(::CUDABackendTrait, stokes::JustRelax.StokesArrays, dt)
    return _displacement2velocity!(stokes, dt)
end

function displacement2velocity!(::CUDABackendTrait, stokes::JustRelax.StokesArrays, dt)
    return _displacement2velocity!(stokes, dt)
end

# Solvers
function JR2D.solve!(::CUDABackendTrait, stokes, args...; kwargs)
    return _solve!(stokes, args...; kwargs...)
end

function JR2D.solve_VariationalStokes!(::CUDABackendTrait, stokes, args...; kwargs)
    return _solve_VS!(stokes, args...; kwargs...)
end

function JR2D.heatdiffusion_PT!(::CUDABackendTrait, thermal, args...; kwargs)
    return _heatdiffusion_PT!(thermal, args...; kwargs...)
end

# Utils
function JR2D.compute_dt(::CUDABackendTrait, S::JustRelax.StokesArrays, args...)
    return _compute_dt(S, args...)
end

# Subgrid diffusion

function JR2D.subgrid_characteristic_time!(
    subgrid_arrays,
    particles,
    dt₀::CuArray,
    phases::JustPIC.PhaseRatios,
    rheology,
    thermal::JustRelax.ThermalArrays,
    stokes::JustRelax.StokesArrays,
    xci,
    di,
)
    ni = size(stokes.P)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases.center, rheology, thermal.Tc, stokes.P, di
    )
    return nothing
end

function JR2D.subgrid_characteristic_time!(
    subgrid_arrays,
    particles,
    dt₀::CuArray,
    phases::AbstractArray{Int,N},
    rheology,
    thermal::JustRelax.ThermalArrays,
    stokes::JustRelax.StokesArrays,
    xci,
    di,
) where {N}
    ni = size(stokes.P)
    @parallel (@idx ni) subgrid_characteristic_time!(
        dt₀, phases, rheology, thermal.Tc, stokes.P, di
    )
    return nothing
end

# shear heating

function JR2D.compute_shear_heating!(::CUDABackendTrait, thermal, stokes, rheology, dt)
    ni = size(thermal.shear_heating)
    @parallel (ni) compute_shear_heating_kernel!(
        thermal.shear_heating,
        @tensor_center(stokes.τ),
        @tensor_center(stokes.τ_o),
        @strain(stokes),
        rheology,
        dt,
    )
    return nothing
end

function JR2D.compute_shear_heating!(
    ::CUDABackendTrait, thermal, stokes, phase_ratios::JustPIC.PhaseRatios, rheology, dt
)
    ni = size(thermal.shear_heating)
    @parallel (@idx ni) compute_shear_heating_kernel!(
        thermal.shear_heating,
        @tensor_center(stokes.τ),
        @tensor_center(stokes.τ_o),
        @strain(stokes),
        phase_ratios.center,
        rheology,
        dt,
    )
    return nothing
end

# WENO-5 advection
function JR2D.WENO_advection!(u::CuArray, Vxi::NTuple, weno, di, dt)
    return WENO_advection!(u, Vxi, weno, di, dt)
end

# stress rotation on particles

function JR2D.rotate_stress_particles!(
    τ::NTuple, ω::NTuple, particles::Particles{CUDABackend}, dt; method::Symbol=:matrix
)
    fn = if method === :matrix
        rotate_stress_particles_rotation_matrix!

    elseif method === :jaumann
        rotate_stress_particles_jaumann!

    else
        error("Unknown method: $method. Valid methods are :matrix and :jaumann")
    end
    @parallel (@idx size(particles.index)) fn(τ..., ω..., particles.index, dt)

    return nothing
end

# rock ratios

function JR2D.update_rock_ratio!(
    ϕ::JustRelax.RockRatio{CuArray{T,nD,D},2}, phase_ratios, air_phase
) where {T,nD,D}
    update_rock_ratio!(ϕ, phase_ratios, air_phase)
    return nothing
end

function JR2D.stress2grid!(
    stokes, τ_particles::JustRelax.StressParticles{CUDABackend}, xvi, xci, particles
)
    stress2grid!(stokes, τ_particles, xvi, xci, particles)
    return nothing
end

function JR2D.rotate_stress!(
    τ_particles::JustRelax.StressParticles{CUDABackend}, stokes, particles, xci, xvi, dt
)
    rotate_stress!(τ_particles, stokes, particles, xci, xvi, dt)
    return nothing
end

end
