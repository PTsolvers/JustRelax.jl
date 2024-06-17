module JustRelax3D

using JustRelax: JustRelax
using AMDGPU
using StaticArrays
using CellArrays
using ParallelStencil, ParallelStencil.FiniteDifferences3D
using ImplicitGlobalGrid
using GeoParams, LinearAlgebra, Printf
using MPI

import JustRelax.JustRelax3D as JR3D

import JustRelax:
    IGG,
    BackendTrait,
    CPUBackendTrait,
    AMDGPUBackendTrait,
    backend,
    CPUBackend,
    AMDGPUBackend,
    Geometry,
    @cell
import JustRelax:
    AbstractBoundaryConditions, TemperatureBoundaryConditions, FlowBoundaryConditions

@init_parallel_stencil(AMDGPU, Float64, 3)

include("../../common.jl")
include("../../stokes/Stokes3D.jl")

# Types
function JR3D.StokesArrays(::Type{AMDGPUBackend}, ni::NTuple{N,Integer}) where {N}
    return StokesArrays(ni)
end

function JR3D.velocity2displacement!(
    stokes::JustRelax.StokesArrays, ::AMDGPUBackendTrait, dt
)
    return _velocity2displacement!(stokes, dt)
end

function JR3D.displacement2velocity!(
    stokes::JustRelax.StokesArrays, ::AMDGPUBackendTrait, dt
)
    return _displacement2velocity!(stokes, dt)
end

function JR3D.ThermalArrays(::Type{AMDGPUBackend}, ni::NTuple{N,Number}) where {N}
    return ThermalArrays(ni...)
end

function JR3D.ThermalArrays(::Type{AMDGPUBackend}, ni::Vararg{Number,N}) where {N}
    return ThermalArrays(ni...)
end

function JR3D.PhaseRatio(::Type{AMDGPUBackend}, ni, num_phases)
    return PhaseRatio(ni, num_phases)
end

function JR3D.PTThermalCoeffs(
    ::Type{AMDGPUBackend}, K, ρCp, dt, di::NTuple, li::NTuple; ϵ=1e-8, CFL=0.9 / √3
)
    return PTThermalCoeffs(K, ρCp, dt, di, li; ϵ=ϵ, CFL=CFL)
end

function JR3D.PTThermalCoeffs(
    ::Type{AMDGPUBackend},
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

function JR3D.PTThermalCoeffs(
    ::Type{AMDGPUBackend},
    rheology,
    args,
    dt,
    ni,
    di::NTuple{nDim,T},
    li::NTuple{nDim,Any};
    ϵ=1e-8,
    CFL=0.9 / √3,
) where {nDim,T}
    return PTThermalCoeffs(rheology, args, dt, ni, di, li; ϵ=ϵ, CFL=CFL)
end

# Boundary conditions
function JR3D.flow_bcs!(::AMDGPUBackendTrait, stokes::JustRelax.StokesArrays, bcs)
    return _flow_bcs!(bcs, @velocity(stokes))
end

function flow_bcs!(::AMDGPUBackendTrait, stokes::JustRelax.StokesArrays, bcs)
    return _flow_bcs!(bcs, @velocity(stokes))
end

function JR3D.thermal_bcs!(::AMDGPUBackendTrait, thermal::JustRelax.ThermalArrays, bcs)
    return thermal_bcs!(thermal.T, bcs)
end

function thermal_bcs!(::AMDGPUBackendTrait, thermal::JustRelax.ThermalArrays, bcs)
    return thermal_bcs!(thermal.T, bcs)
end

# Phases
function JR3D.phase_ratios_center!(
    ::AMDGPUBackendTrait,
    phase_ratios::JustRelax.PhaseRatio,
    particles,
    grid::Geometry,
    phases,
)
    return _phase_ratios_center!(phase_ratios, particles, grid, phases)
end

# Rheology

## viscosity
function JR3D.compute_viscosity!(::AMDGPUBackendTrait, stokes, ν, args, rheology, cutoff)
    return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
end

function JR3D.compute_viscosity!(
    ::AMDGPUBackendTrait, stokes, ν, phase_ratios, args, rheology, cutoff
)
    return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, cutoff)
end

function JR3D.compute_viscosity!(η, ν, εII::ROCArray, args, rheology, cutoff)
    return compute_viscosity!(η, ν, εII, args, rheology, cutoff)
end

function compute_viscosity!(::AMDGPUBackendTrait, stokes, ν, args, rheology, cutoff)
    return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
end

function compute_viscosity!(
    ::AMDGPUBackendTrait, stokes, ν, phase_ratios, args, rheology, cutoff
)
    return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, cutoff)
end

function compute_viscosity!(η, ν, εII::ROCArray, args, rheology, cutoff)
    return compute_viscosity!(η, ν, εII, args, rheology, cutoff)
end

## Stress
function JR3D.tensor_invariant!(::AMDGPUBackendTrait, A::JustRelax.SymmetricTensor)
    return _tensor_invariant!(A)
end

## Buoyancy forces
function JR3D.compute_ρg!(ρg::ROCArray, rheology, args)
    return compute_ρg!(ρg, rheology, args)
end
function JR3D.compute_ρg!(ρg::ROCArray, phase_ratios::JustRelax.PhaseRatio, rheology, args)
    return compute_ρg!(ρg, phase_ratios, rheology, args)
end

# Interpolations
function JR3D.temperature2center!(::AMDGPUBackendTrait, thermal::JustRelax.ThermalArrays)
    return _temperature2center!(thermal)
end

function temperature2center!(::AMDGPUBackendTrait, thermal::JustRelax.ThermalArrays)
    return _temperature2center!(thermal)
end

function JR3D.vertex2center!(center::T, vertex::T) where {T<:ROCArray}
    return vertex2center!(center, vertex)
end

function JR3D.center2vertex!(vertex::T, center::T) where {T<:ROCArray}
    return center2vertex!(vertex, center)
end

function JR3D.center2vertex!(
    vertex_yz::T, vertex_xz::T, vertex_xy::T, center_yz::T, center_xz::T, center_xy::T
) where {T<:ROCArray}
    return center2vertex!(vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy)
end

function JR3D.velocity2vertex!(
    Vx_v::ROCArray, Vy_v::ROCArray, Vz_v::ROCArray, Vx::ROCArray, Vy::ROCArray, Vz::ROCArray
)
    velocity2vertex!(Vx_v, Vy_v, Vz_v, Vx, Vy, Vz)
    return nothing
end

# Solvers
function JR3D.solve!(::AMDGPUBackendTrait, stokes, args...; kwargs)
    return _solve!(stokes, args...; kwargs...)
end

function JR3D.heatdiffusion_PT!(::AMDGPUBackendTrait, thermal, args...; kwargs)
    return _heatdiffusion_PT!(thermal, args...; kwargs...)
end

# Utils
function JR3D.compute_dt(::AMDGPUBackendTrait, S::JustRelax.StokesArrays, args...)
    return _compute_dt(S, args...)
end

function JR3D.subgrid_characteristic_time!(
    subgrid_arrays,
    particles,
    dt₀::ROCArray,
    phases::JustRelax.PhaseRatio,
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

function JR3D.subgrid_characteristic_time!(
    subgrid_arrays,
    particles,
    dt₀::ROCArray,
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

function JR3D.compute_shear_heating!(::AMDGPUBackendTrait, thermal, stokes, rheology, dt)
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

function JR3D.compute_shear_heating!(
    ::AMDGPUBackendTrait, thermal, stokes, phase_ratios::JustRelax.PhaseRatio, rheology, dt
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
end
