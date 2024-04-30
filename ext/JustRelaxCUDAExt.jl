module JustRelaxCUDAExt

using CUDA
using JustRelax: JustRelax
import JustRelax: PTArray

JustRelax.PTArray(::Type{CUDABackend}) = CuArray

module JustRelax2D

    using JustRelax: JustRelax
    using CUDA
    using StaticArrays
    using CellArrays
    using ParallelStencil, ParallelStencil.FiniteDifferences2D
    using ImplicitGlobalGrid
    using GeoParams, LinearAlgebra, Printf
    using MPI

    import JustRelax.JustRelax2D as JR2D

    import JustRelax:
        IGG, BackendTrait, CPUBackendTrait, CUDABackendTrait, backend, CPUBackend, Geometry, @cell
    import JustRelax: AbstractBoundaryConditions, TemperatureBoundaryConditions, FlowBoundaryConditions

    @init_parallel_stencil(CUDA, Float64, 2)

    include("../src/common.jl")
    include("../src/stokes/Stokes2D.jl")

    @inline backend(::CuArray) = CUDABackendTrait()
    @inline backend(::Type{<:CuArray}) = CUDABackendTrait()

    # Types
    function JR2D.StokesArrays(
        ::Type{CUDABackend}, ni::NTuple{N,Integer}
    ) where {N}
        return StokesArrays(ni)
    end

    function JR2D.ThermalArrays(
        ::Type{CUDABackend}, ni::NTuple{N,Number}
    ) where {N}
        return ThermalArrays(ni...)
    end

    function JR2D.ThermalArrays(
        ::Type{CUDABackend}, ni::Vararg{Number,N}
    ) where {N}
        return ThermalArrays(ni...)
    end

    function JR2D.PhaseRatio(::Type{CUDABackend}, ni, num_phases)
        return PhaseRatio(ni, num_phases)
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
        return PTThermalCoeffs(
            rheology, phase_ratios, args, dt, ni, di, li; ϵ=ϵ, CFL=CFL
        )
    end

    function JR2D.PTThermalCoeffs(
        ::Type{CUDABackend},
        rheology,
        args,
        dt,
        ni,
        di::NTuple{nDim,T},
        li::NTuple{nDim,Any};
        ϵ=1e-8,
        CFL=0.9 / √3,
    ) where {nDim,T}
        return PTThermalCoeffs(
            rheology, args, dt, ni, di, li; ϵ=ϵ, CFL=CFL
        )
    end

    # Boundary conditions
    function JR2D.flow_bcs!(::CUDABackendTrait, stokes::JustRelax.StokesArrays, bcs)
        return _flow_bcs!(bcs, @velocity(stokes))
    end

    function flow_bcs!(::CUDABackendTrait, stokes::JustRelax.StokesArrays, bcs)
        return _flow_bcs!(bcs, @velocity(stokes))
    end

    function JR2D.thermal_bcs!(::CUDABackendTrait, thermal::JustRelax.ThermalArrays, bcs)
        return thermal_bcs!(thermal.T, bcs)
    end
    
    function thermal_bcs!(::CUDABackendTrait, thermal::JustRelax.ThermalArrays, bcs)
        return thermal_bcs!(thermal.T, bcs)
    end

    # Phases
    function JR2D.phase_ratios_center(
        ::CUDABackendTrait, phase_ratios::JustRelax.PhaseRatio, particles, grid::Geometry, phases
    )
        return _phase_ratios_center(phase_ratios, particles, grid, phases)
    end

    # Rheology
    ## viscosity
    function JR2D.compute_viscosity!(
        ::CUDABackendTrait, stokes, ν, args, rheology, cutoff
    )
        return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
    end
    function JR2D.compute_viscosity!(
        ::CUDABackendTrait, stokes, ν, phase_ratios, args, rheology, cutoff
    )
        return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, cutoff)
    end
    function JR2D.compute_viscosity!(
        η, ν, εII::CuArray, args, rheology, cutoff
    )
        return compute_viscosity!(η, ν, εII, args, rheology, cutoff)
    end

    function compute_viscosity!(::CUDABackendTrait, stokes, ν, args, rheology, cutoff)
        return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
    end
    function compute_viscosity!(
        ::CUDABackendTrait, stokes, ν, phase_ratios, args, rheology, cutoff
    )
        return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, cutoff)
    end
    function compute_viscosity!(η, ν, εII::CuArray, args, rheology, cutoff)
        return compute_viscosity!(η, ν, εII, args, rheology, cutoff)
    end

    ## Stress
    function JR2D.tensor_invariant!(::CUDABackendTrait, A::JustRelax.SymmetricTensor)
        _tensor_invariant!(A)
    end

    ## Buoyancy forces
    function JR2D.compute_ρg!(ρg::CuArray, rheology, args)
        return compute_ρg!(ρg, rheology, args)
    end
    function JR2D.compute_ρg!(
        ρg::CuArray, phase_ratios::JustRelax.PhaseRatio, rheology, args
    )
        return compute_ρg!(ρg, phase_ratios, rheology, args)
    end

    # Interpolations
    function JR2D.temperature2center!(
        ::CUDABackendTrait, thermal::JustRelax.ThermalArrays
    )
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
        return center2vertex!(
            vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy
        )
    end

    # Solvers
    function JR2D.solve!(::CUDABackendTrait, stokes, args...; kwargs)
        return _solve!(stokes, args...; kwargs...)
    end

    function JR2D.heatdiffusion_PT!(::CUDABackendTrait, thermal, args...; kwargs)
        return _heatdiffusion_PT!(thermal, args...; kwargs...)
    end

    # Utils
    function JR2D.compute_dt(::CUDABackendTrait, S::JustRelax.StokesArrays, args...)
        _compute_dt(S, args...)
    end

end

end
