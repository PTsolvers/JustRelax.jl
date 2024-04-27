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

    import JustRelax:
        IGG,
        BackendTrait,
        CPUBackendTrait,
        backend,
        CPUBackend,
        Geometry,
        @cell

    @init_parallel_stencil(CUDA, Float64, 2)

    include("../src/common.jl")
    include("../src/stokes/Stokes2D.jl")

    # add CUDA traits
    struct CUDABackendTrait <: BackendTrait end

    @inline backend(::CuArray) = CUDABackendTrait()
    @inline backend(::Type{<:CuArray}) = CUDABackendTrait()

    # Types
    function JustRelax.JustRelax2D.StokesArrays(
        ::Type{CUDABackend}, ni::NTuple{N,Integer}
    ) where {N}
        return StokesArrays(ni)
    end

    function JustRelax.JustRelax2D.ThermalArrays(
        ::Type{CUDABackend}, ni::NTuple{N,Number}
    ) where {N}
        return ThermalArrays(ni...)
    end

    function JustRelax.JustRelax2D.ThermalArrays(
        ::Type{CUDABackend}, ni::Vararg{Number,N}
    ) where {N}
        return ThermalArrays(ni...)
    end

    function JustRelax.JustRelax2D.PhaseRatio(::Type{CUDABackend}, ni, num_phases)
        return PhaseRatio(ni, num_phases)
    end

    function PTThermalCoeffs(
        ::Type{CUDABackend},
        rheology,
        phase_ratios,
        args,
        dt,
        ni,
        di::NTuple{nDim,T},
        li::NTuple{nDim,Any};
        ϵ = 1e-8,
        CFL = 0.9 / √3,
    ) where {nDim,T}
        return JustRelax.JustRelax2D.PTThermalCoeffs(
            rheology,
            phase_ratios,
            args,
            dt,
            ni,
            di,
            li;
            ϵ = ϵ,
            CFL = CFL,
        )
    end
    
    
    function PTThermalCoeffs(
        ::Type{CUDABackend},
        rheology,
        args,
        dt,
        ni,
        di::NTuple{nDim,T},
        li::NTuple{nDim,Any};
        ϵ = 1e-8,
        CFL = 0.9 / √3,
    ) where {nDim,T}
        return JustRelax.JustRelax2D.PTThermalCoeffs(
            rheology,
            args,
            dt,
            ni,
            di,
            li;
            ϵ = ϵ,
            CFL = CFL,
        )
    end

    # Boundary conditions
    function JustRelax.JustRelax2D.flow_bcs!(
        ::CUDABackendTrait, stokes::StokesArrays, bcs
    )
        return _flow_bcs!(bcs, @velocity(stokes))
    end

    function flow_bcs!(
        ::CUDABackendTrait, stokes::StokesArrays, bcs
    )
        return _flow_bcs!(bcs, @velocity(stokes))
    end

    function JustRelax.JustRelax2D.thermal_bcs!(
        ::CUDABackendTrait, thermal::ThermalArrays, bcs
    )
        return thermal_bcs!(thermal.T, bcs)
    end
    
    function thermal_bcs!(
        ::CUDABackendTrait, thermal::ThermalArrays, bcs
    )
        return thermal_bcs!(thermal.T, bcs)
    end

    # Phases
    function JustRelax.JustRelax2D.phase_ratios_center(
        ::CUDABackendTrait, phase_ratios::PhaseRatio, particles, grid::Geometry, phases
    )
        return _phase_ratios_center(phase_ratios, particles, grid, phases)
    end

    # Rheology
    ## viscosity
    function JustRelax.JustRelax2D.compute_viscosity!(
        ::CUDABackendTrait, stokes, ν, args, rheology, cutoff
    )
        return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
    end
    function JustRelax.JustRelax2D.compute_viscosity!(
        ::CUDABackendTrait, stokes, ν, phase_ratios, args, rheology, cutoff
    )
        return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, cutoff)
    end
    function JustRelax.JustRelax2D.compute_viscosity!(
        η, ν, εII::CuArray, args, rheology, cutoff
    )
        return compute_viscosity!(η, ν, εII, args, rheology, cutoff)
    end

    function compute_viscosity!(
        ::CUDABackendTrait, stokes, ν, args, rheology, cutoff
    )
        return _compute_viscosity!(stokes, ν, args, rheology, cutoff)
    end
    function compute_viscosity!(
        ::CUDABackendTrait, stokes, ν, phase_ratios, args, rheology, cutoff
    )
        return _compute_viscosity!(stokes, ν, phase_ratios, args, rheology, cutoff)
    end
    function compute_viscosity!(
        η, ν, εII::CuArray, args, rheology, cutoff
    )
        return compute_viscosity!(η, ν, εII, args, rheology, cutoff)
    end

    ## Stress
    JustRelax.JustRelax2D.tensor_invariant!(A::SymmetricTensor) = tensor_invariant!(A)

    ## Buoyancy forces
    function JustRelax.JustRelax2D.compute_ρg!(ρg::CuArray, rheology, args)
        return compute_ρg!(ρg, rheology, args)
    end
    function JustRelax.JustRelax2D.compute_ρg!(
        ρg::CuArray, phase_ratios::PhaseRatio, rheology, args
    )
        return compute_ρg!(ρg, phase_ratios, rheology, args)
    end

    # Interpolations
    function JustRelax.JustRelax2D.temperature2center!(
        ::CUDABackendTrait, thermal::ThermalArrays
    )
        return _temperature2center!(thermal)
    end
    function JustRelax.JustRelax2D.vertex2center!(center::T, vertex::T) where {T<:CuArray}
        return vertex2center!(center, vertex)
    end
    function JustRelax.JustRelax2D.center2vertex!(vertex::T, center::T) where {T<:CuArray}
        return center2vertex!(vertex, center)
    end

    function JustRelax.JustRelax2D.center2vertex!(
        vertex_yz::T, vertex_xz::T, vertex_xy::T, center_yz::T, center_xz::T, center_xy::T
    ) where {T<:CuArray}
        return center2vertex!(
            vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy
        )
    end

    # Solvers
    JustRelax.JustRelax2D.solve!(::CUDABackendTrait, stokes, args...; kwargs) = _solve!(stokes, args...; kwargs...) 
    
    # Utils
    JustRelax.JustRelax2D.compute_dt(S::StokesArrays, di, dt_diff, I) = compute_dt(S, di, dt_diff, I::IGG)
    JustRelax.JustRelax2D.compute_dt(S::StokesArrays, di, dt_diff) = compute_dt(S, di, dt_diff)
    JustRelax.JustRelax2D.compute_dt(S::StokesArrays, di) = compute_dt(S, di)
    
end

end
