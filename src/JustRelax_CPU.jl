"""
    JustRelax.JustRelax2D

Two-dimensional CPU implementation of the JustRelax solvers, initialized for
`ParallelStencil` in 2D. It provides the concrete Stokes and thermal-diffusion entry points
(such as [`solve!`](@ref), [`solve_DYREL!`](@ref), and [`solve_VariationalStokes!`](@ref))
together with the array types, boundary conditions, and helper kernels used to build 2D
models. The CUDA and AMDGPU backends extend the same functions through package extensions.
"""
module JustRelax2D

    using ..JustRelax
    using JustPIC, JustPIC._2D
    using StaticArrays
    using CellArrays
    using ParallelStencil, ParallelStencil.FiniteDifferences2D
    using ImplicitGlobalGrid
    using GeoParams, LinearAlgebra, Printf
    using Statistics
    using MPI

    import JustRelax: IGG, BackendTrait, CPUBackendTrait, backend, CPUBackend
    import JustRelax: PTStokesCoeffs
    import JustRelax:
        AbstractBoundaryConditions,
        TemperatureBoundaryConditions,
        AbstractFlowBoundaryConditions,
        DisplacementBoundaryConditions,
        VelocityBoundaryConditions,
        apply_dirichlet,
        apply_dirichlet!,
        isdirichlet

    import JustRelax: normal_stress, shear_stress, shear_vorticity
    import JustRelax: @dxi, @dx, @dy, @dz

    import JustPIC._2D: numphases, nphases, PhaseRatios, update_phase_ratios!, compute_dx, face_offset

    __init__() = @init_parallel_stencil(Threads, Float64, 2)

    include("common.jl")

    include("stokes/Stokes2D.jl")
    export solve!

    include("DYREL/solver.jl")
    export solve_DYREL!, DYREL

    include("variational_stokes/Stokes2D.jl")
    export solve_VariationalStokes!

end

"""
    JustRelax.JustRelax3D

Three-dimensional CPU implementation of the JustRelax solvers, initialized for
`ParallelStencil` in 3D. It mirrors [`JustRelax.JustRelax2D`](@ref) in three dimensions,
providing the concrete Stokes and thermal-diffusion entry points (such as [`solve!`](@ref)
and [`solve_VariationalStokes!`](@ref)) and the supporting types and kernels. The CUDA and
AMDGPU backends extend the same functions through package extensions.
"""
module JustRelax3D

    using ..JustRelax
    using JustPIC, JustPIC._3D
    using StaticArrays
    using CellArrays
    using ParallelStencil, ParallelStencil.FiniteDifferences3D
    using ImplicitGlobalGrid
    using GeoParams, LinearAlgebra, Printf
    using Statistics
    using MPI

    import JustRelax: IGG, BackendTrait, CPUBackendTrait, backend, CPUBackend
    import JustRelax: PTStokesCoeffs
    import JustRelax:
        AbstractBoundaryConditions,
        TemperatureBoundaryConditions,
        AbstractFlowBoundaryConditions,
        DisplacementBoundaryConditions,
        VelocityBoundaryConditions,
        apply_dirichlet,
        apply_dirichlet!,
        isdirichlet

    import JustRelax: normal_stress, shear_stress, shear_vorticity
    import JustRelax: @dxi, @dx, @dy, @dz

    import JustPIC._3D: numphases, nphases, PhaseRatios, update_phase_ratios!, compute_dx, face_offset

    __init__() = @init_parallel_stencil(Threads, Float64, 3)

    include("common.jl")
    include("stokes/Stokes3D.jl")
    export solve!

    include("variational_stokes/Stokes3D.jl")
    export solve_VariationalStokes!

end
