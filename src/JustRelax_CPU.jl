module JustRelax2D

    @doc """
        JustRelax.JustRelax2D

    Two-dimensional solvers, kernels, and constructors, on the CPU backend.

    The submodule is loaded with `using JustRelax.JustRelax2D`, and its ParallelStencil
    environment is initialized for two dimensions when it loads. Loading CUDA.jl or
    AMDGPU.jl before it adds the matching GPU methods through a package extension; the
    entry points and their signatures stay the same, and the backend is chosen by the
    array type of the containers passed in. See [`JustRelax.JustRelax3D`](@ref) for the
    three-dimensional counterpart.
    """ JustRelax2D

    using ..JustRelax
    using JustPIC
    using CellArraysIndexing: @index
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

    import JustPIC: numphases, nphases, PhaseRatios, update_phase_ratios!, cell_index

    # `@index` is CellArraysIndexing's cell accessor. JustPIC does not export it -- a bare
    # `@index` there is KernelAbstractions' kernel index macro -- so it is re-exported here.
    export @index

    __init__() = @init_parallel_stencil(Threads, Float64, 2)

    include("common.jl")

    include("stokes/Stokes2D.jl")
    export solve!

    include("DYREL/solver.jl")
    include("DYREL/solver_VS.jl")
    export solve_DYREL!, solve_VariationalDYREL!, DYREL

    include("variational_stokes/Stokes2D.jl")
    export solve_VariationalStokes!

end

module JustRelax3D

    @doc """
        JustRelax.JustRelax3D

    Three-dimensional solvers, kernels, and constructors, on the CPU backend.

    The submodule is loaded with `using JustRelax.JustRelax3D`, and its ParallelStencil
    environment is initialized for three dimensions when it loads. Loading CUDA.jl or
    AMDGPU.jl before it adds the matching GPU methods through a package extension; the
    entry points and their signatures stay the same, and the backend is chosen by the
    array type of the containers passed in. See [`JustRelax.JustRelax2D`](@ref) for the
    two-dimensional counterpart.
    """ JustRelax3D

    using ..JustRelax
    using JustPIC
    using CellArraysIndexing: @index
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

    import JustPIC: numphases, nphases, PhaseRatios, update_phase_ratios!, cell_index

    # `@index` is CellArraysIndexing's cell accessor. JustPIC does not export it -- a bare
    # `@index` there is KernelAbstractions' kernel index macro -- so it is re-exported here.
    export @index

    __init__() = @init_parallel_stencil(Threads, Float64, 3)

    include("common.jl")
    include("stokes/Stokes3D.jl")
    export solve!

    include("variational_stokes/Stokes3D.jl")
    export solve_VariationalStokes!

end
