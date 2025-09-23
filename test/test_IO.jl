using Test, Suppressor

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
# const backend = CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using GeoParams
using WriteVTK, JLD2

@testset "Test IO" begin
    @suppress begin
        # Set up mock data
        # Physical domain ------------------------------------
        ly = 1.0       # domain length in y
        lx = 1.0       # domain length in x
        nx, ny, nz = 4, 4, 4   # number of cells
        ni = nx, ny     # number of cells
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
        li = lx, ly     # domain length in x- and y-
        di = @. li / ni # grid step in x- and -y
        origin = 0.0, -ly   # origin coordinates (15km f sticky air layer)
        grid = Geometry(ni, li; origin = origin)
        (; xci, xvi) = grid

        # 2D case
        dst = "test_IO"
        stokes = StokesArrays(backend_JR, ni)

        thermal = ThermalArrays(backend_JR, 4, 4)
        @test size(thermal.Tc) === (4, 4)

        thermal = ThermalArrays(backend_JR, ni)
        @test size(thermal.Tc) === (4, 4)

        nxcell, max_xcell, min_xcell = 20, 32, 12
        particles = init_particles(
            backend, nxcell, max_xcell, min_xcell, xvi...
        )
        # temperature
        pT, pPhases = init_cell_arrays(particles, Val(2))
        time = 1.0
        dt = 0.1

        stokes.viscosity.η .= @fill(1.0)
        stokes.V.Vy .= @fill(10)
        thermal.T .= @fill(100)

        # Save metadata to directory
        metadata(pwd(), dst, "test_traits.jl", "test_types.jl")
        @test isfile(joinpath(dst, "test_traits.jl"))
        @test isfile(joinpath(dst, "test_types.jl"))
        @test isfile(joinpath(dst, "Project.toml"))

        # Call the function
        checkpointing_jld2(dst, stokes, thermal, time, dt, igg)
        checkpointing_jld2(dst, stokes, thermal, time, dt)

        # Check that the file was created
        fname = joinpath(dst, "checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2")
        @test isfile(fname)

        # Load the data from the file
        stokes1, thermal1, t, dt1 = load_checkpoint_jld2(fname)

        @test stokes1.viscosity.η[1] == 1.0
        @test stokes1.V.Vy[1] == 10
        @test thermal1.T[1] == 100
        @test isnothing(stokes.V.Vz)
        @test dt1 == 0.1


        # check the if the hdf5 function also works
        checkpointing_hdf5(dst, stokes, thermal.T, time, dt)

        # Check that the file was created
        fname = joinpath(dst, "checkpoint.h5")
        @test isfile(fname)

        # Load the data from the file
        P, T, Vx, Vy, Vz, η, t, dt = load_checkpoint_hdf5(fname)

        stokes.viscosity.η .= η
        stokes.V.Vy .= Vy
        thermal.T .= T
        @test stokes.viscosity.η[1] == 1.0
        @test stokes.V.Vy[1] == 10
        @test thermal.T[1] == 100
        @test isnothing(Vz)
        @test dt == 0.1

        # test VTK save
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        data_v = (;
            T = Array(thermal.T),
            τII = Array(stokes.τ.II),
            εII = Array(stokes.ε.II),
            Vx = Array(Vx_v),
            Vy = Array(Vy_v),
        )
        data_c = (;
            P = Array(stokes.P),
            η = Array(stokes.viscosity.η),
        )
        velocity_v = (
            Array(Vx_v),
            Array(Vy_v),
        )
        save_vtk(
            joinpath(dst, "vtk_" * lpad("1", 6, "0")),
            xvi,
            xci,
            data_v,
            data_c,
            velocity_v,
            t = time,
            pvd = joinpath(dst, "pvd_test"),
        )
        @test isfile(joinpath(dst, "vtk_000001_1.vti"))
        @test isfile(joinpath(dst, "vtk_000001_2.vti"))
        @test isfile(joinpath(dst, "vtk_000001.vtm"))
        @test isfile(joinpath(dst, "pvd_test.pvd"))


        save_vtk(
            joinpath(dst, "vtk_" * lpad("1", 6, "0")),
            xci,
            data_c,
            velocity_v,
            t = time,
            pvd = joinpath(dst, "pvd_test1"),
        )

        @test isfile(joinpath(dst, "vtk_000001.vti"))
        @test isfile(joinpath(dst, "pvd_test1.pvd"))

        save_vtk(
            joinpath(dst, "vtk_" * lpad("2", 6, "0")),
            xci,
            (P = stokes.P, η = stokes.viscosity.η);
            t = time,
            pvd = joinpath(dst, "pvd_test2"),
        )
        @test isfile(joinpath(dst, "vtk_000002.vti"))
        @test isfile(joinpath(dst, "pvd_test2.pvd"))

        # VTK data series
        vtk = VTKDataSeries(joinpath(dst, "vtk_series"), xci)
        @test vtk isa VTKDataSeries

        DataIO.append!(vtk, (Vy = stokes.V.Vy, η = stokes.viscosity.η), dt, time)
        @test isfile(joinpath(dst, "vtk_series.pvd"))

        ## Test save_marker_chain
        nxcell, max_xcell, min_xcell = 100, 150, 75
        initial_elevation = 0.0e0
        chain = init_markerchain(backend, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)

        save_marker_chain(joinpath(dst, "MarkerChain"), chain.cell_vertices, chain.h_vertices)
        @test isfile(joinpath(dst, "MarkerChain.vtp"))

        # 3D case
        ni = nx, ny, nz
        stokes = StokesArrays(backend_JR, ni)

        thermal = ThermalArrays(backend_JR, 4, 4, 4)
        @test size(thermal.Tc) === (4, 4, 4)
        thermal = ThermalArrays(backend_JR, ni)
        @test size(thermal.Tc) === (4, 4, 4)

        nxcell, max_xcell, min_xcell = 20, 32, 12
        particles = init_particles(
            backend, nxcell, max_xcell, min_xcell, xvi...
        )
        # temperature
        pT, pPhases = init_cell_arrays(particles, Val(2))
        time = 1.0
        dt = 0.1

        stokes.viscosity.η .= fill(1.0)
        stokes.V.Vy .= fill(10)
        thermal.T .= fill(100)


        # Call the function
        checkpointing_jld2(dst, stokes, thermal, time, dt, igg)
        checkpointing_jld2(dst, stokes, thermal, time, dt)

        # Check that the file was created
        fname = joinpath(dst, "checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2")
        @test isfile(fname)

        # Load the data from the file
        stokes, thermal, time, dt = load_checkpoint_jld2(fname)

        @test stokes.viscosity.η[1] == 1.0
        @test stokes.V.Vy[1] == 10
        @test thermal.T[1] == 100
        @test !isnothing(stokes.V.Vz)

        checkpointing_jld2(dst, stokes, time, dt)
        checkpointing_jld2(dst, stokes, time, dt, igg)

        # Check that the file was created
        fname = joinpath(dst, "checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2")
        @test isfile(fname)
        # Load the data from the file
        stokes, _, time, dt = load_checkpoint_jld2(fname)

        @test stokes.viscosity.η[1] == 1.0
        @test stokes.V.Vy[1] == 10
        @test !isnothing(stokes.V.Vz)

        restart_data = load(fname)
        @test !haskey(restart_data, "thermal")


        # check the if the hdf5 function also works
        checkpointing_hdf5(dst, stokes, thermal.T, time, dt)

        # Check that the file was created
        fname = joinpath(dst, "checkpoint.h5")
        @test isfile(fname)

        # Load the data from the file
        P, T, Vx, Vy, Vz, η, t, dt = load_checkpoint_hdf5(fname)

        stokes.viscosity.η .= η
        stokes.V.Vy .= Vy
        thermal.T .= T
        @test stokes.viscosity.η[1] == 1.0
        @test stokes.V.Vy[1] == 10
        @test thermal.T[1] == 100
        @test !isnothing(Vz)

        # Test center and vertex coordinates function
        xci_c = center_coordinates(grid)
        @test (xci_c[1][1], xci_c[1][end]) === (0.125, 0.875)
        @test (xci_c[2][1], xci_c[2][end]) === (-0.875, -0.125)
        xvi_v = vertex_coordinates(grid)
        @test (xvi_v[1][1], xvi_v[1][end]) === (0.0, 1.0)
        @test (xvi_v[2][1], xvi_v[2][end]) === (-1.0, 0.0)

        # test save_data function
        save_data(joinpath(dst, "save_data.hdf5"), grid)
        @test isfile(joinpath(dst, "save_data.hdf5"))

        # Remove the generated directory
        rm(dst, recursive = true)
    end
end
