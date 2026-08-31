using Test, Suppressor

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPU # Options: CPUBackend, CUDABackend, AMDGPUBackend
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
        init_MPI = JustRelax.MPI.Initialized() ? false : true
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_MPI)...)
        li = lx, ly     # domain length in x- and y-
        di = @. li / ni # grid step in x- and -y
        origin = 0.0, -ly   # origin coordinates (15km f sticky air layer)
        grid = Geometry(ni, li; origin = origin)
        (; xci, xvi) = grid

        # 2D case
        dst = "test_IO"
        stokes = StokesArrays(backend_JR, ni)
        thermal = ThermalArrays(backend_JR, ni)

        nxcell, max_xcell, min_xcell = 20, 32, 12
        particles = init_particles(
            backend, nxcell, max_xcell, min_xcell, grid.xi_vel...
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

        # Call the function
        checkpointing_jld2(dst, stokes, thermal, time, dt)
        # Check that the file was created
        fname = joinpath(dst, "checkpoint.jld2")
        @test isfile(fname)

        checkpointing_jld2(dst, stokes, thermal, time, dt, igg)
        # Check that the file was created
        fname = joinpath(dst, "checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2")
        @test isfile(fname)

        # Load the data from the file
        stokes1, thermal1, t, dt1 = load_checkpoint_jld2(dst, igg)

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
            τII = Array(stokes.τ.II),
            εII = Array(stokes.ε.II),
            Vx = Array(Vx_v),
            Vy = Array(Vy_v),
        )
        data_c = (;
            T = Array(thermal.T[2:(end - 1), 2:(end - 1)]),
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
        @test isfile(joinpath(dst, "vtk_000001.vti"))
        @test isfile(joinpath(dst, "pvd_test.pvd"))

        # vertex and center fields share one file: point data and cell data
        vtk_str = String(read(joinpath(dst, "vtk_000001.vti")))
        for name_i in (keys(data_v)..., keys(data_c)...)
            @test occursin("Name=\"$name_i\"", vtk_str)
        end
        point_data = vtk_str[findfirst("<PointData", vtk_str)[1]:findfirst("</PointData", vtk_str)[1]]
        cell_data = vtk_str[findfirst("<CellData", vtk_str)[1]:findfirst("</CellData", vtk_str)[1]]
        @test occursin("Name=\"Velocity\"", point_data)
        # fields land where their size fits, not in the group they were passed in:
        # Vx is given on the vertices, τII on the cell centers
        @test occursin("Name=\"Vx\"", point_data)
        @test occursin("Name=\"τII\"", cell_data)
        @test occursin("Name=\"P\"", cell_data)
        @test_throws DimensionMismatch save_vtk(
            joinpath(dst, "vtk_bad"), xvi, xci, (; bad = zeros(nx + 2, ny + 2)),
            data_c, velocity_v
        )

        # VTK vectors carry three components; the out-of-plane one is zero in 2D
        @test occursin("Name=\"Velocity\" NumberOfComponents=\"3\"", point_data)
        Vpacked = DataIO.pack_velocity(velocity_v, Float32)
        @test size(Vpacked) == (3, size(Vx_v)...)
        @test Vpacked[1, :, :] == Float32.(Array(Vx_v))
        @test Vpacked[2, :, :] == Float32.(Array(Vy_v))
        @test all(iszero, Vpacked[3, :, :])
        @test_throws DimensionMismatch DataIO.pack_velocity(
            (Vx_v, view(Vy_v, :, 1:(size(Vy_v, 2) - 1))), Float32
        )

        # the vertex grid must bound the cell grid, and velocity must live on the nodes
        @test_throws DimensionMismatch save_vtk(
            joinpath(dst, "vtk_bad"), xci, xci, data_v, data_c, velocity_v
        )
        @test_throws DimensionMismatch save_vtk(
            joinpath(dst, "vtk_bad"), xci, data_c, velocity_v
        )

        velocity_c = (Array(stokes.V.Vx[1:nx, 1:ny]), Array(stokes.V.Vy[1:nx, 1:ny]))
        save_vtk(
            joinpath(dst, "vtk_center_" * lpad("1", 6, "0")),
            xci,
            data_c,
            velocity_c,
            t = time,
            pvd = joinpath(dst, "pvd_test1"),
        )

        @test isfile(joinpath(dst, "vtk_center_000001.vti"))
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

        # exercise the pvd collection branch of save_marker_chain
        save_marker_chain(
            joinpath(dst, "MarkerChainPVD"), chain.cell_vertices, chain.h_vertices;
            pvd = joinpath(dst, "markerchain_pvd"), t = 1.0,
        )
        @test isfile(joinpath(dst, "MarkerChainPVD.vtp"))
        @test isfile(joinpath(dst, "markerchain_pvd.pvd"))

        save_vtk(joinpath(dst, "vtk_default_t"), xci, data_c, velocity_c)
        @test isfile(joinpath(dst, "vtk_default_t.vti"))

        # save_particles (2D) with and without phases
        save_particles(particles, pPhases; fname = joinpath(dst, "particles2D_phases"))
        @test isfile(joinpath(dst, "particles2D_phases.vtu"))
        save_particles(particles; fname = joinpath(dst, "particles2D"))
        @test isfile(joinpath(dst, "particles2D.vtu"))

        # 3D case
        ni = nx, ny, nz
        stokes = StokesArrays(backend_JR, ni)

        thermal = ThermalArrays(backend_JR, 4, 4, 4)
        thermal = ThermalArrays(backend_JR, ni)

        nxcell, max_xcell, min_xcell = 20, 32, 12
        particles = init_particles(
            backend, nxcell, max_xcell, min_xcell, grid.xi_vel...
        )
        # temperature
        pT, pPhases = init_cell_arrays(particles, Val(2))
        time = 1.0
        dt = 0.1

        stokes.viscosity.η .= fill(1.0)
        stokes.V.Vy .= fill(10)
        thermal.T .= fill(100)


        # Call the function
        checkpointing_jld2(dst, stokes, thermal, time, dt)
        # Check that the file was created
        fname = joinpath(dst, "checkpoint.jld2")
        @test isfile(fname)

        checkpointing_jld2(dst, stokes, thermal, time, dt, igg)

        # Check that the file was created
        fname = joinpath(dst, "checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2")
        @test isfile(fname)

        # Load the data from the file
        stokes, thermal, time, dt = load_checkpoint_jld2(dst, igg)

        @test stokes.viscosity.η[1] == 1.0
        @test stokes.V.Vy[1] == 10
        @test thermal.T[1] == 100
        @test !isnothing(stokes.V.Vz)

        checkpointing_jld2(dst, stokes, time, dt)

        # Check that the file was created
        fname = joinpath(dst, "checkpoint.jld2")
        @test isfile(fname)

        checkpointing_jld2(dst, stokes, time, dt, igg)

        # Check that the file was created
        fname = joinpath(dst, "checkpoint" * lpad("$(igg.me)", 4, "0") * ".jld2")
        @test isfile(fname)
        # Load the data from the file
        stokes, _, time, dt = load_checkpoint_jld2(dst, igg)

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

        # 3D save_data exercises the `N == 3` Zc/Zv branch in IO/H5.jl
        grid3d = Geometry((4, 4, 4), (1.0, 1.0, 1.0); origin = (0.0, 0.0, -1.0))
        save_data(joinpath(dst, "save_data3D.hdf5"), grid3d)
        @test isfile(joinpath(dst, "save_data3D.hdf5"))

        # JLD2 kwargs path: exercise the AbstractArray, Tuple, scalar, and nothing branches
        checkpointing_jld2(
            dst, stokes, thermal, time, dt;
            extra_vec = [1.0, 2.0, 3.0],
            extra_tuple = ([1.0, 2.0], [3.0, 4.0]),
            extra_scalar = 42,
            extra_nothing = nothing,
        )
        restart_data = load(joinpath(dst, "checkpoint.jld2"))
        @test restart_data["extra_vec"] == [1.0, 2.0, 3.0]
        @test restart_data["extra_scalar"] == 42
        @test restart_data["extra_nothing"] === nothing

        # metadata fallback: file only present under <src>/test/
        srcdir = mktempdir()
        testsub = joinpath(srcdir, "test")
        mkpath(testsub)
        write(joinpath(testsub, "only_in_test.toml"), "name = \"x\"\n")
        write(joinpath(srcdir, "Project.toml"), "name = \"x\"\n")
        write(joinpath(srcdir, "Manifest.toml"), "manifest_format = \"2.0\"\n")
        metadata_dst = joinpath(dst, "meta_fallback")
        metadata(srcdir, metadata_dst, "only_in_test.toml")
        @test isfile(joinpath(metadata_dst, "only_in_test.toml"))
        @test isfile(joinpath(metadata_dst, "Project.toml"))

        # Remove the generated directory
        rm(dst, recursive = true)
    end

    @suppress @testset "save_particles3D phase and no-phase variants" begin
        dst3 = mktempdir()
        n = 8
        particles_mock = (
            coords = ((data = rand(n),), (data = rand(n),), (data = rand(n),)),
            index = (data = trues(n),),
        )
        JustRelax.DataIO.save_particles3D(
            particles_mock, Float32; fname = joinpath(dst3, "p3d"),
        )
        @test isfile(joinpath(dst3, "p3d.vtu"))

        pPhases_mock = (data = rand(Float32, n),)
        JustRelax.DataIO.save_particles3D(
            particles_mock, pPhases_mock, Float32; fname = joinpath(dst3, "p3d_phases"),
        )
        @test isfile(joinpath(dst3, "p3d_phases.vtu"))
    end
end
