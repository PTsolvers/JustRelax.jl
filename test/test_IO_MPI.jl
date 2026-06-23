push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using ParallelStencil
import ImplicitGlobalGrid

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

function run_pvtk_MPI()
    nx, ny = 8, 6
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = true, select_device = false)...)
    grid = Geometry((nx, ny), (1.4, 0.6))
    di = grid.di.center
    me = igg.me

    # The parallel writer relies on the native ImplicitGlobalGrid parallel-VTK API
    # (`extents`, `extents_g`, `metagrid`), introduced in ImplicitGlobalGrid 0.17.
    # Skip cleanly on older releases so the suite stays green.
    if !isdefined(ImplicitGlobalGrid, :metagrid)
        me == 0 && @info "Skipping parallel VTK test: ImplicitGlobalGrid < 0.17"
        finalize_global_grid(; finalize_MPI = false)
        return nothing
    end

    data_c = (; P = fill(Float64(me + 1), nx, ny), eta = fill(10.0, nx, ny))   # center
    data_v = (; T = fill(Float64(me), nx + 1, ny + 1))                          # vertex
    vel = (fill(0.1, nx + 1, ny + 1), fill(-0.2, nx + 1, ny + 1))               # vertex velocity

    dst = joinpath(tempdir(), "JR_pvtk_test")
    if me == 0
        isdir(dst) && rm(dst; recursive = true)
        mkpath(dst)
    end
    JustRelax.MPI.Barrier(igg.comm_cart)

    # multiblock: vertex data + velocity, center data, with a pvd time series
    save_pvtk(
        joinpath(dst, "fields"), di, data_v, data_c, vel, igg;
        t = 2.5, precision = Float32, pvd = joinpath(dst, "series"),
    )
    # single group + velocity (Float64 precision)
    save_pvtk(joinpath(dst, "cvel"), di, data_c, (fill(0.1, nx, ny), fill(0.2, nx, ny)), igg; precision = Float64)
    # single group, no velocity, no time
    save_pvtk(joinpath(dst, "center_only"), di, data_c, igg)
    JustRelax.MPI.Barrier(igg.comm_cart)

    if me == 0
        @test isfile(joinpath(dst, "fields_center.pvti"))
        @test isfile(joinpath(dst, "fields_vertex.pvti"))
        @test count(endswith(".vti"), readdir(joinpath(dst, "fields_center"))) == igg.nprocs
        @test count(endswith(".vti"), readdir(joinpath(dst, "fields_vertex"))) == igg.nprocs
        vhdr = read(joinpath(dst, "fields_vertex.pvti"), String)
        @test occursin("WholeExtent", vhdr)
        @test occursin("Velocity", vhdr)                       # velocity written
        # pvd time-series collections (one per location)
        @test isfile(joinpath(dst, "series_vertex.pvd"))
        @test isfile(joinpath(dst, "series_center.pvd"))
        @test occursin("step", read(joinpath(dst, "series_vertex.pvd"), String)) ||
            occursin("fields", read(joinpath(dst, "series_vertex.pvd"), String))
        # velocity on the single-group method, written at Float64 precision
        chdr = read(joinpath(dst, "cvel.pvti"), String)
        @test occursin("Velocity", chdr)
        @test occursin("Float64", chdr)
        # t defaulted to nothing for center_only ⇒ no TimeValue array
        @test !occursin("TimeValue", read(joinpath(dst, "center_only.pvti"), String))
        rm(dst; recursive = true)
    end

    finalize_global_grid(; finalize_MPI = false)
    return nothing
end

@testset "IO MPI — parallel VTK (pvtk)" begin
    @suppress run_pvtk_MPI()
end
