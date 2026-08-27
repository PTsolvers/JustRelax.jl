push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
using ImplicitGlobalGrid
import ImplicitGlobalGrid: y_g, ny_g

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

const nx, ny = 4, 8

# buoyancy and cell height of the global column, evaluated on global cell indices
ρg_global(j) = 1 + 0.25 * j
dz_global(j) = 0.1 * j

function start_grid(; kwargs...)
    init_mpi = !JustRelax.MPI.Initialized()
    return IGG(
        init_global_grid(
            nx, ny, 1; init_MPI = init_mpi, select_device = false, kwargs...
        )...,
    )
end

# global index of the local cell `j`
jg(ρg, j) = round(Int, y_g(j, 1.0, ρg)) + 1

# pressure at the global cell `j` of a column of `nyg` cells
function P_global(j, nyg, dz::Number)
    return sum(ρg_global(k) for k in (j + 1):nyg; init = 0.0) * dz + ρg_global(j) * dz / 2
end

function P_global(j, nyg, ::Nothing)
    return sum(ρg_global(k) * dz_global(k) for k in (j + 1):nyg; init = 0.0) +
        ρg_global(j) * dz_global(j) / 2
end

function global_column()
    ρg = @zeros(nx, ny)
    ρg .= PTArray(backend_JR)([ρg_global(jg(ρg, j)) for _ in 1:nx, j in 1:ny])
    return ρg
end

# ranks stacked along the vertical direction: every column spans several subdomains
function vertical_split()
    igg = start_grid(; dimx = 1, dimz = 1)
    nyg = ny_g()
    ρg = global_column()
    P = @zeros(nx, ny)

    @testset "constant cell height" begin
        dz = 0.5
        compute_lithostatic_pressure!(P, ρg, dz, igg)
        P_cpu = Array(P)
        for j in 1:ny
            @test all(P_cpu[:, j] .≈ P_global(jg(ρg, j), nyg, dz))
        end
    end

    @testset "variable cell height" begin
        dz = PTArray(backend_JR)([dz_global(jg(ρg, j)) for j in 1:ny])
        compute_lithostatic_pressure!(P, ρg, dz, igg)
        P_cpu = Array(P)
        for j in 1:ny
            @test all(P_cpu[:, j] .≈ P_global(jg(ρg, j), nyg, nothing))
        end
    end

    @testset "neighboring ranks agree on the cells they share" begin
        # the halo cells are integrated from the same weights on both sides
        halo = @zeros(nx, ny)
        halo .= P
        update_halo!(halo)
        @test Array(halo) ≈ Array(P)
    end

    if igg.dims[2] > 1
        @testset "guards" begin
            @test_throws "split across MPI ranks" compute_lithostatic_pressure!(P, ρg, 0.5)
            # a column that does not span the local subdomain cannot be placed in the global one
            @test_throws "must be a field of the global grid" compute_lithostatic_pressure!(
                @zeros(nx, 2), @zeros(nx, 2), 0.5, igg
            )
        end
    end

    finalize_global_grid(; finalize_MPI = false)
    return nothing
end

# ranks side by side: every rank holds a whole column, so both methods integrate it alone
function horizontal_split()
    igg = start_grid(; dimy = 1, dimz = 1)
    nyg = ny_g()
    ρg = global_column()
    dz = 0.5

    P_igg = @zeros(nx, ny)
    compute_lithostatic_pressure!(P_igg, ρg, dz, igg)
    P_local = @zeros(nx, ny)
    compute_lithostatic_pressure!(P_local, ρg, dz)

    @testset "columns are complete on every rank" begin
        @test nyg == ny
        @test Array(P_igg) == Array(P_local)
        P_cpu = Array(P_igg)
        for j in 1:ny
            @test all(P_cpu[:, j] .≈ P_global(jg(ρg, j), nyg, dz))
        end
    end

    finalize_global_grid(; finalize_MPI = false)
    return nothing
end

function periodic_column()
    igg = start_grid(; dimx = 1, dimz = 1, periody = 1)
    ρg = @zeros(nx, ny)
    P = @zeros(nx, ny)

    if igg.dims[2] > 1
        @testset "vertically periodic columns are rejected" begin
            @test_throws "periodic along the vertical" compute_lithostatic_pressure!(
                P, ρg, 0.5, igg
            )
        end
    end

    finalize_global_grid(; finalize_MPI = false)
    return nothing
end

@testset "compute_lithostatic_pressure! MPI" begin
    @suppress vertical_split()
    @suppress horizontal_split()
    @suppress periodic_column()
end
