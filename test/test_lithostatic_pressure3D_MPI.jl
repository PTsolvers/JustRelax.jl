push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test, Suppressor
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil
using ImplicitGlobalGrid
import ImplicitGlobalGrid: z_g, nz_g

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    CPUBackend
end

const nx, ny, nz = 3, 2, 8

# buoyancy and cell height of the global column, evaluated on global cell indices
ρg_global(k) = 1 + 0.25 * k
dz_global(k) = 0.1 * k

function start_grid(; kwargs...)
    init_mpi = !JustRelax.MPI.Initialized()
    return IGG(
        init_global_grid(
            nx, ny, nz; init_MPI = init_mpi, select_device = false, kwargs...
        )...,
    )
end

# global index of the local cell `k`
kg(ρg, k) = round(Int, z_g(k, 1.0, ρg)) + 1

# pressure at the global cell `k` of a column of `nzg` cells
function P_global(k, nzg, dz::Number)
    return sum(ρg_global(m) for m in (k + 1):nzg; init = 0.0) * dz + ρg_global(k) * dz / 2
end

function P_global(k, nzg, ::Nothing)
    return sum(ρg_global(m) * dz_global(m) for m in (k + 1):nzg; init = 0.0) +
        ρg_global(k) * dz_global(k) / 2
end

function global_column()
    ρg = @zeros(nx, ny, nz)
    ρg .= PTArray(backend_JR)([ρg_global(kg(ρg, k)) for _ in 1:nx, _ in 1:ny, k in 1:nz])
    return ρg
end

# ranks stacked along the vertical direction: every column spans several subdomains
function vertical_split()
    igg = start_grid(; dimx = 1, dimy = 1)
    nzg = nz_g()
    ρg = global_column()
    P = @zeros(nx, ny, nz)

    @testset "constant cell height" begin
        dz = 0.5
        compute_lithostatic_pressure!(P, ρg, dz, igg)
        P_cpu = Array(P)
        for k in 1:nz
            @test all(P_cpu[:, :, k] .≈ P_global(kg(ρg, k), nzg, dz))
        end
    end

    @testset "variable cell height" begin
        dz = PTArray(backend_JR)([dz_global(kg(ρg, k)) for k in 1:nz])
        compute_lithostatic_pressure!(P, ρg, dz, igg)
        P_cpu = Array(P)
        for k in 1:nz
            @test all(P_cpu[:, :, k] .≈ P_global(kg(ρg, k), nzg, nothing))
        end
    end

    @testset "neighboring ranks agree on the cells they share" begin
        # the halo cells are integrated from the same weights on both sides
        halo = @zeros(nx, ny, nz)
        halo .= P
        update_halo!(halo)
        @test Array(halo) ≈ Array(P)
    end

    if igg.dims[3] > 1
        @testset "guards" begin
            @test_throws "split across MPI ranks" compute_lithostatic_pressure!(P, ρg, 0.5)
            # a column that does not span the local subdomain cannot be placed in the global one
            @test_throws "must be a field of the global grid" compute_lithostatic_pressure!(
                @zeros(nx, ny, 2), @zeros(nx, ny, 2), 0.5, igg
            )
        end
    end

    finalize_global_grid(; finalize_MPI = false)
    return nothing
end

# ranks side by side: every rank holds a whole column, so both methods integrate it alone
function horizontal_split()
    igg = start_grid(; dimy = 1, dimz = 1)
    nzg = nz_g()
    ρg = global_column()
    dz = 0.5

    P_igg = @zeros(nx, ny, nz)
    compute_lithostatic_pressure!(P_igg, ρg, dz, igg)
    P_local = @zeros(nx, ny, nz)
    compute_lithostatic_pressure!(P_local, ρg, dz)

    @testset "columns are complete on every rank" begin
        @test nzg == nz
        @test Array(P_igg) == Array(P_local)
        P_cpu = Array(P_igg)
        for k in 1:nz
            @test all(P_cpu[:, :, k] .≈ P_global(kg(ρg, k), nzg, dz))
        end
    end

    finalize_global_grid(; finalize_MPI = false)
    return nothing
end

function periodic_column()
    igg = start_grid(; dimx = 1, dimy = 1, periodz = 1)
    ρg = @zeros(nx, ny, nz)
    P = @zeros(nx, ny, nz)

    if igg.dims[3] > 1
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
