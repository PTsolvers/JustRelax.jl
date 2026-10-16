push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    JustRelax.CPUBackend
end

const JR3K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax3D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax3D
else
    JustRelax.JustRelax3D
end

@testset "Velocity boundary kernels 3D" begin
    nx, ny, nz = 5, 4, 3
    ni = nx, ny, nz
    # Vx is (nx+1, ny+2, nz+2) and cyclic permutations
    V0 = (
        [1.0 + 0.3 * sin(i + 0.7j - k) for i in 1:(nx + 1), j in 1:(ny + 2), k in 1:(nz + 2)],
        [-0.5 + 0.2 * cos(0.4i - j + k) for i in 1:(nx + 2), j in 1:(ny + 1), k in 1:(nz + 2)],
        [0.25 + 0.1 * sin(i * j - k) for i in 1:(nx + 2), j in 1:(ny + 2), k in 1:(nz + 1)],
    )
    inactive = (left = false, right = false, front = false, back = false, bot = false, top = false)
    # face => (normal direction, side)
    faces = (left = (1, :lo), right = (1, :hi), front = (2, :lo), back = (2, :hi), bot = (3, :lo), top = (3, :hi))

    @testset "no-slip on the $face boundary" for face in keys(faces)
        d, side = faces[face]
        stokes = StokesArrays(backend, ni)
        foreach(copyto!, (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz), V0)
        JR3K.no_slip!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz, merge(inactive, NamedTuple{(face,)}((true,))))
        V = Array.((stokes.V.Vx, stokes.V.Vy, stokes.V.Vz))
        for n in 1:3
            A, A0 = V[n], V0[n]
            N = size(A, d)
            ghost, inner = side === :lo ? (1, 2) : (N, N - 1)
            if n == d
                # normal velocity sits on the wall
                @test all(iszero, selectdim(A, d, ghost))
            else
                # tangential velocity is mirrored so that it vanishes on the wall
                @test selectdim(A, d, ghost) == -selectdim(A0, d, inner)
            end
            rest = side === :lo ? (2:N) : (1:(N - 1))
            @test selectdim(A, d, rest) == selectdim(A0, d, rest)
        end
    end

    xv = [0.0, 0.2, 0.5, 0.6, 1.0, 1.3]
    yv = [-0.4, -0.1, 0.0, 0.35, 0.5]
    zv = [-1.0, -0.9, -0.5, 0.0]
    grid = Geometry(Array, xv, yv, zv)
    yc = 0.5 .* (yv[1:(end - 1)] .+ yv[2:end])

    @testset "pure shear background field" begin
        εbg = 0.35
        for call in (s -> pureshear_bc!(s, grid.xci, grid.xvi, εbg), s -> pureshear_bc!(s, grid.xci, grid.xvi, εbg, backend))
            stokes = StokesArrays(backend, ni)
            foreach(copyto!, (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz), V0)
            call(stokes)
            Vx, Vy, Vz = Array.((stokes.V.Vx, stokes.V.Vy, stokes.V.Vz))
            @test Vx[:, 2:(end - 1), 2:(end - 1)] ≈ [εbg * x for x in xv, _ in 1:ny, _ in 1:nz]
            @test Vy[2:(end - 1), :, 2:(end - 1)] ≈ [εbg * y for _ in 1:nx, y in yv, _ in 1:nz]
            @test Vz[2:(end - 1), 2:(end - 1), :] ≈ [-εbg * z for _ in 1:nx, _ in 1:ny, z in zv]
            @test Vx[:, 1, :] == V0[1][:, 1, :]
            @test Vy[:, :, end] == V0[2][:, :, end]
            @test Vz[end, :, :] == V0[3][end, :, :]
        end
    end

    @testset "simple shear background field" begin
        γbg = -0.8
        stokes = StokesArrays(backend, ni)
        foreach(copyto!, (stokes.V.Vx, stokes.V.Vy, stokes.V.Vz), V0)
        simpleshear_bc!(stokes, grid.xci, grid.xvi, γbg)
        Vx, Vy, Vz = Array.((stokes.V.Vx, stokes.V.Vy, stokes.V.Vz))
        @test Vx[:, 2:(end - 1), 2:(end - 1)] ≈ [γbg * y for _ in xv, y in yc, _ in 1:nz]
        @test all(iszero, Vy[2:(end - 1), :, 2:(end - 1)])
        @test all(iszero, Vz[2:(end - 1), 2:(end - 1), :])
        @test Vx[:, end, :] == V0[1][:, end, :]
        @test Vy[1, :, :] == V0[2][1, :, :]
        @test Vz[:, 1, :] == V0[3][:, 1, :]
        simpleshear_bc!(stokes, grid.xci, grid.xvi, 2γbg, backend)
        @test Array(stokes.V.Vx)[:, 2:(end - 1), 2:(end - 1)] ≈ [2γbg * y for _ in xv, y in yc, _ in 1:nz]
    end
end
