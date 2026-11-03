push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    JustRelax.CPUBackend
end

const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax.JustRelax2D
end

@testset "Velocity boundary kernels 2D" begin
    nx, ny = 6, 5
    ni = nx, ny
    # Vx is (nx+1, ny+2), Vy is (nx+2, ny+1)
    Vx0 = [1.0 + 0.3 * sin(i + 0.7j) for i in 1:(nx + 1), j in 1:(ny + 2)]
    Vy0 = [-0.5 + 0.2 * cos(0.4i - j) for i in 1:(nx + 2), j in 1:(ny + 1)]
    inactive = (left = false, right = false, bot = false, top = false)

    function no_slip_reference(face)
        Vx, Vy = copy(Vx0), copy(Vy0)
        if face === :left
            Vx[1, :] .= 0
            Vy[1, :] .= -Vy[2, :]
        elseif face === :right
            Vx[end, :] .= 0
            Vy[end, :] .= -Vy[end - 1, :]
        elseif face === :bot
            Vx[:, 1] .= -Vx[:, 2]
            Vy[:, 1] .= 0
        else
            Vx[:, end] .= -Vx[:, end - 1]
            Vy[:, end] .= 0
        end
        return Vx, Vy
    end

    @testset "no-slip on the $face boundary" for face in (:left, :right, :bot, :top)
        stokes = StokesArrays(backend, ni)
        copyto!(stokes.V.Vx, Vx0)
        copyto!(stokes.V.Vy, Vy0)
        JR2K.no_slip!(stokes.V.Vx, stokes.V.Vy, merge(inactive, NamedTuple{(face,)}((true,))))
        Vx, Vy = Array(stokes.V.Vx), Array(stokes.V.Vy)
        @test (Vx, Vy) == no_slip_reference(face)
        # the wall velocity, the mean of the ghost and first interior value, vanishes
        if face === :left
            @test all(iszero, Vx[1, :])
            @test all(iszero, Vy[1, :] .+ Vy[2, :])
        elseif face === :top
            @test all(iszero, Vy[:, end])
            @test all(iszero, Vx[:, end] .+ Vx[:, end - 1])
        end
    end

    @testset "no-slip through flow_bcs!" begin
        stokes = StokesArrays(backend, ni)
        copyto!(stokes.V.Vx, Vx0)
        copyto!(stokes.V.Vy, Vy0)
        all_on = (left = true, right = true, bot = true, top = true)
        flow_bcs!(stokes, VelocityBoundaryConditions(; no_slip = all_on, free_slip = inactive))
        Vx, Vy = Array(stokes.V.Vx), Array(stokes.V.Vy)
        @test all(iszero, Vx[[1, end], :])
        @test all(iszero, Vy[:, [1, end]])
        @test Vx[2:(end - 1), 1] ≈ -Vx[2:(end - 1), 2]
        @test Vx[2:(end - 1), end] ≈ -Vx[2:(end - 1), end - 1]
        @test Vy[1, 2:(end - 1)] ≈ -Vy[2, 2:(end - 1)]
        @test Vy[end, 2:(end - 1)] ≈ -Vy[end - 1, 2:(end - 1)]
        @test Vx[2:(end - 1), 2:(end - 1)] == Vx0[2:(end - 1), 2:(end - 1)]
    end

    # non-uniform vertex coordinates
    xv = [0.0, 0.2, 0.5, 0.6, 1.0, 1.3, 1.8]
    yv = [-1.0, -0.9, -0.65, -0.55, -0.3, 0.0]
    grid = Geometry(Array, xv, yv)
    yc = 0.5 .* (yv[1:(end - 1)] .+ yv[2:end])

    @testset "pure shear background field" begin
        εbg = 0.35
        for call in (s -> pureshear_bc!(s, grid.xci, grid.xvi, εbg), s -> pureshear_bc!(s, grid.xci, grid.xvi, εbg, backend))
            stokes = StokesArrays(backend, ni)
            copyto!(stokes.V.Vx, Vx0)
            copyto!(stokes.V.Vy, Vy0)
            call(stokes)
            Vx, Vy = Array(stokes.V.Vx), Array(stokes.V.Vy)
            @test Vx[:, 2:(end - 1)] ≈ [εbg * x for x in xv, _ in 1:ny]
            @test Vy[2:(end - 1), :] ≈ [-εbg * y for _ in 1:nx, y in yv]
            # ghost layers are left to the flow boundary conditions
            @test Vx[:, [1, end]] == Vx0[:, [1, end]]
            @test Vy[[1, end], :] == Vy0[[1, end], :]
            # uniform, divergence-free strain rate on the non-uniform grid
            @test all((Vx[2:end, 2:(end - 1)] .- Vx[1:(end - 1), 2:(end - 1)]) ./ diff(xv) .≈ εbg)
            @test all((Vy[2:(end - 1), 2:end] .- Vy[2:(end - 1), 1:(end - 1)]) ./ diff(yv)' .≈ -εbg)
        end
    end

    @testset "simple shear background field" begin
        γbg = -0.8
        for call in (s -> simpleshear_bc!(s, grid.xci, grid.xvi, γbg), s -> simpleshear_bc!(s, grid.xci, grid.xvi, γbg, backend))
            stokes = StokesArrays(backend, ni)
            copyto!(stokes.V.Vx, Vx0)
            copyto!(stokes.V.Vy, Vy0)
            call(stokes)
            Vx, Vy = Array(stokes.V.Vx), Array(stokes.V.Vy)
            # Vx lives on the cell-center rows in y
            @test Vx[:, 2:(end - 1)] ≈ [γbg * y for _ in xv, y in yc]
            @test all(iszero, Vy[2:(end - 1), :])
            @test Vx[:, [1, end]] == Vx0[:, [1, end]]
            @test Vy[[1, end], :] == Vy0[[1, end], :]
            @test all((Vx[:, 3:(end - 1)] .- Vx[:, 2:(end - 2)]) ./ diff(yc)' .≈ γbg)
        end
    end
end
