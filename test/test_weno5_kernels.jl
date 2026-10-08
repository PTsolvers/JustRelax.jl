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

function to_device(A::AbstractArray{T}) where {T}
    B = @zeros(size(A)..., eltype = T)
    copyto!(B, A)
    return B
end

@testset "WENO5 advection kernels" begin
    @testset "one SSP-RK3 step translates a quadratic exactly ($(m == 1 ? "Jiang-Shu" : "Z") weights)" for m in (1, 2)
        nx, ny = 24, 22
        dx, dy = 0.1, 0.07
        x = [(i - 1) * dx for i in 1:nx]
        y = [(j - 1) * dy for j in 1:ny]
        # Every candidate stencil reproduces the derivative of a quadratic, so the spatial
        # operator is exact and the third-order Runge-Kutta step matches the exact translation.
        u_exact(x, y) = 0.4 + 1.3 * x - 0.7 * y + 2.1 * x^2 - 1.6 * y^2 + 0.9 * x * y
        for (vx, vy) in ((0.8, -0.5), (-0.6, 0.9))
            weno = WENO5(backend, Val(m), (nx, ny))
            u = to_device([u_exact(xi, yj) for xi in x, yj in y])
            V = (to_device(fill(vx, nx, ny)), to_device(fill(vy, nx, ny)))
            dt = 0.3 * min(dx, dy) / max(abs(vx), abs(vy))
            WENO_advection!(u, V, weno, (dx, dy), dt)
            # each of the three stages widens the region touched by the clamped boundary stencil by three nodes
            I, J = 10:(nx - 9), 10:(ny - 9)
            @test Array(u)[I, J] ≈ [u_exact(xi - vx * dt, yj - vy * dt) for xi in x[I], yj in y[J]] rtol = 1.0e-12
        end
    end

    @testset "translation of a smooth profile converges at high order" begin
        # Gaussian pulse moved by vx * t_end, far from the clamped boundaries
        profile(x) = exp(-((x - 0.3) / 0.08)^2)
        vx, t_end, ny = 1.0, 0.3, 5
        function max_error(nx, m)
            dx = 1.0 / (nx - 1)
            x = [(i - 1) * dx for i in 1:nx]
            weno = WENO5(backend, Val(m), (nx, ny))
            u = to_device(repeat(profile.(x), 1, ny))
            V = (to_device(fill(vx, nx, ny)), @zeros(nx, ny))
            nt = ceil(Int, t_end / (0.4 * dx))
            for _ in 1:nt
                WENO_advection!(u, V, weno, (dx, 1.0), t_end / nt)
            end
            return maximum(abs, Array(u)[:, 3] .- profile.(x .- vx * t_end))
        end
        for m in (1, 2)
            e1, e2 = max_error(81, m), max_error(161, m)
            @test e2 < e1
            @test log2(e1 / e2) > 3
        end
    end

    @testset "helpers" begin
        @test [JR2K.limit_periodic(a, 7) for a in -2:10] == clamp.(-2:10, 1, 7)
        weno = WENO5(backend, Val(1), (4, 4))
        @test_throws "Unknown method for the WENO Scheme" JR2K.weno_alphas_downwind(weno, Val(3), 1.0, 1.0, 1.0)
    end
end
