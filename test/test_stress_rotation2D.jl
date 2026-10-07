@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end
using Test
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D

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

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

# host copy of the values held by the active particles
active_values(p, particles) = Array(p.data)[Array(particles.index.data)]

@testset "Stress rotation on particles 2D" begin
    ni = (8, 6)
    grid = Geometry(ni, (1.0, 1.0))
    particles = init_particles(backend_JP, 9, 12, 3, grid.xi_vel...)
    stokes = StokesArrays(backend, ni)

    @testset "rigid rotation" begin
        pτ = StressParticles(particles)
        components = (JustRelax.normal_stress(pτ)..., JustRelax.shear_stress(pτ)...)
        τ0 = (1.0, -0.5, 0.2)
        Ω, dt = 0.3, 1.0
        foreach((p, v) -> fill!(p.data, v), components, τ0)
        fill!(JustRelax.shear_vorticity(pτ)[1].data, Ω)

        JustRelax2D.rotate_stress_particles!(components, JustRelax.shear_vorticity(pτ), particles, dt)

        # counterclockwise rotation by Ω dt, with Ω = ½(∂Vy/∂x - ∂Vx/∂y)
        c, s = cos(Ω * dt), sin(Ω * dt)
        R = [c -s; s c]
        τ_exact = R * [τ0[1] τ0[3]; τ0[3] τ0[2]] * R'
        for (p, expected) in zip(components, (τ_exact[1, 1], τ_exact[2, 2], τ_exact[1, 2]))
            @test all(isapprox.(active_values(p, particles), expected; atol = 1.0e-12))
        end
    end

    @testset "FLIP update keeps sub-grid stress" begin
        pτ = StressParticles(particles)
        components = (JustRelax.normal_stress(pτ)..., JustRelax.shear_stress(pτ)...)
        # particle stress that varies inside the cells
        for p in components
            p.data .= PTArray(backend)(rand(size(p.data)...))
        end

        stress2grid!(stokes, pτ, particles)
        @test Array(JustRelax.grid_stress(pτ)[1]) == Array(stokes.τ_o.xx)
        @test Array(JustRelax.grid_stress(pτ)[2]) == Array(stokes.τ_o.yy)
        @test Array(JustRelax.grid_stress(pτ)[3]) == Array(stokes.τ_o.xy)

        # the solve changes the grid stress by a uniform increment, without vorticity
        Δτ = (0.1, -0.2, 0.3)
        stokes.τ.xx .= stokes.τ_o.xx .+ Δτ[1]
        stokes.τ.yy .= stokes.τ_o.yy .+ Δτ[2]
        stokes.τ.xy .= stokes.τ_o.xy .+ Δτ[3]
        fill!(stokes.ω.xy, 0.0)
        before = map(p -> active_values(p, particles), components)

        rotate_stress!(pτ, stokes, particles, 1.0)

        for (p, τ_before, Δ) in zip(components, before, Δτ)
            @test active_values(p, particles) ≈ τ_before .+ Δ
        end
    end
end
