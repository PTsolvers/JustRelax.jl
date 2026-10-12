@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end
using Test
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil, ParallelStencil.FiniteDifferences3D

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

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

const JR3K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax3D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax3D
else
    JustRelax.JustRelax3D
end

# host copy of the values held by the active particles
active_values(p, particles) = Array(p.data)[Array(particles.index.data)]

@testset "Stress rotation on particles 3D" begin
    ni = (5, 4, 3)
    grid = Geometry(ni, (1.0, 1.0, 1.0))
    particles = init_particles(backend_JP, 9, 18, 3, grid.xi_vel...)
    stokes = StokesArrays(backend, ni)

    @testset "stress_fields" begin
        pτ = StressParticles(particles)
        grid_fields = stress_fields(stokes, pτ)
        @test length(grid_fields) == length(unwrap(pτ))
        @test grid_fields[1:3] == (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz)
        @test grid_fields[4] === stokes.τ.yz_c
        @test grid_fields[5] === stokes.τ.xz_c
        @test grid_fields[6] === stokes.τ.xy_c
        @test grid_fields[7:9] == (stokes.ω.yz_c, stokes.ω.xz_c, stokes.ω.xy_c)
    end

    @testset "rigid rotation" begin
        pτ = StressParticles(particles)
        components = (JustRelax.normal_stress(pτ)..., JustRelax.shear_stress(pτ)...)
        # Voigt order xx, yy, zz, yz, xz, xy
        τ0 = (1.0, -0.5, -0.5, 0.1, -0.3, 0.2)
        # angular velocity, i.e. the half components ½(∂vᵢ/∂xⱼ - ∂vⱼ/∂xᵢ) as (yz, xz, xy)
        Ω = (0.1, -0.2, 0.3)
        dt = 1.0
        foreach((p, v) -> fill!(p.data, v), components, τ0)
        foreach((p, v) -> fill!(p.data, v), JustRelax.shear_vorticity(pτ), Ω)

        JustRelax3D.rotate_stress_particles!(components, JustRelax.shear_vorticity(pτ), particles, dt)

        # rotation by |Ω| dt about Ω / |Ω| (Rodrigues)
        θ = sqrt(sum(abs2, Ω)) * dt
        n = collect(Ω) ./ sqrt(sum(abs2, Ω))
        K = [0 -n[3] n[2]; n[3] 0 -n[1]; -n[2] n[1] 0]
        R = cos(θ) * [1 0 0; 0 1 0; 0 0 1] + sin(θ) * K + (1 - cos(θ)) * (n * n')
        T = [τ0[1] τ0[6] τ0[5]; τ0[6] τ0[2] τ0[4]; τ0[5] τ0[4] τ0[3]]
        τ_exact = R * T * R'
        expected = (τ_exact[1, 1], τ_exact[2, 2], τ_exact[3, 3], τ_exact[2, 3], τ_exact[1, 3], τ_exact[1, 2])
        for (p, e) in zip(components, expected)
            @test all(isapprox.(active_values(p, particles), e; atol = 1.0e-12))
        end
    end

    @testset "FLIP update keeps sub-grid stress" begin
        pτ = StressParticles(particles)
        components = (JustRelax.normal_stress(pτ)..., JustRelax.shear_stress(pτ)...)
        for p in components
            p.data .= PTArray(backend)(rand(size(p.data)...))
        end

        stress2grid!(stokes, pτ, particles)
        τ_o = stokes.τ_o
        for (ref, τ) in zip(JustRelax.grid_stress(pτ), (τ_o.xx, τ_o.yy, τ_o.zz, τ_o.yz_c, τ_o.xz_c, τ_o.xy_c))
            @test Array(ref) == Array(τ)
        end

        Δτ = (0.1, -0.2, 0.3, -0.05, 0.15, 0.25)
        τ = stokes.τ
        for (dst, src, Δ) in zip(
                (τ.xx, τ.yy, τ.zz, τ.yz_c, τ.xz_c, τ.xy_c),
                (τ_o.xx, τ_o.yy, τ_o.zz, τ_o.yz_c, τ_o.xz_c, τ_o.xy_c),
                Δτ,
            )
            dst .= src .+ Δ
        end
        foreach(ω -> fill!(ω, 0.0), (stokes.ω.yz_c, stokes.ω.xz_c, stokes.ω.xy_c))
        before = map(p -> active_values(p, particles), components)

        rotate_stress!(pτ, stokes, particles, 1.0)

        for (p, τ_before, Δ) in zip(components, before, Δτ)
            @test active_values(p, particles) ≈ τ_before .+ Δ
        end
    end
end

@testset "Vorticity 3D" begin
    ni = 5, 4, 3
    grid = Geometry(ni, ni .* (0.3, 0.2, 0.25); origin = (0.0, 0.0, 0.0))
    xc = collect.(grid.xci)
    xv = collect.(grid.xvi)

    @testset "rigid rotation" begin
        Ω = (0.3, -0.5, 0.8)
        stokes = StokesArrays(backend, ni)
        # V = Ω × r on the staggered velocity coordinates
        V(x, y, z) = (Ω[2] * z - Ω[3] * y, Ω[3] * x - Ω[1] * z, Ω[1] * y - Ω[2] * x)
        for (d, A) in enumerate((stokes.V.Vx, stokes.V.Vy, stokes.V.Vz))
            copyto!(A, [V(x, y, z)[d] for x in grid.xi_vel[d][1], y in grid.xi_vel[d][2], z in grid.xi_vel[d][3]])
        end
        JR3K.compute_vorticity!(stokes, grid._di, ni, Val(3))
        for (d, (edge, center)) in enumerate(((stokes.ω.yz, stokes.ω.yz_c), (stokes.ω.xz, stokes.ω.xz_c), (stokes.ω.xy, stokes.ω.xy_c)))
            @test all(Array(edge) .≈ Ω[d])
            @test all(Array(center) .≈ Ω[d])
        end
    end

    @testset "evaluated on its own edges" begin
        # Vy = x z²: ω_xy = ∂Vy/∂x / 2 = z² / 2 on the xy edges (z at cell centers),
        # ω_yz = -∂Vy/∂z / 2 = -x z on the yz edges (x at cell centers, z at vertices);
        # centered differences are exact for both
        stokes = StokesArrays(backend, ni)
        copyto!(stokes.V.Vy, [x * z^2 for x in grid.xi_vel[2][1], _ in grid.xi_vel[2][2], z in grid.xi_vel[2][3]])
        ωxy_exact = [0.5 * z^2 for _ in xv[1], _ in xv[2], z in xc[3]]
        ωyz_exact = [-x * z for x in xc[1], _ in xv[2], z in xv[3]]

        JR3K.compute_vorticity!(stokes, grid._di, ni, Val(3))
        @test Array(stokes.ω.xy) ≈ ωxy_exact
        @test Array(stokes.ω.yz) ≈ ωyz_exact
        @test all(iszero, Array(stokes.ω.xz))
    end
end
