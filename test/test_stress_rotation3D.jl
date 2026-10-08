push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil
using LinearAlgebra

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

@parallel_indices (i, j, k) function _gather_particles!(out, A, index, np)
    for ip in 1:np
        out[ip, i, j, k] = @index(index[ip, i, j, k]) ? @index(A[ip, i, j, k]) : NaN
    end
    return nothing
end

function gather(A, particles)
    np = length(eltype(particles.index))
    nip = size(particles.index)
    out = @zeros(np, nip...)
    @parallel (@idx nip) _gather_particles!(out, A, particles.index, np)
    return Array(out)
end

voigt2tensor(τ) = [τ[1] τ[6] τ[5]; τ[6] τ[2] τ[4]; τ[5] τ[4] τ[3]]
tensor2voigt(A) = (A[1, 1], A[2, 2], A[3, 3], A[2, 3], A[1, 3], A[1, 2])

# rotation by the angle |Ω| t about Ω, i.e. R = exp(t [Ω]×) for a rigid rotation dr/dt = Ω × r
function rotation_matrix(Ω, t)
    θ = norm(Ω) * t
    n = collect(Ω) ./ norm(Ω)
    N = [0 -n[3] n[2]; n[3] 0 -n[1]; -n[2] n[1] 0]
    return cos(θ) * I + sin(θ) * N + (1 - cos(θ)) * n * n'
end

@testset "Stress rotation 3D" begin
    nx, ny, nz = 5, 4, 3
    ni = nx, ny, nz
    di = 0.3, 0.2, 0.25
    grid = Geometry(ni, ni .* di; origin = (0.0, 0.0, 0.0))
    xc = collect.(grid.xci)
    xv = collect.(grid.xvi)

    @testset "vorticity of a rigid rotation" begin
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

    @testset "vorticity is evaluated on its own edges" begin
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

        # the form with one spacing tuple per velocity component, launched by the Stokes solvers
        stokes.ω.xy .= 0
        stokes.ω.yz .= 0
        @parallel (@idx ni .+ 1) JR3K.compute_vorticity!(
            stokes.ω.yz, stokes.ω.xz, stokes.ω.xy, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz, grid._di.velocity...
        )
        @test Array(stokes.ω.xy) ≈ ωxy_exact
        @test Array(stokes.ω.yz) ≈ ωyz_exact
    end

    particles = init_particles(backend_JP, 4, 8, 2, grid.xi_vel...)
    active = gather(particles.index, particles) .== 1
    @test count(active) > 0
    τ0 = (1.2, -0.4, -0.8, 0.3, -0.25, 0.7)

    function rotated_particle_stress(Ω, dt)
        stokes = StokesArrays(backend, ni)
        for (A, v) in zip((stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c), τ0)
            A .= v
        end
        for (A, v) in zip((stokes.ω.yz_c, stokes.ω.xz_c, stokes.ω.xy_c), Ω)
            A .= v
        end
        pτ = StressParticles(particles)
        rotate_stress!(pτ, stokes, particles, dt)
        return pτ, stokes
    end

    @testset "zero vorticity leaves the stress unchanged" begin
        pτ, _ = rotated_particle_stress((0.0, 0.0, 0.0), 0.3)
        for (A, e) in zip((pτ.τ_normal..., pτ.τ_shear...), τ0)
            @test all(gather(A, particles)[active] .≈ e)
        end
    end

    @testset "uniform vorticity rotates the stress by |ω| Δt" begin
        Ω, dt = (0.3, -0.5, 0.8), 0.4
        pτ, _ = rotated_particle_stress(Ω, dt)
        R = rotation_matrix(Ω, dt)
        τ_expected = tensor2voigt(R * voigt2tensor(τ0) * R')
        # invariants of the reference rotation
        @test tr(voigt2tensor(τ_expected)) ≈ tr(voigt2tensor(τ0)) atol = 1.0e-12
        @test det(voigt2tensor(τ_expected)) ≈ det(voigt2tensor(τ0))
        got = map(A -> gather(A, particles)[active], (pτ.τ_normal..., pτ.τ_shear...))
        for n in 1:6
            @test all(got[n] .≈ τ_expected[n])
        end
        # the stress is still rotated rigidly, preserving its invariants
        τ_got = voigt2tensor(ntuple(n -> got[n][1], 6))
        @test tr(τ_got) ≈ tr(voigt2tensor(τ0)) atol = 1.0e-12
        @test det(τ_got) ≈ det(voigt2tensor(τ0))
    end

    @testset "stress2grid! maps a uniform particle stress onto centers and edges" begin
        pτ = StressParticles(particles)
        for (A, v) in zip((pτ.τ_normal..., pτ.τ_shear...), τ0)
            fill!(A, eltype(A)(ntuple(_ -> v, length(eltype(A)))))
        end
        stokes = StokesArrays(backend, ni)
        stress2grid!(stokes, pτ, particles)
        τo = stokes.τ_o
        for (A, v) in zip((τo.xx, τo.yy, τo.zz, τo.yz_c, τo.xz_c, τo.xy_c), τ0)
            @test all(Array(A) .≈ v)
        end
        # edges between four cell centers
        @test all(Array(τo.yz)[:, 2:(end - 1), 2:(end - 1)] .≈ τ0[4])
        @test all(Array(τo.xz)[2:(end - 1), :, 2:(end - 1)] .≈ τ0[5])
        @test all(Array(τo.xy)[2:(end - 1), 2:(end - 1), :] .≈ τ0[6])
    end
end
