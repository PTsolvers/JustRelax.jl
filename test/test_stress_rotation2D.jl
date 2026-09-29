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

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax.JustRelax2D
end

# particle value as a function of its slot and cell, so every particle carries a distinct stress
pval(n, ip, i, j) = sin(0.7 * n + 0.3 * ip + 0.5 * i - 0.2 * j) + 0.1 * n

@parallel_indices (i, j) function _fill_particles!(A, n, np)
    for ip in 1:np
        @index A[ip, i, j] = pval(n, ip, i, j)
    end
    return nothing
end

@parallel_indices (i, j) function _gather_particles!(out, A, index, np)
    for ip in 1:np
        out[ip, i, j] = @index(index[ip, i, j]) ? @index(A[ip, i, j]) : NaN
    end
    return nothing
end

function gather(A, particles)
    np = length(eltype(particles.index))
    ni = size(particles.index)
    out = @zeros(np, ni...)
    @parallel (@idx ni) _gather_particles!(out, A, particles.index, np)
    return Array(out)
end

# τ' = R τ Rᵀ, R the counterclockwise rotation by θ
function rotate(τxx, τyy, τxy, θ)
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    τ = R * [τxx τxy; τxy τyy] * R'
    return τ[1, 1], τ[2, 2], τ[1, 2]
end

@testset "Stress rotation 2D" begin
    nx, ny = 6, 5
    ni = nx, ny
    li = 1.2, 0.75
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    particles = init_particles(backend_JP, 4, 8, 2, grid.xi_vel...)
    np = length(eltype(particles.index))
    active = gather(particles.index, particles) .== 1
    @test count(active) > 0

    @testset "rigid rotation has uniform vorticity" begin
        Ω = 0.8
        stokes = StokesArrays(backend, ni)
        # V = Ω × r, with Vx = -Ω y and Vy = Ω x on their staggered coordinates
        xvx, yvx = grid.xi_vel[1]
        xvy, yvy = grid.xi_vel[2]
        copyto!(stokes.V.Vx, [-Ω * y for _ in xvx, y in yvx])
        copyto!(stokes.V.Vy, [Ω * x for x in xvy, _ in yvy])
        JR2K.compute_vorticity!(stokes, grid._di, ni, Val(2))
        # ω_xy = (∂Vy/∂x - ∂Vx/∂y) / 2 is the angular velocity
        @test all(Array(stokes.ω.xy) .≈ Ω)
        @test all(Array(stokes.ω.xy_c) .≈ Ω)

        # rotate_stress! then turns a uniform stress by the angle Ω Δt
        dt = 0.3
        τ0 = (1.2, -0.4, 0.7)
        stokes.τ.xx .= τ0[1]
        stokes.τ.yy .= τ0[2]
        stokes.τ.xy .= τ0[3]
        pτ = StressParticles(particles)
        rotate_stress!(pτ, stokes, particles, dt)
        τ_expected = rotate(τ0..., Ω * dt)
        for (A, e) in zip((pτ.τ_normal..., pτ.τ_shear...), τ_expected)
            @test all(gather(A, particles)[active] .≈ e)
        end
        # the rotation leaves the invariants unchanged
        @test τ_expected[1] + τ_expected[2] ≈ τ0[1] + τ0[2]
        @test τ_expected[1] * τ_expected[2] - τ_expected[3]^2 ≈ τ0[1] * τ0[2] - τ0[3]^2

        # stress2grid! carries a uniform particle stress onto every old-stress node
        stress2grid!(stokes, pτ, particles)
        for (A, e) in zip((stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c, stokes.τ_o.xx_v, stokes.τ_o.yy_v, stokes.τ_o.xy), (τ_expected..., τ_expected...))
            @test all(Array(A) .≈ e)
        end
    end

    @testset "per-particle rotation kernels" begin
        dt = 0.25
        # particle cells span the velocity grids, which extend beyond the pressure cells
        nip = size(particles.index)
        xx, yy, xy, ω = init_cell_arrays(particles, Val(4))
        for (n, A) in enumerate((xx, yy, xy, ω))
            @parallel (@idx nip) _fill_particles!(A, n, np)
        end
        vals0 = gather.((xx, yy, xy, ω), Ref(particles))
        expected = ntuple(_ -> fill(NaN, np, nip...), 3)
        for I in findall(active)
            r = rotate(vals0[1][I], vals0[2][I], vals0[3][I], vals0[4][I] * dt)
            foreach(n -> expected[n][I] = r[n], 1:3)
        end

        @parallel (@idx nip) JR2K.rotate_stress_particles_rotation_matrix!(xx, yy, xy, ω, particles.index, dt)
        for n in 1:3
            @test gather((xx, yy, xy)[n], particles)[active] ≈ expected[n][active]
        end
        # inactive slots are untouched
        @test all(isnan, gather(xx, particles)[.!active])

        for (A, n) in zip((xx, yy, xy), 1:3)
            @parallel (@idx nip) _fill_particles!(A, n, np)
        end
        rotate_stress_particles!((xx, yy, xy), (ω,), particles, dt)
        for n in 1:3
            @test gather((xx, yy, xy)[n], particles)[active] ≈ expected[n][active]
        end

        # the Jaumann update is the first-order expansion of the rotation:
        # τxx - 2θτxy, τyy + 2θτxy, τxy + θ(τxx - τyy)
        for (A, n) in zip((xx, yy, xy), 1:3)
            @parallel (@idx nip) _fill_particles!(A, n, np)
        end
        θ = vals0[4] .* dt
        jaumann = (
            vals0[1] .- 2 .* θ .* vals0[3],
            vals0[2] .+ 2 .* θ .* vals0[3],
            vals0[3] .+ θ .* (vals0[1] .- vals0[2]),
        )
        @parallel (@idx nip) JR2K.rotate_stress_particles_jaumann!(xx, yy, xy, ω, particles.index, dt)
        for n in 1:3
            @test gather((xx, yy, xy)[n], particles)[active] ≈ jaumann[n][active]
        end
    end
end
