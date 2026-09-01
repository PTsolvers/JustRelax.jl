@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end
using Test
using GeoParams
using JustRelax, JustRelax.JustRelax3D

using ParallelStencil, ParallelStencil.FiniteDifferences3D
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

using JustPIC, JustPIC._3D
const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    JustPIC.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPUBackend
end

# all active particles carry the value `v`
function particles_equal(A, index, v)
    Ac, idx = Array(A), Array(index)
    for k in axes(idx, 3), j in axes(idx, 2), i in axes(idx, 1)
        for ip in cellaxes(idx)
            @index(idx[ip, i, j, k]) || continue
            isapprox(@index(Ac[ip, i, j, k]), v) || return false
        end
    end
    return true
end

@testset "Stress rotation on particles 3D" begin
    ni = 4, 4, 4
    li = 1.0, 1.0, 1.0
    grid = Geometry(ni, li; origin = (0.0, 0.0, 0.0))

    stokes = StokesArrays(backend_JR, ni)
    particles = init_particles(backend, 12, 24, 8, grid.xi_vel...)
    pτ = StressParticles(particles)
    index = particles.index
    dt = 0.1

    # uniform stress state and rigid rotation about z
    τxx, τyy, τzz, τyz, τxz, τxy = 1.0, -1.0, 0.5, 0.0, 0.0, 2.0
    ωxy = 0.3
    fill!(stokes.τ.xx, τxx)
    fill!(stokes.τ.yy, τyy)
    fill!(stokes.τ.zz, τzz)
    fill!(stokes.τ.xy_c, τxy)
    fill!(stokes.τ.xy, τxy)
    fill!(stokes.ω.xy, ωxy)

    rotate_stress!(pτ, stokes, particles, dt)
    pτxx, pτyy, pτzz = JustRelax.normal_stress(pτ)
    pτyz, pτxz, pτxy = JustRelax.shear_stress(pτ)

    # a rotation about z reproduces the 2D result for the same vorticity: the 3D
    # GeoParams routine takes the curl, twice the vorticity
    ref = GeoParams.rotate_elastic_stress2D(ωxy, (τxx, τyy, τxy), dt)
    @test particles_equal(pτxx, index, ref[1])
    @test particles_equal(pτyy, index, ref[2])
    @test particles_equal(pτxy, index, ref[3])
    # components out of the rotation plane are untouched
    @test particles_equal(pτzz, index, τzz)
    @test particles_equal(pτyz, index, τyz)
    @test particles_equal(pτxz, index, τxz)

    # vanishing vorticity leaves the stress unchanged instead of producing NaNs
    fill!(stokes.ω.xy, 0.0)
    fill!(stokes.τ.xx, τxx)
    fill!(stokes.τ.yy, τyy)
    fill!(stokes.τ.xy_c, τxy)
    rotate_stress!(pτ, stokes, particles, dt)
    @test particles_equal(pτxx, index, τxx)
    @test particles_equal(pτyy, index, τyy)
    @test particles_equal(pτzz, index, τzz)
    @test particles_equal(pτxy, index, τxy)

    # the shear history consumed by the stress kernels is restored on both the
    # cell centers and the cell edges
    for A in (stokes.τ_o.xy_c, stokes.τ_o.xy, stokes.τ_o.yz_c, stokes.τ_o.yz)
        fill!(A, NaN)
    end
    stress2grid!(stokes, pτ, particles)
    @test all(isfinite, Array(stokes.τ_o.yz_c))
    @test all(x -> isapprox(x, τxy), Array(stokes.τ_o.xy_c))
    @test all(isfinite, Array(stokes.τ_o.xy)[2:(end - 1), 2:(end - 1), :])
    @test all(isfinite, Array(stokes.τ_o.yz)[:, 2:(end - 1), 2:(end - 1)])
end
