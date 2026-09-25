# 3D variational Stokes on a sinking sphere.
#
# A: with every rock fraction equal to one the variational solver is the standard solver,
#    so both must converge to the same velocity field.
# B: with a flat free surface below sticky air the variational solver must converge.

push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
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

using JustPIC

const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

@parallel_indices (I...) function sphere_phases!(phases, px, py, pz, index, zsurf)
    for ip in cellaxes(phases)
        @index(index[ip, I...]) || continue
        x = @index px[ip, I...]
        y = @index py[ip, I...]
        z = @index pz[ip, I...]
        in_sphere = (x - 0.5)^2 + (y - 0.5)^2 + (z + 0.5)^2 ≤ 0.15^2
        @index phases[ip, I...] = in_sphere ? 2.0 : (z > zsurf ? 3.0 : 1.0)
    end
    return nothing
end

function sinking_sphere(solver; n = 16, zsurf = Inf, no_slip_top = false, free_surface = false)
    ni = n, n, n
    li = 1.0, 1.0, 1.0
    igg = IGG(init_global_grid(n, n, n; init_MPI = !JustRelax.MPI.Initialized(), quiet = true)...)
    grid = Geometry(ni, li; origin = (0.0, 0.0, -1.0))
    el = ConstantElasticity(; G = 1.0e6, ν = 0.25)
    material(phase, ρ, η) = SetMaterialParams(;
        Phase = phase,
        Density = ConstantDensity(; ρ = ρ),
        Gravity = ConstantGravity(; g = 1.0),
        CompositeRheology = CompositeRheology((LinearViscous(; η = η), el)),
        Elasticity = el,
    )
    rheology = (material(1, 1.0, 1.0), material(2, 2.0, 0.1), material(3, 0.0, 1.0e-2))

    particles = init_particles(backend_JP, 20, 40, 10, grid.xi_vel...)
    pPhases, = init_cell_arrays(particles, Val(1))
    @parallel (@idx size(pPhases)) sphere_phases!(pPhases, particles.coords..., particles.index, zsurf)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, minimum.(grid.di.vertex); ϵ_abs = 1.0e-8, ϵ_rel = 1.0e-6, CFL = 0.9 / √3.1)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, front = true, back = true, top = !no_slip_top, bot = true),
        no_slip = (left = false, right = false, front = false, back = false, top = no_slip_top, bot = false),
    )
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    compute_ρg!(ρg[3], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    out = if solver === :standard
        solve!(
            stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, 1.0, igg;
            kwargs = (; iterMax = 10.0e3, nout = 100, verbose = false)
        )
    else
        ϕ = RockRatio(backend, ni)
        if isinf(zsurf)
            foreach(f -> fill!(getfield(ϕ, f), 1.0), fieldnames(typeof(ϕ)))
        else
            surf = init_marker_surface(backend_JP, grid.xvi[1], grid.xvi[2], zsurf)
            compute_rock_fraction!(ϕ, surf, grid.xvi, grid.di.vertex)
        end
        solve_VariationalStokes!(
            stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, ϕ, rheology, args, 1.0, igg;
            iterMax = 10.0e3, nout = 100, verbose = false, air_phase = isinf(zsurf) ? 0 : 3, free_surface,
        )
    end
    finalize_global_grid(; finalize_MPI = false)
    return out, Array(stokes.V.Vz), stokes
end

@testset "Variational Stokes 3D" begin
    @testset "matches standard solver when all rock fractions are one" begin
        out_std, Vz_std, _ = sinking_sphere(:standard)
        out_vs, Vz_vs, _ = sinking_sphere(:variational)
        @test out_std.iter < 10.0e3
        @test out_vs.iter < 10.0e3
        @test minimum(Vz_vs) < 0
        # The two stress kernels average viscosity onto the shear edges differently;
        # the gap is a discretization difference (4% at n = 16, 1.6% at n = 24).
        @test isapprox(Vz_vs, Vz_std; rtol = 5.0e-2)
    end

    @testset "converges with a flat free surface" begin
        out, Vz, _ = sinking_sphere(:variational; zsurf = -0.2)
        @test out.iter < 10.0e3
        @test all(isfinite, Vz)
        @test minimum(Vz) < 0
    end

    @testset "converges with free-surface stabilization" begin
        out, Vz, _ = sinking_sphere(:variational; zsurf = -0.2, free_surface = true)
        @test out.iter < 10.0e3
        @test all(isfinite, Vz)
        @test minimum(Vz) < 0
    end

    @testset "no-slip top wall carries shear strain rate" begin
        _, _, stokes = sinking_sphere(:standard; no_slip_top = true)
        @test maximum(abs, Array(stokes.ε.xz)[:, :, end]) > 0
        @test maximum(abs, Array(stokes.ε.yz)[:, :, end]) > 0
    end

    @testset "3D solvers reject nonuniform grids" begin
        xv = [0.0, 0.1, 0.3, 0.6, 1.0]
        grid = Geometry(Array, xv, xv, xv)
        @test_throws "supports only uniform grids in 3D" JustRelax3D.require_uniform_spacing(grid, "`solve!`")
    end
end

@testset "Cross-edge averages reach the last vertex plane" begin
    nx, ny, nz = 4, 4, 4
    M = JustRelax3D
    # xz edges: x-vertex, y-centre, z-vertex, filled with their x- and z-vertex index
    xz_x = [Float64(i) for i in 1:(nx + 1), j in 1:ny, k in 1:(nz + 1)]
    xz_z = [Float64(k) for i in 1:(nx + 1), j in 1:ny, k in 1:(nz + 1)]
    # yz edge at the last x-centre and the top z-vertex
    Ic = M.clamped_indices((nx, ny, nz), nx, 2, nz + 1)
    @test M.av_clamped_yz_y(xz_x, Ic...) == nx + 0.5
    @test M.av_clamped_yz_y(xz_z, Ic...) == nz + 1
    # yz edges: x-centre, y-vertex, z-vertex, filled with their y-vertex index
    yz_y = [Float64(j) for i in 1:nx, j in 1:(ny + 1), k in 1:(nz + 1)]
    # xz edge at the last y-centre
    Ic = M.clamped_indices((nx, ny, nz), 2, ny, 2)
    @test M.av_clamped_xz_x(yz_y, Ic...) == ny + 0.5
end
