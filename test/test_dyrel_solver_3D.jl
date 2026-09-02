push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

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

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPU
end

@parallel_indices (i, j, k) function _init_single_phase_solver_3D!(phases)
    @index phases[1, i, j, k] = 1.0
    return nothing
end

@testset "DYREL 3D solver" begin
    ni = 8, 4, 8
    init_mpi = !JustRelax.MPI.Initialized()
    igg = IGG(init_global_grid(ni...; init_MPI = init_mpi)...)

    try
        grid = Geometry(ni, (1.0, 0.5, 1.0))
        dt = 1.0
        elasticity = ConstantElasticity(; G = 1.0, Kb = 5.0)
        rheology = (
            SetMaterialParams(;
                Phase = 1,
                Density = ConstantDensity(; ρ = 0.0),
                Gravity = ConstantGravity(; g = 0.0),
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), elasticity)),
                Elasticity = elasticity,
            ),
        )

        phase_ratios = PhaseRatios(backend_JP, 1, ni)
        for ratios in (
                phase_ratios.center, phase_ratios.vertex,
                phase_ratios.yz, phase_ratios.xz, phase_ratios.xy,
            )
            @parallel (@idx size(ratios)) _init_single_phase_solver_3D!(ratios)
        end

        stokes = StokesArrays(backend_JR, ni)
        nx, ny, nz = ni
        stokes.V.Vx .= PTArray(backend_JR)(
            [
                sinpi((i - 1) / nx) * sinpi((j - 1) / (ny + 1)) * sinpi((k - 1) / (nz + 1))
                    for i in 1:(nx + 1), j in 1:(ny + 2), k in 1:(nz + 2)
            ]
        )
        stokes.V.Vy .= PTArray(backend_JR)(
            [
                -0.7 * sinpi((i - 1) / (nx + 1)) * sinpi((j - 1) / ny) * sinpi((k - 1) / (nz + 1))
                    for i in 1:(nx + 2), j in 1:(ny + 1), k in 1:(nz + 2)
            ]
        )
        stokes.V.Vz .= PTArray(backend_JR)(
            [
                0.4 * sinpi((i - 1) / (nx + 1)) * sinpi((j - 1) / (ny + 1)) * sinpi((k - 1) / nz)
                    for i in 1:(nx + 2), j in 1:(ny + 2), k in 1:(nz + 1)
            ]
        )

        flow_bcs = VelocityBoundaryConditions(;
            free_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
            no_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        )
        flow_bcs!(stokes, flow_bcs)
        update_halo!(@velocity(stokes)...)

        ρg = ntuple(_ -> @zeros(ni...), Val(3))
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
        dyrel = JustRelax3D.DYREL(
            backend_JR, stokes, rheology, phase_ratios, grid.di, dt;
            ϵ = 1.0e-6, CFL = 0.99,
        )

        linear_rheology = (
            SetMaterialParams(;
                Phase = 1,
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
            ),
        )
        linear_stokes = StokesArrays(backend_JR, ni)
        linear_stokes.viscosity.η .= PTArray(backend_JR)(
            [
                j + 2k for _ in 1:nx, j in 1:ny, k in 1:nz
            ]
        )
        linear_stokes.ε.yz .= 1.0
        linear_args = (; T = @zeros(ni .+ 2...), P = linear_stokes.P, dt = dt)
        linear_dyrel = JustRelax3D.DYREL(
            backend_JR, linear_stokes, linear_rheology, phase_ratios, grid.di, dt
        )
        θc = copy(linear_dyrel.P_num)
        JustRelax3D.compute_stress_viscosity_DRYEL!(
            linear_stokes, θc, linear_dyrel.γ_eff, linear_rheology, phase_ratios,
            1.0, dt, 1.0, linear_args, (-Inf, Inf), true,
        )
        η = Array(linear_stokes.viscosity.η)
        ηyz = η[2, 1, 1], η[2, 2, 1], η[2, 1, 2], η[2, 2, 2]
        @test Array(linear_stokes.τ.yz)[2, 2, 2] ≈
            2 * length(ηyz) / sum(inv, ηyz)

        @test_throws ErrorException solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            kwargs = (;
                free_surface = true,
                verbose_PH = false,
                verbose_DR = false,
            ),
        )
        out = solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            kwargs = (;
                verbose_PH = false,
                verbose_DR = false,
                iterMax = 5.0e3,
                total_iterMax = 5.0e3,
                nout = 20,
                rel_drop = 0.1,
                viscosity_relaxation = 1.0,
                linear_viscosity = true,
            ),
        )

        @test !isempty(out.err_evo_tot)
        @test out.iter > 0
        @test all(isfinite, out.err_evo_tot)
        @test last(out.err_evo_tot) < first(out.err_evo_tot)
        @test last(out.err_evo_tot) < dyrel.ϵ
        @test maximum(abs, Array(stokes.V.Vx)) < 1.0e-4
        @test maximum(abs, Array(stokes.V.Vy)) < 1.0e-4
        @test maximum(abs, Array(stokes.V.Vz)) < 1.0e-4
    finally
        finalize_global_grid(; finalize_MPI = init_mpi)
    end
end
