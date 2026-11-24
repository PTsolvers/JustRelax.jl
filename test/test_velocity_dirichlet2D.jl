push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test, Suppressor
using GeoParams
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

const TA = PTArray(backend)

@testset "Interior Dirichlet velocity: constructor" begin
    mask = zeros(5, 6)
    mask[2:3, 2:4] .= 1
    bcs = VelocityBoundaryConditions(; dirichlet = (; Vx = (; constant = 2.0, mask = mask)))
    @test JustRelax.has_dirichlet(bcs.dirichlet.Vx)
    @test !JustRelax.has_dirichlet(bcs.dirichlet.Vy)
    @test JustRelax.isdirichlet(bcs.dirichlet.Vx, 2, 2)
    @test !JustRelax.isdirichlet(bcs.dirichlet.Vx, 1, 1)

    # a prebuilt condition is passed through unchanged
    bc = JustRelax.DirichletBoundaryCondition(zeros(5, 6), JustRelax.Mask(mask))
    @test VelocityBoundaryConditions(; dirichlet = (; Vy = bc)).dirichlet.Vy === bc

    # no `dirichlet` keeps the existing behavior
    @test !any(JustRelax.has_dirichlet, VelocityBoundaryConditions().dirichlet)

    @test_throws ArgumentError VelocityBoundaryConditions(; dirichlet = (; Vz = (; constant = 1.0, mask = mask)))

    # a constant of zero is still prescribed: the mask, not the value, marks the nodes
    zero_bcs = VelocityBoundaryConditions(; dirichlet = (; Vy = (; constant = 0.0, mask = mask)))
    @test JustRelax.isdirichlet(zero_bcs.dirichlet.Vy, 2, 2)

    # displacement conditions carry no interior conditions
    @test !any(JustRelax.has_dirichlet, JustRelax.velocity_dirichlet(DisplacementBoundaryConditions()))
    @test JustRelax.velocity_dirichlet(bcs) === bcs.dirichlet

    # the mask must be sized like the velocity component it constrains
    V = (zeros(5, 6), zeros(6, 5))
    @test isnothing(JustRelax.check_velocity_dirichlet(bcs, V))
    @test_throws DimensionMismatch JustRelax.check_velocity_dirichlet(bcs, (zeros(6, 5), zeros(6, 5)))

    # the 3D kernels take no interior conditions, so a 3D solve must reject them
    faces3D = (left = true, right = true, front = true, back = true, top = true, bot = true)
    bcs3D = VelocityBoundaryConditions(;
        free_slip = faces3D, no_slip = map(_ -> false, faces3D),
        dirichlet = (; Vz = (; constant = 1.0, mask = ones(3, 3, 3))),
    )
    V3D = ntuple(_ -> zeros(3, 3, 3), 3)
    @test_throws ArgumentError JustRelax.check_velocity_dirichlet(bcs3D, V3D)
    @test isnothing(JustRelax.check_velocity_dirichlet(VelocityBoundaryConditions(; free_slip = faces3D, no_slip = map(_ -> false, faces3D)), V3D))
end

# A rigid block translating along x inside a viscous box with free-slip walls: Vx = 1 and Vy = 0
# on the block. The block's velocities must be held exactly, the solve must converge, and the
# fluid around the block must be dragged along.
function rigid_block(igg; variational)
    ni = (16, 16)
    grid = Geometry(ni, (1.0, 1.0); origin = (0.0, 0.0))
    dt = Inf
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    phase_ratios.center.data .= 1.0
    phase_ratios.vertex.data .= 1.0

    stokes = StokesArrays(backend, ni)
    block(n) = [(7 ≤ i ≤ 11) && (7 ≤ j ≤ 11) for i in 1:n[1], j in 1:n[2]]
    mask_x = TA(Float64.(block(size(stokes.V.Vx))))
    mask_y = TA(Float64.(block(size(stokes.V.Vy))))
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        dirichlet = (;
            Vx = (; constant = 1.0, mask = mask_x),
            Vy = (; constant = 0.0, mask = mask_y),
        ),
    )
    flow_bcs!(stokes, flow_bcs)

    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @ones(ni .+ 2...), P = stokes.P, dt)
    solver_kwargs = (;
        verbose_PH = false, verbose_DR = false, iterMax = 1.0e5, total_iterMax = 2.0e5,
        nout = 100, linear_viscosity = true,
    )
    out = if variational
        ϕ = RockRatio(backend, ni)
        update_rock_ratio!(ϕ, phase_ratios, 0)
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-6)
        solve_VariationalDYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
            solver_kwargs...,
        )
    else
        dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)
        solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
            solver_kwargs...,
        )
    end
    return out, stokes, Array(mask_x) .== 1, Array(mask_y) .== 1
end

@testset "Interior Dirichlet velocity: DYREL solvers" begin
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(16, 16, 1; init_MPI = init_mpi)...)
    @testset "$(variational ? "variational" : "standard") solver" for variational in (false, true)
        out, stokes, in_x, in_y = @suppress rigid_block(igg; variational)
        Vx, Vy = Array(stokes.V.Vx), Array(stokes.V.Vy)
        @test out.converged
        @test all(==(1.0), Vx[in_x])
        @test all(iszero, Vy[in_y])
        # the residual of a prescribed row is not part of the solve
        @test all(iszero, Array(stokes.R.Rx)[in_x[2:(end - 1), 2:(end - 1)]])
        @test all(iszero, Array(stokes.R.Ry)[in_y[2:(end - 1), 2:(end - 1)]])
        # nothing outruns the block, and the closed box forces a return flow
        @test maximum(Vx[.!in_x]) < 1.0 + 1.0e-6
        @test minimum(Vx) < -0.05
        @test all(isfinite, Vx) && all(isfinite, Vy)
    end
    finalize_global_grid(; finalize_MPI = false)
end
