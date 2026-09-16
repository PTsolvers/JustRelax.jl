push!(LOAD_PATH, "..")
@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end

using JustPIC
import JustPIC.GridGeometryUtils as GGU
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

# single trivial phase everywhere: viscosity/density are supplied directly via `args`
function init_single_phase!(phase_ratios)
    ni = size(phase_ratios.center)
    @parallel_indices (i, j) function _init!(phases)
        @index phases[1, i, j] = 1.0
        return nothing
    end
    @parallel (@idx ni) _init!(phase_ratios.center)
    @parallel (@idx ni .+ 1) _init!(phase_ratios.vertex)
    return nothing
end

# viscosity passthrough: η is whatever `args.η_target` holds at this cell (SolCx, SolKz)
@inline passthrough_εII(a::CustomRheology, TauII; η_target = 1.0, kwargs...) = TauII / η_target * 0.5
@inline passthrough_τII(a::CustomRheology, EpsII; η_target = 1.0, kwargs...) = 2.0 * η_target * EpsII

# density passthrough: ρ is whatever `args.ρ_target` holds at this cell (SolCx, SolKz). The field
# travels through `args` rather than inside the rheology, because a `MaterialParams` holding an
# array is not a bitstype and cannot be passed to a GPU kernel.
struct PassthroughDensity <: AbstractDensity{Float64} end
@inline GeoParams.compute_density(::PassthroughDensity, args) = args.ρ_target

function solve_sol_benchmark(rheology, phase_ratios, args, flow_bcs, grid, dt, igg; γfact, ϵ, iterMax, variational)
    ni = size(phase_ratios.center)
    stokes = StokesArrays(backend_JR, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    kwargs = (;
        verbose_PH = false, verbose_DR = false,
        iterMax = iterMax, total_iterMax = iterMax,
        nout = 100, rel_drop = 0.1,
        linear_viscosity = true, viscosity_cutoff = (-Inf, Inf),
    )

    if variational
        ϕ = RockRatio(backend_JR, ni)
        update_rock_ratio!(ϕ, phase_ratios, 0)
        dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ, γfact)
        iters = solve_VariationalDYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg; kwargs)
    else
        dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ, γfact)
        iters = solve_DYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg; kwargs)
    end
    return stokes, iters
end

# ------------------------------------------------------------------------------------ #
# SolCx: viscosity jump across x = 0.5
# ------------------------------------------------------------------------------------ #
function solcx_viscosity(xci, ni; Δη = 1.0e6)
    xc, _ = xci
    η = @zeros(ni...)
    @parallel_indices (i, j) function _viscosity!(η, xc, Δη)
        η[i, j] = ifelse(xc[i] ≤ 0.5, 1.0e0, Δη)
        return nothing
    end
    @parallel (@idx ni) _viscosity!(η, xc, Δη)
    return η
end

function solcx_density(xci, ni)
    xc, yc = xci
    ρ = @zeros(ni...)
    @parallel_indices (i, j) function _density!(ρ, xc, yc)
        ρ[i, j] = sin(π * yc[j]) * cos(π * xc[i])
        return nothing
    end
    @parallel (@idx ni) _density!(ρ, xc, yc)
    return ρ
end

function run_solcx(igg, nx, ny; variational, Δη = 1.0e6, γfact = 20, ϵ = 1.0e-6)
    ni = nx, ny
    li = 1.0, 1.0
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    dt = 1.0

    η_target = solcx_viscosity(grid.xci, ni; Δη = Δη)
    ρ = solcx_density(grid.xci, ni)
    creep = CustomRheology(passthrough_εII, passthrough_τII, NamedTuple())
    rheology = (
        SetMaterialParams(;
            Phase = 1, Density = PassthroughDensity(),
            Gravity = ConstantGravity(; g = 1.0), CompositeRheology = CompositeRheology((creep,)),
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_single_phase!(phase_ratios)

    args = (; T = @zeros(ni .+ 2...), P = @zeros(ni...), dt = dt, η_target = η_target, ρ_target = ρ)
    flow_bcs = VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))

    return solve_sol_benchmark(rheology, phase_ratios, args, flow_bcs, grid, dt, igg; γfact, ϵ, iterMax = 5.0e3 * nx, variational)
end

# ------------------------------------------------------------------------------------ #
# SolKz: smooth exponential viscosity in y
# ------------------------------------------------------------------------------------ #
function solkz_viscosity(xci, ni; B = log(1.0e6))
    _, yc = xci
    η = @zeros(ni...)
    @parallel_indices (i, j) function _viscosity!(η, yc, B)
        η[i, j] = exp(B * yc[j])
        return nothing
    end
    @parallel (@idx ni) _viscosity!(η, yc, B)
    return η
end

function solkz_density(xci, ni; km = 2)
    xc, yc = xci
    ρ = @zeros(ni...)
    @parallel_indices (i, j) function _density!(ρ, xc, yc, km)
        ρ[i, j] = -sin(km * yc[j]) * cos(3 * π * xc[i])
        return nothing
    end
    @parallel (@idx ni) _density!(ρ, xc, yc, km)
    return ρ
end

function run_solkz(igg, nx, ny; variational, Δη = 1.0e6, km = 2, γfact = 3, ϵ = 1.0e-6)
    ni = nx, ny
    li = 1.0, 1.0
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    dt = 0.1

    η_target = solkz_viscosity(grid.xci, ni; B = log(Δη))
    ρ = solkz_density(grid.xci, ni; km = km)
    creep = CustomRheology(passthrough_εII, passthrough_τII, NamedTuple())
    rheology = (
        SetMaterialParams(;
            Phase = 1, Density = PassthroughDensity(),
            Gravity = ConstantGravity(; g = 1.0), CompositeRheology = CompositeRheology((creep,)),
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_single_phase!(phase_ratios)

    args = (; T = @zeros(ni .+ 2...), P = @zeros(ni...), dt = dt, η_target = η_target, ρ_target = ρ)
    flow_bcs = VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))

    return solve_sol_benchmark(rheology, phase_ratios, args, flow_bcs, grid, dt, igg; γfact, ϵ, iterMax = 5.0e3 * nx, variational)
end

# ------------------------------------------------------------------------------------ #
# SolVi: circular viscous inclusion under pure shear
# ------------------------------------------------------------------------------------ #
function solvi_init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)
    @parallel_indices (i, j) function _init!(phases, xc, yc, circle)
        x, y = xc[i], yc[j]
        if GGU.inside(GGU.Point(x, y), circle)
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0
        else
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0
        end
        return nothing
    end
    @parallel (@idx ni) _init!(phase_ratios.center, xci..., circle)
    @parallel (@idx ni .+ 1) _init!(phase_ratios.vertex, xvi..., circle)
    return nothing
end

function run_solvi(igg, nx, ny; variational, Δη = 1.0e-3, lx = 2.0, ly = 2.0, rc = 0.2, εbg = 1.0, ϵ = 1.0e-8)
    ni = nx, ny
    li = lx, ly
    grid = Geometry(ni, li; origin = (0.0, 0.0))
    (; xvi) = grid
    dt = 1.0

    η0 = 1.0
    ηi = Δη * η0
    rheology = (
        SetMaterialParams(;
            Phase = 1, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η0),)),
        ),
        SetMaterialParams(;
            Phase = 2, Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = ηi),)),
        ),
    )
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    circle = GGU.Circle((lx / 2, ly / 2), rc)
    solvi_init_phases!(phase_ratios, grid.xci, xvi, circle)

    args = (; T = @zeros(ni .+ 2...), P = @zeros(ni...), dt = dt)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )

    # background pure-shear velocity, set on a scratch StokesArrays before the real solve so
    # both solver paths start from the identical non-zero initial condition
    ny_ = ny
    Vx0 = PTArray(backend_JR)([-x * εbg for x in xvi[1], _ in 1:(ny_ + 2)])
    Vy0 = PTArray(backend_JR)([y * εbg for _ in 1:(nx + 2), y in xvi[2]])

    stokes = StokesArrays(backend_JR, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    stokes.V.Vx .= Vx0
    stokes.V.Vy .= Vy0
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    kwargs = (;
        verbose_PH = false, verbose_DR = false,
        iterMax = 100.0e3, total_iterMax = 200.0e3, nout = 100,
        linear_viscosity = true, viscosity_cutoff = (-Inf, Inf),
    )

    if variational
        ϕ = RockRatio(backend_JR, ni)
        update_rock_ratio!(ϕ, phase_ratios, 0)
        dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ)
        iters = solve_VariationalDYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg; kwargs)
    else
        dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ)
        iters = solve_DYREL!(stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg; kwargs)
    end
    return stokes, iters
end

# ------------------------------------------------------------------------------------ #
@testset "Sol benchmarks DYREL (plain and variational)" begin
    begin
        nx = ny = 32
        init_mpi = JustRelax.MPI.Initialized() ? false : true
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

        @testset "SolCx" begin
            stokes, iters = run_solcx(igg, nx, ny; variational = false)
            stokes_vs, iters_vs = run_solcx(igg, nx, ny; variational = true)

            @test iters.err_evo_tot[end] < 1.0e-6
            @test iters_vs.converged
            @test all(isfinite, Array(stokes.P)) && all(isfinite, Array(stokes_vs.P))
            @test maximum(abs, Array(stokes.V.Vx)) ≈ 0.0013939995765589134 rtol = 1.0e-6
            @test maximum(abs, Array(stokes.V.Vy)) ≈ 0.0031585683659085865 rtol = 1.0e-6
            @test maximum(abs, Array(stokes.P)) ≈ 0.25700842902418092 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.V.Vx)) ≈ 0.0016306775444691899 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.V.Vy)) ≈ 0.0035636067234839051 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.P)) ≈ 0.25341866240639133 rtol = 1.0e-6
        end

        @testset "SolKz" begin
            stokes, iters = run_solkz(igg, nx, ny; variational = false)
            stokes_vs, iters_vs = run_solkz(igg, nx, ny; variational = true)

            @test iters.err_evo_tot[end] < 1.0e-6
            @test iters_vs.converged
            @test all(isfinite, Array(stokes.P)) && all(isfinite, Array(stokes_vs.P))
            @test maximum(abs, Array(stokes.V.Vx)) ≈ 0.00010280771330343728 rtol = 1.0e-6
            @test maximum(abs, Array(stokes.V.Vy)) ≈ 6.0491243393540986e-5 rtol = 1.0e-6
            @test maximum(abs, Array(stokes.P)) ≈ 0.099060763007113209 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.V.Vx)) ≈ 0.00010415876959958915 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.V.Vy)) ≈ 6.1781558832754387e-5 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.P)) ≈ 0.1024518065842967 rtol = 1.0e-6
        end

        @testset "SolVi" begin
            stokes, iters = run_solvi(igg, nx, ny; variational = false)
            stokes_vs, iters_vs = run_solvi(igg, nx, ny; variational = true)

            @test iters.err_evo_tot[end] < 1.0e-6
            @test iters_vs.converged
            @test all(isfinite, Array(stokes.P)) && all(isfinite, Array(stokes_vs.P))
            @test maximum(abs, Array(stokes.V.Vx)) ≈ 2.0 rtol = 1.0e-6
            @test maximum(abs, Array(stokes.V.Vy)) ≈ 2.0 rtol = 1.0e-6
            @test maximum(abs, Array(stokes.P)) ≈ 2.517811125543723 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.V.Vx)) ≈ 2.0 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.V.Vy)) ≈ 2.0 rtol = 1.0e-6
            @test maximum(abs, Array(stokes_vs.P)) ≈ 2.0521009636837029 rtol = 1.0e-6
        end

        finalize_global_grid(; finalize_MPI = true)
    end
end
