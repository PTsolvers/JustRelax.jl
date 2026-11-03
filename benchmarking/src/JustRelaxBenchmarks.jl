module JustRelaxBenchmarks

using Chairmarks: @be
using Dates: UTC, now
using GeoParams: CompositeRheology, ConstantDensity, ConstantElasticity, ConstantGravity,
    DislocationCreep, DruckerPrager_regularised, LinearViscous, SetMaterialParams
using JSON: JSON
using JustRelax
using JustRelax: JustRelax2D, JustRelax3D
using ImplicitGlobalGrid: finalize_global_grid, init_global_grid
using MPI: MPI
using Statistics: mean, median, quantile

export benchmark_cases, dashboard_main, main, print_comparison, run_benchmarks, write_dashboard_data,
    write_results

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PERFORMANCE_MODEL_VERSION = "justrelax_teff_v2"
const REPOSITORY_URL = "https://github.com/PTsolvers/JustRelax.jl"

# Effective memory access per pseudo-transient iteration, counted as in Räss et al. (2022,
# GMD 15, 5757): each unknown field is read and written once, each known field is read once,
# and staggering is ignored (every field has `nᴰ` entries). Auxiliary fields such as stresses,
# fluxes, and iteration parameters are excluded. FLOPs are counted per cell and iteration from
# the solver kernels; benchmarking/README.md lists the counting rules and coefficients.
struct PerformanceModel
    flops::Int
    memory_bytes::Int
    description::String
end

struct BenchmarkCase{S, R, V}
    name::String
    group::String
    grid_cells::NTuple
    setup::S
    run::R
    validate::V
    work_units::Int
    work_unit::String
    parameters::Dict{String, Any}
    # `nothing` for rheologies whose cost per cell depends on the material state
    performance_model::Union{PerformanceModel, Nothing}
end

teff_model(flops_per_cell, unknowns, knowns, cells, ::Type{T}, description) where {T} =
    PerformanceModel(flops_per_cell * cells, (2 * unknowns + knowns) * cells * sizeof(T), description)

grid_label(ni) = join(ni, "×")

device_array(backend, A) = JustRelax.PTArray(backend)(A)

function validate_fields(label, fields...)
    for field in fields
        all(isfinite, Array(field)) || error("$label benchmark produced nonfinite values")
    end
    return nothing
end

# `solve!` only reads the residual every `nout` iterations; an `nout` larger than `iterMax`
# skips that check, so the solver runs exactly `iterMax + 1` iterations.
const NO_RESIDUAL_CHECK = 10^9

# Infinite shear and bulk moduli make the visco-elastic solver purely viscous and incompressible;
# both moduli are infinite, so their order in the 2D (G, K) and 3D (K, G) signatures is moot.
elastic_moduli(backend, ni, ::Type{T}) where {T} = ntuple(_ -> device_array(backend, fill(T(Inf), ni)), 2)

"""
Viscous Stokes flow driven by a SolCx-like buoyancy field, free-slip on every side, with a
smooth viscosity contrast.
"""
function stokes_state(backend, ::Val{2}, ni, ::Type{T}) where {T}
    li = (one(T), one(T))
    grid = JustRelax2D.Geometry(ni, li; origin = (zero(T), zero(T)))
    xc, yc = Array.(grid.xci)
    stokes = JustRelax2D.StokesArrays(backend, ni)
    stokes.viscosity.η .= device_array(backend, T[1 + 9 * (x > 0.5) for x in xc, _ in yc])
    ρg = (
        device_array(backend, zeros(T, ni)),
        device_array(backend, T[-sin(π * y) * cos(π * x) for x in xc, y in yc]),
    )
    flow_bcs = JustRelax2D.VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true)
    )
    pt_stokes = JustRelax2D.PTStokesCoeffs(li, grid.di.center; CFL = 1 / √2.1)
    return (; stokes, pt_stokes, grid, flow_bcs, ρg, moduli = elastic_moduli(backend, ni, T), iterations = Ref(0))
end

function stokes_state(backend, ::Val{3}, ni, ::Type{T}) where {T}
    li = (one(T), one(T), one(T))
    grid = JustRelax3D.Geometry(ni, li; origin = (zero(T), zero(T), zero(T)))
    xc, yc, zc = Array.(grid.xci)
    stokes = JustRelax3D.StokesArrays(backend, ni)
    stokes.viscosity.η .= device_array(backend, T[1 + 9 * (x > 0.5) for x in xc, _ in yc, _ in zc])
    ρg = (
        device_array(backend, zeros(T, ni)),
        device_array(backend, zeros(T, ni)),
        device_array(backend, T[-sin(π * z) * cos(π * x) for x in xc, _ in yc, z in zc]),
    )
    flow_bcs = JustRelax3D.VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, front = true, back = true),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )
    pt_stokes = JustRelax3D.PTStokesCoeffs(li, grid.di.center; CFL = 1 / √3.1)
    return (; stokes, pt_stokes, grid, flow_bcs, ρg, moduli = elastic_moduli(backend, ni, T), iterations = Ref(0))
end

function stokes_case(backend, n, iterations, ::Type{T}, dims::Val{D}) where {T, D}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    ni = ntuple(_ -> n, Val(D))
    setup() = stokes_state(backend, dims, ni, T)
    function run(state)
        (; stokes, pt_stokes, grid, flow_bcs, ρg, moduli) = state
        out = JR.solve!(
            stokes, pt_stokes, grid, flow_bcs, ρg, moduli..., one(T), IGG_HANDLE[];
            kwargs = (; iterMax = iterations - 1, nout = NO_RESIDUAL_CHECK, verbose = false),
        )
        state.iterations[] = out.iter
        return state
    end
    function validate(state)
        state.iterations[] == iterations ||
            error("Stokes benchmark ran $(state.iterations[]) iterations instead of $iterations")
        return validate_fields("Stokes", JR.unpack_velocity(state.stokes.V)..., state.stokes.P)
    end
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(T),
        "grid_cells" => collect(ni),
        "iterations" => iterations,
        "rheology" => "linear viscous (G = K = Inf)",
    )
    model = teff_model(
        D == 2 ? 123 : 222, D + 1, 2, n^D * iterations, T,
        "$D velocity components and pressure updated; viscosity and buoyancy read",
    )
    return BenchmarkCase(
        "APT Stokes ($(D)D, $(grid_label(ni)), $iterations iterations, $T)",
        "Stokes", ni, setup, run, validate, iterations, "iterations", parameters, model,
    )
end

"""
The `stokes_state` problem solved with `solve_DYREL!`. Two linear viscous phases reproduce its
viscosity field, from which DYREL also takes the vertex viscosity; viscosity and buoyancy are
not recomputed during the solve. One Powell-Hestenes pass with `rel_drop = 0` runs exactly
`iterMax_DR + 1` iterations, without residual checks or pseudo-time-step updates.
"""
function dyrel_stokes_case(backend, backend_JP, n, iterations, ::Type{T}, dims::Val{D}) where {T, D}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    ni = ntuple(_ -> n, Val(D))
    function setup()
        state = stokes_state(backend, dims, ni, T)
        (; stokes, grid) = state
        stiff = [x > 0.5 for x in first(Array.(grid.xci)), _ in CartesianIndices(ni[2:end])]
        phase_ratios = PhaseRatios(backend_JP, 2, ni)
        update_phase_ratios! = D == 2 ? JR.update_phase_ratios_2D! : JR.update_phase_ratios_3D!
        update_phase_ratios!(
            phase_ratios, device_array.(backend, (T.(.!stiff), T.(stiff))), grid.xci, grid.xvi
        )
        rheology = (
            material_phase(1, LinearViscous(; η = 1.0)), material_phase(2, LinearViscous(; η = 10.0)),
        )
        dt = one(T)
        args = (; T = device_array(backend, zeros(T, ni .+ 2)), P = stokes.P, dt)
        η = Array(stokes.viscosity.η)
        JR.compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
        Array(stokes.viscosity.η) == η || error("the phase viscosities differ from the APT Stokes viscosity")
        dyrel = JR.DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = eps(T))
        return merge(state, (; phase_ratios, rheology, args, dt, dyrel))
    end
    function run(state)
        (; stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt) = state
        out = JR.solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, IGG_HANDLE[];
            iterMax_PH = 1, iterMax_DR = iterations - 1, rel_drop = 0, nout = NO_RESIDUAL_CHECK,
            verbose_PH = false, verbose_DR = false, linear_viscosity = true, update_material = false,
        )
        state.iterations[] = out.iter
        return state
    end
    function validate(state)
        state.iterations[] == iterations ||
            error("DYREL Stokes benchmark ran $(state.iterations[]) iterations instead of $iterations")
        return validate_fields("DYREL Stokes", JR.unpack_velocity(state.stokes.V)..., state.stokes.P)
    end
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(T),
        "grid_cells" => collect(ni),
        "iterations" => iterations,
        "rheology" => "linear viscous (G = K = Inf)",
    )
    # The 2D stress kernel reads the vertex viscosity, the 3D one averages η to the edges.
    model = teff_model(
        D == 2 ? 268 : 786, D, D == 2 ? 4 : 3, n^D * iterations, T,
        "$D velocity components updated; pressure, viscosity$(D == 2 ? ", vertex viscosity," : "") and buoyancy read",
    )
    return BenchmarkCase(
        "DYREL Stokes ($(D)D, $(grid_label(ni)), $iterations iterations, $T)",
        "Stokes", ni, setup, run, validate, iterations, "iterations", parameters, model,
    )
end

"""
Linear heat diffusion with uniform unit properties, from a cold interior toward a hot bottom
and a cold top boundary; the side faces are insulated.
"""
function diffusion_state(backend, ::Val{D}, ni, ::Type{T}) where {D, T}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    li = ntuple(_ -> one(T), Val(D))
    grid = JR.Geometry(ni, li; origin = ntuple(_ -> zero(T), Val(D)))
    thermal = JR.ThermalArrays(backend, ni)
    K = device_array(backend, ones(T, ni))
    ρCp = device_array(backend, ones(T, ni))
    dt = T(1.0e-3)
    pt_thermal = JR.PTThermalCoeffs(backend, K, ρCp, dt, grid.di.center, li; CFL = 0.95 / √(D + 0.1))
    sides = D == 2 ? (left = true, right = true) : (left = true, right = true, front = true, back = true)
    thermal_bc = JR.TemperatureBoundaryConditions(;
        no_flux = merge(sides, (top = false, bot = false)),
        constant_value = merge(map(!, sides), (top = zero(T), bot = one(T))),
    )
    JR.thermal_bcs!(thermal, thermal_bc)
    return (; thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid, iterations = Ref(0))
end

function diffusion_case(backend, n, iterations, ::Type{T}, dims::Val{D}) where {T, D}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    ni = ntuple(_ -> n, Val(D))
    setup() = diffusion_state(backend, dims, ni, T)
    function run(state)
        (; thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid) = state
        out = JR.heatdiffusion_PT!(
            thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid;
            kwargs = (; igg = IGG_HANDLE[], iterMax = iterations, nout = iterations, verbose = false),
        )
        state.iterations[] = only(out.iter_count)
        return state
    end
    function validate(state)
        state.iterations[] == iterations ||
            error("diffusion benchmark ran $(state.iterations[]) iterations instead of $iterations")
        return validate_fields("diffusion", state.thermal.T)
    end
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(T),
        "grid_cells" => collect(ni),
        "iterations" => iterations,
    )
    model = teff_model(
        D == 2 ? 38 : 52, 1, 3, n^D * iterations, T,
        "temperature updated; old temperature, conductivity, and heat capacity read; one residual evaluation",
    )
    return BenchmarkCase(
        "PT heat diffusion ($(D)D, $(grid_label(ni)), $iterations iterations, $T)",
        "Thermal diffusion", ni, setup, run, validate, iterations, "iterations", parameters, model,
    )
end

function material_phase(Phase, laws...; Elasticity = nothing)
    return SetMaterialParams(;
        Phase, Elasticity, CompositeRheology = CompositeRheology(laws),
        Density = ConstantDensity(; ρ = 0.0), Gravity = ConstantGravity(; g = 0.0),
    )
end

"""
Visco-elasto-plastic material of `miniapps/benchmarks/stokes2D/shear_band`: linear viscosity,
elasticity, and regularized Drucker-Prager plasticity, with an inclusion of half the matrix
shear modulus. The cohesion is low enough for every cell to yield within the first time step.
"""
function shear_band_material()
    plasticity = DruckerPrager_regularised(; C = 0.1 / cosd(30), ϕ = 30, η_vp = 8.0e-3, Ψ = 0)
    phase(Phase, G) = let elasticity = ConstantElasticity(; G, Kb = 4)
        material_phase(Phase, LinearViscous(; η = 1.0), elasticity, plasticity; Elasticity = elasticity)
    end
    return (;
        rheology = (phase(1, 1.0), phase(2, 0.5)), li = (1.0, 1.0, 1.0), εbg = 1.0, dt = 0.25,
        temperature = 0.0, yields = true, description = "visco-elasto-plastic (Drucker-Prager)",
    )
end

"""
Temperature-dependent dislocation creep of the shear-heating benchmark of Duretz et al.
(2014, doi:10.1002/2014GL060438): a power-law matrix with a weaker power-law inclusion at
673 K, without elasticity or plasticity.
"""
function dislocation_creep_material()
    matrix = DislocationCreep(; A = 3.2e-20, n = 3.0, E = 276.0e3, V = 0.0, r = 0.0, R = 8.3145)
    inclusion = DislocationCreep(; A = 3.16e-26, n = 3.3, E = 186.0e3, V = 0.0, r = 0.0, R = 8.3145)
    return (;
        rheology = (material_phase(1, matrix), material_phase(2, inclusion)),
        li = (70.0e3, 70.0e3, 40.0e3), εbg = 5.0e-14, dt = 1.0e11, temperature = 673.0,
        yields = false, description = "power-law dislocation creep",
    )
end

"""
Two-phase Stokes flow with GeoParams rheology: a spherical inclusion at the center of the
domain, deformed in pure shear at the strain rate `material.εbg` (extension along `x`,
shortening along the vertical), free-slip on every side, without buoyancy.
"""
function nonlinear_stokes_state(backend, backend_JP, ::Val{D}, ni, ::Type{T}, material) where {D, T}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    (; rheology, εbg, dt) = material
    li = T.(material.li[(4 - D):3])
    grid = JR.Geometry(ni, li; origin = ntuple(_ -> zero(T), Val(D)))
    xc, xv = Array.(grid.xci), Array.(grid.xvi)

    radius = minimum(li) / 10
    inclusion = [sum(abs2, x .- li ./ 2) < radius^2 for x in Iterators.product(xc...)]
    any(inclusion) || error("the inclusion of radius $radius contains no cell center of the $ni grid")
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios! = D == 2 ? JR.update_phase_ratios_2D! : JR.update_phase_ratios_3D!
    update_phase_ratios!(
        phase_ratios, device_array.(backend, (T.(.!inclusion), T.(inclusion))), grid.xci, grid.xvi
    )

    stokes = JR.StokesArrays(backend, ni)
    velocity = JR.unpack_velocity(stokes.V)
    first(velocity) .= device_array(backend, T.(εbg .* xv[1]))
    last(velocity) .= device_array(backend, reshape(T.(-εbg .* xv[D]), ntuple(_ -> 1, D - 1)..., :))
    sides = (:left, :right, :top, :bot, :front, :back)[1:(2 * D)]
    flow_bcs = JR.VelocityBoundaryConditions(;
        free_slip = NamedTuple{sides}(ntuple(_ -> true, 2 * D)),
        no_slip = NamedTuple{sides}(ntuple(_ -> false, 2 * D)),
    )
    JR.flow_bcs!(stokes, flow_bcs)

    ρg = ntuple(_ -> device_array(backend, zeros(T, ni)), Val(D))
    args = (; T = device_array(backend, fill(T(material.temperature), ni .+ 2)), P = stokes.P, dt)
    # Strain rate and viscous stress of the initial velocity. The solvers evaluate the viscosity
    # from the stress invariant, which a power law maps to an unbounded viscosity at zero stress.
    vertical = D == 2 ? :yy : :zz
    stokes.ε.xx .= εbg
    getfield(stokes.ε, vertical) .= -εbg
    JR.compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
    @. stokes.τ.xx = 2 * stokes.viscosity.η * εbg
    getfield(stokes.τ, vertical) .= .-stokes.τ.xx
    pt_stokes = JR.PTStokesCoeffs(li, grid.di.center; CFL = 0.9 / √(D + 0.1))
    return (; stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, dt, iterations = Ref(0))
end

function nonlinear_stokes_case(backend, backend_JP, label, material, n, iterations, ::Type{T}, dims::Val{D}) where {T, D}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    ni = ntuple(_ -> n, Val(D))
    setup() = nonlinear_stokes_state(backend, backend_JP, dims, ni, T, material)
    function run(state)
        (; stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, dt) = state
        out = JR.solve!(
            stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, dt, IGG_HANDLE[];
            kwargs = (;
                iterMax = iterations - 1, nout = NO_RESIDUAL_CHECK, verbose = false,
                viscosity_cutoff = (-Inf, Inf),
            ),
        )
        state.iterations[] = out.iter
        return state
    end
    function validate(state)
        state.iterations[] == iterations ||
            error("$label benchmark ran $(state.iterations[]) iterations instead of $iterations")
        material.yields && !any(>(0), Array(state.stokes.λ)) &&
            error("$label benchmark did not yield; its plastic branch was not measured")
        return validate_fields(label, JR.unpack_velocity(state.stokes.V)..., state.stokes.P)
    end
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(T),
        "grid_cells" => collect(ni),
        "iterations" => iterations,
        "rheology" => material.description,
    )
    return BenchmarkCase(
        "$label ($(D)D, $(grid_label(ni)), $iterations iterations, $T)",
        "Nonlinear Stokes", ni, setup, run, validate, iterations, "iterations", parameters, nothing,
    )
end

"""
The `nonlinear_stokes_state` problem solved with `solve_DYREL!`: one Powell-Hestenes pass
whose dynamic-relaxation loop never meets its tolerance (`rel_drop = 0`), so the solver runs
exactly `iterMax_DR + 1` iterations, including the eigenvalue and pseudo-time-step updates
every `nout` iterations.
"""
function dyrel_case(backend, backend_JP, label, material, n, iterations, ::Type{T}, dims::Val{D}) where {T, D}
    JR = D == 2 ? JustRelax2D : JustRelax3D
    ni = ntuple(_ -> n, Val(D))
    function setup()
        state = nonlinear_stokes_state(backend, backend_JP, dims, ni, T, material)
        (; stokes, rheology, phase_ratios, grid, dt) = state
        dyrel = JR.DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = eps(T))
        return merge(state, (; dyrel))
    end
    function run(state)
        (; stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt) = state
        out = JR.solve_DYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, IGG_HANDLE[];
            iterMax_PH = 1, iterMax_DR = iterations - 1, rel_drop = 0, verbose_PH = false,
            verbose_DR = false, viscosity_cutoff = (-Inf, Inf),
        )
        state.iterations[] = out.iter
        return state
    end
    function validate(state)
        state.iterations[] == iterations ||
            error("DYREL $label benchmark ran $(state.iterations[]) iterations instead of $iterations")
        material.yields && !any(>(0), Array(state.stokes.λ)) &&
            error("DYREL $label benchmark did not yield; its plastic branch was not measured")
        return validate_fields("DYREL $label", JR.unpack_velocity(state.stokes.V)..., state.stokes.P)
    end
    parameters = Dict{String, Any}(
        "dimension" => D,
        "float_type" => string(T),
        "grid_cells" => collect(ni),
        "iterations" => iterations,
        "rheology" => material.description,
    )
    return BenchmarkCase(
        "DYREL $label ($(D)D, $(grid_label(ni)), $iterations iterations, $T)",
        "DYREL", ni, setup, run, validate, iterations, "iterations", parameters, nothing,
    )
end

# GPU grids are large enough to saturate device memory bandwidth, yet fit an 8 GB card.
function benchmark_cases(
        backend = CPUBackend; backend_JP = JustPIC.CPU,
        size_2d = backend === CPUBackend ? 128 : 2048, size_3d = backend === CPUBackend ? 32 : 128,
        iterations = 100, precision::Type = Float64,
    )
    nonlinear = (
        ("Shear band", shear_band_material()), ("Dislocation creep", dislocation_creep_material()),
    )
    return (
        stokes_case(backend, size_2d, iterations, precision, Val(2)),
        stokes_case(backend, size_3d, iterations, precision, Val(3)),
        dyrel_stokes_case(backend, backend_JP, size_2d, iterations, precision, Val(2)),
        dyrel_stokes_case(backend, backend_JP, size_3d, iterations, precision, Val(3)),
        diffusion_case(backend, size_2d, iterations, precision, Val(2)),
        diffusion_case(backend, size_3d, iterations, precision, Val(3)),
        (
            nonlinear_stokes_case(backend, backend_JP, label, material, n, iterations, precision, dims)
                for (label, material) in nonlinear for (n, dims) in ((size_2d, Val(2)), (size_3d, Val(3)))
        )...,
        (
            dyrel_case(backend, backend_JP, label, material, n, iterations, precision, dims)
                for (label, material) in nonlinear for (n, dims) in ((size_2d, Val(2)), (size_3d, Val(3)))
        )...,
    )
end

# The solvers exchange halos through ImplicitGlobalGrid, whose global grid must match the
# case's cell counts; each case initializes it for the duration of its measurement.
const IGG_HANDLE = Ref{Any}()

function with_global_grid(f, ni)
    nx, ny, nz = length(ni) == 2 ? (ni..., 1) : ni
    IGG_HANDLE[] = JustRelax.IGG(init_global_grid(nx, ny, nz; init_MPI = !MPI.Initialized(), quiet = true)...)
    try
        return f()
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

function git_metadata()
    commit = readchomp(`git -C $REPOSITORY_ROOT rev-parse HEAD`)
    status = read(`git -C $REPOSITORY_ROOT status --porcelain`, String)
    dirty = !isempty(status)
    dirty && @warn "benchmarking a dirty worktree; results will not be published" status
    subject = readchomp(`git -C $REPOSITORY_ROOT log -1 --format=%s`)
    author = readchomp(`git -C $REPOSITORY_ROOT log -1 --format=%an`)
    return (; commit, dirty, subject, author)
end

function justrelax_source()
    source = realpath(dirname(dirname(pathof(JustRelax))))
    source == realpath(REPOSITORY_ROOT) ||
        error("benchmark loaded JustRelax from $source instead of $REPOSITORY_ROOT")
    return source
end

function loaded_package_versions()
    tracked = Set(
        (
            "AMDGPU", "CUDA", "CellArrays", "GeoParams", "ImplicitGlobalGrid", "JustPIC",
            "ParallelStencil",
        ),
    )
    versions = Dict{String, String}()
    for (id, package) in Base.loaded_modules
        id.name in tracked || continue
        version = pkgversion(package)
        isnothing(version) || (versions[id.name] = string(version))
    end
    return versions
end

triad!(a::Array, b, c, s) = Threads.@threads for i in eachindex(a, b, c)
    a[i] = b[i] + s * c[i]
end
triad!(a, b, c, s) = a .= b .+ s .* c

# Eight independent chains hide FMA latency; summing them keeps them live.
function fma_chain(x, s, c, iterations)
    x1, x2, x3, x4 = x, x + 1, x + 2, x + 3
    x5, x6, x7, x8 = x + 4, x + 5, x + 6, x + 7
    for _ in 1:iterations
        x1 = muladd(x1, s, c); x2 = muladd(x2, s, c)
        x3 = muladd(x3, s, c); x4 = muladd(x4, s, c)
        x5 = muladd(x5, s, c); x6 = muladd(x6, s, c)
        x7 = muladd(x7, s, c); x8 = muladd(x8, s, c)
    end
    return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
end

fma_chains!(out::Array, x, s, c, iterations) = Threads.@threads for i in eachindex(out, x)
    out[i] = fma_chain(x[i], s, c, iterations)
end
fma_chains!(out, x, s, c, iterations) = out .= fma_chain.(x, s, c, iterations)

# Fastest of `repeats` calls of `f`, after `warmup` seconds of calls that let the device reach
# its sustained clock.
function best_time(f, synchronized; repeats = 5, warmup = 1.0)
    timed() = @elapsed (f(); synchronize(synchronized))
    start = time()
    while time() - start < warmup
        timed()
    end
    return minimum(_ -> timed(), 1:repeats)
end

"""
    measure_peaks(backend, T; n = 2^26, fma_items = 2^22, fma_iterations = 256)

Attainable `T` memory bandwidth (STREAM triad over `n` elements, counting 3 transfers per
element) and compute rate (8 FMA chains of `fma_iterations` per item), in GB/s and GFLOP/s.
The CPU chains run one item per loop iteration without SIMD across items, so the CPU compute
rate is a lower bound.
"""
function measure_peaks(
        backend, ::Type{T} = Float64; n = 2^26, fma_items = 2^22, fma_iterations = 256
    ) where {T}
    a, b, c = (device_array(backend, ones(T, n)) for _ in 1:3)
    bandwidth = 3 * n * sizeof(T) / best_time(() -> triad!(a, b, c, T(3)), a) / 1.0e9

    x = device_array(backend, T.(1:fma_items))
    out = similar(x)
    flops = 2 * 8 * fma_iterations * fma_items
    compute = flops / best_time(() -> fma_chains!(out, x, T(0.999), T(0.001), fma_iterations), out) / 1.0e9
    all(isfinite, Array(out)) || error("FMA probe produced non-finite values")
    return (; bandwidth, compute)
end

# Device arrays are synchronized by downloading one element, which works on every backend
# without depending on the vendor package.
synchronize(a::Array) = nothing
synchronize(a) = (Array(view(a, 1:1)); nothing)

function benchmark_metadata(backend, backend_name, device, ::Type{T}) where {T}
    backend_name == "CPU" || !isnothing(device) ||
        throw(ArgumentError("a device description is required for the $backend_name backend"))
    git = git_metadata()
    source = justrelax_source()
    cpu = first(Sys.cpu_info())
    cpu_model = "$(cpu.model) ($(Sys.CPU_NAME))"
    device = something(device, cpu_model)
    peaks = measure_peaks(backend, T)
    return Dict{String, Any}(
        "timestamp_utc" => string(now(UTC)),
        "commit" => git.commit,
        "commit_subject" => git.subject,
        "commit_author" => git.author,
        "dirty" => git.dirty,
        "julia_version" => string(VERSION),
        "justrelax_version" => string(pkgversion(JustRelax)),
        "justrelax_source" => relpath(source, REPOSITORY_ROOT),
        "backend" => backend_name,
        "float_type" => string(T),
        "device" => device,
        "cpu_model" => cpu_model,
        "hardware_fingerprint" =>
            "$(Sys.KERNEL) | $(Sys.ARCH) | $device | $cpu_model | $(Threads.nthreads()) threads",
        "threads" => Threads.nthreads(),
        "package_versions" => loaded_package_versions(),
        "peak_memory_bandwidth_gb_per_second" => peaks.bandwidth,
        "peak_compute_gflops" => peaks.compute,
        "peak_source" => "measured: $T STREAM triad and FMA-chain kernels",
    )
end

function measure(case::BenchmarkCase, samples, metadata)
    trial = with_global_grid(case.grid_cells) do
        case.validate(case.run(case.setup()))
        setup, run_case, validate = case.setup, case.run, case.validate
        @be setup run_case validate evals = 1 samples = samples seconds = Inf
    end
    timings = [sample.time for sample in trial.samples]
    median_time = median(timings)

    result = Dict{String, Any}(
        "name" => case.name,
        "group" => case.group,
        "unit" => "s",
        "value" => median_time,
        "time_min_seconds" => minimum(timings),
        "time_median_seconds" => median_time,
        "time_mean_seconds" => mean(timings),
        "time_max_seconds" => maximum(timings),
        "time_iqr_seconds" => quantile(timings, 0.75) - quantile(timings, 0.25),
        "samples" => length(timings),
        "allocations" => round(Int, minimum(sample -> sample.allocs, trial.samples)),
        "bytes" => round(Int, minimum(sample -> sample.bytes, trial.samples)),
        "work_units" => case.work_units,
        "work_unit" => case.work_unit,
        "throughput_per_second" => case.work_units / median_time,
        "sanity_check" => "passed",
        "parameters" => case.parameters,
        "metadata" => metadata,
    )
    return merge!(result, model_metrics(case.performance_model, median_time))
end

model_metrics(::Nothing, time) = Dict{String, Any}()

function model_metrics(model::PerformanceModel, time)
    flops_per_second = model.flops / time
    return Dict{String, Any}(
        "modeled_flops" => model.flops,
        "modeled_memory_bytes" => model.memory_bytes,
        "arithmetic_intensity_flops_per_byte" => model.flops / model.memory_bytes,
        "effective_flops_per_second" => flops_per_second,
        "effective_gflops_per_second" => flops_per_second / 1.0e9,
        "effective_bandwidth_gb_per_second" => model.memory_bytes / time / 1.0e9,
        "performance_metric_source" => "algorithmic_model",
        "performance_model_version" => PERFORMANCE_MODEL_VERSION,
        "performance_model_description" => model.description,
    )
end

function run_benchmarks(;
        backend = CPUBackend, backend_JP = JustPIC.CPU, backend_name = "CPU", device = nothing,
        samples = 10, group = "all", precision::Type = Float64,
        cases = benchmark_cases(backend; backend_JP, precision),
    )
    samples > 0 || throw(ArgumentError("samples must be positive"))
    selected = group == "all" ? cases : filter(case -> case.group == group, cases)
    isempty(selected) && throw(ArgumentError("unknown or empty benchmark group: $group"))
    metadata = benchmark_metadata(backend, backend_name, device, precision)
    return map(selected) do case
        @info "Benchmarking $(case.name)" samples
        measure(case, samples, metadata)
    end
end
function write_results(path, results)
    output = abspath(path)
    mkpath(dirname(output))
    open(output, "w") do io
        JSON.print(io, results, 2)
        write(io, '\n')
    end
    return output
end

function dashboard_run(path)
    results = JSON.parsefile(path)
    results isa Vector || error("dashboard input $path must contain a JSON array")
    isempty(results) && error("dashboard input $path contains no benchmark results")

    names = getindex.(results, "name")
    allunique(names) || error("dashboard input $path contains duplicate benchmark names")
    metadata = results[1]["metadata"]
    required = ("commit", "timestamp_utc", "backend", "hardware_fingerprint", "dirty")
    for key in required
        haskey(metadata, key) || error("dashboard input $path is missing metadata.$key")
    end
    for result in results
        result["metadata"] == metadata ||
            error("dashboard input $path mixes results from different benchmark runs")
    end

    return Dict{String, Any}(
        "source" => basename(normpath(path)),
        "commit" => metadata["commit"],
        "timestamp_utc" => metadata["timestamp_utc"],
        "backend" => metadata["backend"],
        "hardware_fingerprint" => metadata["hardware_fingerprint"],
        "dirty" => metadata["dirty"],
        "benchmarks" => results,
    )
end

function dashboard_data(inputs)
    isempty(inputs) && throw(ArgumentError("at least one dashboard input is required"))
    runs = dashboard_run.(inputs)
    identities = [(run["commit"], run["backend"], run["hardware_fingerprint"]) for run in runs]
    allunique(identities) || error(
        "dashboard inputs contain more than one run for the same commit, backend, and hardware",
    )
    sort!(runs; by = run -> run["timestamp_utc"])
    return Dict{String, Any}(
        "schema_version" => 1,
        "generated_at_utc" => string(now(UTC)),
        "repository_url" => REPOSITORY_URL,
        "runs" => runs,
    )
end

write_dashboard_data(path, inputs) = write_results(path, dashboard_data(inputs))

format_seconds(t) = t < 1.0e-3 ? "$(round(t * 1.0e6; sigdigits = 3)) μs" :
    t < 1 ? "$(round(t * 1.0e3; sigdigits = 3)) ms" : "$(round(t; sigdigits = 3)) s"

relative_spread(result) = result["time_iqr_seconds"] / result["time_median_seconds"]

format_timing(result) =
    "$(format_seconds(result["time_median_seconds"])) ±$(round(Int, 100 * relative_spread(result)))%"

"""
    print_comparison(io, baseline, candidate)

Print a markdown table of the change in median time of each benchmark from `baseline` to
`candidate`, both result vectors as written by [`write_results`](@ref), in percent. The spread
is the interquartile range as a percentage of the median. A change larger than the two spreads
combined is marked 🔴 (slower) or 🟢 (faster). Returns the candidate-to-baseline median ratios.
"""
function print_comparison(io::IO, baseline, candidate)
    base_meta, cand_meta = baseline[1]["metadata"], candidate[1]["metadata"]
    for key in ("backend", "float_type", "hardware_fingerprint")
        base_meta[key] == cand_meta[key] || error(
            "cannot compare runs with different $key: $(repr(base_meta[key])) vs $(repr(cand_meta[key]))",
        )
    end
    getindex.(baseline, "name") == getindex.(candidate, "name") ||
        error("baseline and candidate ran different benchmarks")

    ratios = map((b, c) -> c["time_median_seconds"] / b["time_median_seconds"], baseline, candidate)
    rows = map(baseline, candidate, ratios) do base, cand, ratio
        noise = relative_spread(base) + relative_spread(cand)
        marker = ratio > 1 + noise ? " 🔴" : ratio < 1 - noise ? " 🟢" : ""
        change = round(100 * (ratio - 1); digits = 1)
        [
            cand["name"], "$(change > 0 ? "+" : "")$(change)%$marker", format_timing(base),
            format_timing(cand), "$(base["allocations"]) → $(cand["allocations"])",
        ]
    end
    header = ["Benchmark", "Change", "Baseline", "Candidate", "Allocations"]
    widths = [maximum(length, getindex.([[header]; collect(rows)], j)) for j in eachindex(header)]
    cells(row) = join((j == 1 ? rpad(x, w) : lpad(x, w) for (j, (x, w)) in enumerate(zip(row, widths))), " | ")
    label(meta) = "`$(first(meta["commit"], 8))`" * (meta["dirty"] ? " (dirty)" : "")

    println(io, "Baseline $(label(base_meta)) vs candidate $(label(cand_meta)) on `$(cand_meta["hardware_fingerprint"])`")
    println(io)
    println(io, "| ", cells(header), " |")
    println(io, "| :", "-"^(widths[1] - 1), " | ", join(("-"^(w - 1) * ":" for w in widths[2:end]), " | "), " |")
    foreach(row -> println(io, "| ", cells(row), " |"), rows)
    return ratios
end

const PRECISIONS = Dict("Float64" => Float64, "Float32" => Float32)

function parse_commandline(args)
    output = "benchmark_results.json"
    samples = 10
    group = "all"
    precision = Float64
    for arg in args
        if startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--samples=")
            samples = parse(Int, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--group=")
            group = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--precision=")
            name = split(arg, '='; limit = 2)[2]
            precision = get(PRECISIONS, name) do
                throw(ArgumentError("unknown precision $(repr(name)); use --precision=Float64|Float32"))
            end
        else
            throw(
                ArgumentError(
                    "unknown argument $arg; use --output=PATH, --samples=N, --group=NAME, or --precision=TYPE"
                )
            )
        end
    end
    return (; output, samples, group, precision)
end

function main(
        args = ARGS; backend = CPUBackend, backend_JP = JustPIC.CPU, backend_name = "CPU", device = nothing,
    )
    options = parse_commandline(args)
    results = run_benchmarks(;
        backend, backend_JP, backend_name, device, options.samples, options.group, options.precision
    )
    output = write_results(options.output, results)
    @info "Wrote benchmark results" output
    return results
end

function parse_dashboard_commandline(args)
    output = "benchmark_history.json"
    inputs = String[]
    for arg in args
        if startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        elseif startswith(arg, "--input=")
            push!(inputs, split(arg, '='; limit = 2)[2])
        else
            throw(ArgumentError("unknown argument $arg; use --input=PATH or --output=PATH"))
        end
    end
    isempty(inputs) && push!(inputs, "benchmark_results.json")
    return (; output, inputs)
end

function dashboard_main(args = ARGS)
    options = parse_dashboard_commandline(args)
    output = write_dashboard_data(options.output, options.inputs)
    @info "Wrote benchmark history" output
    return output
end

end
