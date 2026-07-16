# SolKz (smooth exponential viscosity in y) solved with the DYREL solver.
# solve_DYREL! always derives viscosity from GeoParams rheology + phase_ratios
# (compute_viscosity!), so a spatially varying field can't be written to
# stokes.viscosity.η directly. Instead the exact discretized field (identical to
# the normal-Stokes SolKz benchmark) is exposed through a single-phase
# CustomRheology that reads it back per-cell via `args.η_target`.

using GeoParams, CairoMakie, JLD2, LinearAlgebra
using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
const backend_JP = JustPIC.CPUBackend

# analytical solution + Li_error(geometry, stokes; order)
include("../../benchmarks/stokes2D/solkz/vizSolKz.jl")

function solKz_viscosity(xci, ni; B = log(1.0e6))
    _, yc = xci
    η = @zeros(ni...)
    @parallel_indices (i, j) function _viscosity!(η, yc, B)
        η[i, j] = exp(B * yc[j])
        return nothing
    end
    @parallel (@idx ni) _viscosity!(η, yc, B)
    return η
end

function solKz_density(xci, ni; km = 2)
    xc, yc = xci
    ρ = @zeros(ni...)
    # matches Stokes2D_SolKz_Zhong1996's reference density: ρ = -σ*sin(km*y)*cos(kn*x), σ=1, kn=3π
    @parallel_indices (i, j) function _density!(ρ, xc, yc, km)
        ρ[i, j] = -sin(km * yc[j]) * cos(3 * π * xc[i])
        return nothing
    end
    @parallel (@idx ni) _density!(ρ, xc, yc, km)
    return ρ
end

# single trivial phase everywhere: viscosity comes from args.η_target, not from phase ratios
function init_phases!(phase_ratios)
    ni = size(phase_ratios.center)
    @parallel_indices (i, j) function init_phases!(phases)
        @index phases[1, i, j] = 1.0
        return nothing
    end
    @parallel (@idx ni) init_phases!(phase_ratios.center)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex)
    return nothing
end

# viscosity passthrough: η is whatever `args.η_target` holds at this cell, no strain-rate dependence
@inline custom_εII(a::CustomRheology, TauII; η_target = 1.0, kwargs...) = TauII / η_target * 0.5
@inline custom_τII(a::CustomRheology, EpsII; η_target = 1.0, kwargs...) = 2.0 * η_target * EpsII

function solKz_DYREL(;
        Δη = 1.0e6, km = 2, nx = 64, ny = 64, lx = 1.0e0, ly = 1.0e0,
        init_MPI = true, finalize_MPI = false, figdir = nothing
    )

    ni = nx, ny
    li = lx, ly
    origin = 0.0, 0.0
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_MPI)...)
    di = @. li / (nx_g(), ny_g())
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid

    ## (Physical) Time domain and discretization
    ttot = 1   # total simulation time
    Δt = 1     # physical time step
    dt = 0.1

    η_target = solKz_viscosity(xci, ni; B = log(Δη))
    ρ = solKz_density(xci, ni; km = km)

    # density passthrough: solve_DYREL! recomputes ρg from rheology every outer iteration
    # (compute_ρg!/update_ρg!), so a precomputed ρg array would be silently overwritten.
    # Vector_Density looks the value back up per-cell via args.index instead.
    creep = CustomRheology(custom_εII, custom_τII, NamedTuple())
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = Vector_Density(; rho = vec(ρ)),
            Gravity = ConstantGravity(; g = 1.0),
            CompositeRheology = CompositeRheology((creep,)),
        ),
    )

    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(phase_ratios)

    stokes = StokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    index_field = reshape(1:prod(ni), ni)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt, η_target = η_target, index = index_field)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    !isnothing(figdir) && take(figdir)
    # γfact = 200: at the γfact = 20 default the penalty is too weak for the pressure
    # error to track APT's (resolution-persistent ~1e-4 floor); 200 matches APT's L2_p
    # at ~5x the iteration cost
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8, γfact = 200.0)

    # Physical time loop
    t = 0.0
    local iters
    while t < ttot
        iters = solve_DYREL!(
            stokes,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            rheology,
            args,
            grid,
            dt,
            igg;
            kwargs = (;
                verbose_PH = true,
                verbose_DR = false,
                iterMax = 150.0e3,
                # must exceed the cumulative Powell-Hestenes budget: a γfact = 200 solve
                # needs ~500k iterations at nx = 127 and more at finer resolution
                total_iterMax = 200e3,
                nout = 100,
                linear_viscosity = true,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        t += Δt
    end

    if !isnothing(figdir)
        fig = Figure(size = (1200, 500))
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\log_{10}(\eta)")
        ax2 = Axis(fig[1, 3], aspect = 1, title = L"P")
        h1 = heatmap!(ax1, xci..., Array(log10.(stokes.viscosity.η)), colormap = :batlow)
        h2 = heatmap!(ax2, xci..., Array(stokes.P), colormap = :vik)
        Colorbar(fig[1, 2], h1); Colorbar(fig[1, 4], h2)
        save(joinpath(figdir, "SolKz_DYREL.png"), fig)
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

function multiple_solKz_DYREL(; Δη = 1.0e6, km = 2, nrange::UnitRange = 4:10)
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, = solKz_DYREL(; Δη = Δη, km = km, nx = nx, ny = ny, init_MPI = !JustRelax.MPI.Initialized(), finalize_MPI = false)
        L2_vxi, L2_vyi, L2_pi = Li_error(geometry, stokes; order = 1, Δη = Δη, km = km)
        push!(L2_vx, L2_vxi)
        push!(L2_vy, L2_vyi)
        push!(L2_p, L2_pi)
    end

    nx = @. 2^nrange - 1
    h = @. (1 / nx)

    f = Figure(; fontsize = 28)
    ax = Axis(
        f[1, 1];
        yscale = log10,
        xscale = log10,
        yminorticksvisible = true,
        yminorticks = IntervalsBetween(8),
    )
    lines!(ax, h, (L2_vx); linewidth = 3, label = "Vx")
    lines!(ax, h, (L2_vy); linewidth = 3, label = "Vy")
    lines!(ax, h, (L2_p); linewidth = 3, label = "P")
    axislegend(ax; position = :rt)
    ax.xlabel = "h"
    ax.ylabel = "L1 norm"

    save(joinpath(@__DIR__, "SolKz_DYREL_error.png"), f)

    jldsave(joinpath(@__DIR__, "solkz_dyrel_error.jld2"); h, L2_vx, L2_vy, L2_p, Δη, km, nrange)

    return f
end

# single run, only when this file is executed directly (not when `include`d for its functions)
if abspath(PROGRAM_FILE) == @__FILE__
    solKz_DYREL(; nx = 64, ny = 64, figdir = "SolKz_DYREL")
end
