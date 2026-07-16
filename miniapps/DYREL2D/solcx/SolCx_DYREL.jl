# SolCx (viscosity jump across x=0.5) solved with the DYREL solver.
# solve_DYREL! always derives viscosity from GeoParams rheology + phase_ratios
# (compute_viscosity!), so a spatially varying field can't be written to
# stokes.viscosity.η directly. Instead the exact discretized field (identical to
# the normal-Stokes SolCx benchmark) is exposed through a single-phase
# CustomRheology that reads it back per-cell via `args.η_target`.

using GeoParams, CairoMakie, JLD2, LinearAlgebra
using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
const backend_JP = JustPIC.CPUBackend

# analytical solution + solcx_error(geometry, stokes; order)
include("../../benchmarks/stokes2D/solcx/vizSolCx.jl")

@parallel function smooth!(
        A2::AbstractArray{T, 2}, A::AbstractArray{T, 2}, fact::Real
    ) where {T}
    @inn(A2) = @inn(A) + 1.0 / 4.1 / fact * (@d2_xi(A) + @d2_yi(A))
    return nothing
end

function solCx_viscosity(xci, ni; Δη = 1.0e6)
    xc, _ = xci
    η = @zeros(ni...)
    @parallel_indices (i, j) function _viscosity!(η, xc, Δη)
        η[i, j] = ifelse(xc[i] ≤ 0.5, 1.0e0, Δη)
        return nothing
    end
    @parallel (@idx ni) _viscosity!(η, xc, Δη)

    # smooth the viscosity jump exactly like the normal-Stokes SolCx benchmark
    # (5 Laplacian passes) so both solvers discretize the identical problem and
    # the L2-error comparison is solver-only, not solver+problem
    η2 = deepcopy(η)
    for _ in 1:5
        @parallel smooth!(η2, η, 1.0)
        @views η2[1, :] .= η2[2, :]
        @views η2[end, :] .= η2[end - 1, :]
        @views η2[:, 1] .= η2[:, 2]
        @views η2[:, end] .= η2[:, end - 1]
        η, η2 = η2, η # swap
    end
    return η
end

function solCx_density(xci, ni)
    xc, yc = xci
    ρ = @zeros(ni...)
    # matches Stokes2D_SolCx_Zhong1996's reference density: ρ = sin(π*y)*cos(π*x)
    @parallel_indices (i, j) function _density!(ρ, xc, yc)
        ρ[i, j] = sin(π * yc[j]) * cos(π * xc[i])
        return nothing
    end
    @parallel (@idx ni) _density!(ρ, xc, yc)
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

function solCx_DYREL(;
        Δη = 1.0e6, nx = 64, ny = 64, lx = 1.0e0, ly = 1.0e0,
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
    dt = 1   # matches the dt the normal-Stokes SolCx script passes to solve!

    η_target = solCx_viscosity(xci, ni; Δη = Δη)
    ρ = solCx_density(xci, ni)

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
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    !isnothing(figdir) && take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8)

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
                iterMax = 500.0e3,
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
        save(joinpath(figdir, "SolCx_DYREL.png"), fig)
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

function multiple_solCx_DYREL(; Δη = 1.0e6, nrange::UnitRange = 6:10)
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, = solCx_DYREL(; Δη = Δη, nx = nx, ny = ny, init_MPI = !JustRelax.MPI.Initialized(), finalize_MPI = false)
        L2_vxi, L2_vyi, L2_pi = solcx_error(geometry, stokes; order = 1, Δη = Δη)
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
    axislegend(ax; position = :lt)
    ax.xlabel = "h"
    ax.ylabel = "L1 norm"

    save(joinpath(@__DIR__, "SolCx_DYREL_error.png"), f)

    jldsave(joinpath(@__DIR__, "solcx_dyrel_error.jld2"); h, L2_vx, L2_vy, L2_p, Δη, nrange)

    return f
end

# single run, only when this file is executed directly (not when `include`d for its functions)
if abspath(PROGRAM_FILE) == @__FILE__
    solCx_DYREL(; nx = 64, ny = 64, figdir = "SolCx_DYREL")
end
