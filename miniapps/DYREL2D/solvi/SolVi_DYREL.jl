# SolVi (circular viscous inclusion under pure shear) solved with the DYREL solver.
# Matrix viscosity η0, inclusion viscosity Δη·η0, no buoyancy. The material is purely
# viscous and incompressible: DYREL falls back to a numerical penalty when the bulk
# modulus is Inf (see compute_bulk_viscosity_and_penalty!).

using GeoParams, CairoMakie, JLD2, LinearAlgebra
using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend

# analytical solution + Li_error(geometry, stokes, Δη, εbg, rc; order)
include("../../benchmarks/stokes2D/solvi/vizSolVi.jl")

# Set phase 2 (inclusion) inside the circle, phase 1 (matrix) outside.
function init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
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

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., circle)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., circle)
    return nothing
end

function solVi_DYREL(;
        Δη = 1.0e-3, nx = 255, ny = 255, lx = 2.0e0, ly = 2.0e0, rc = 0.2, εbg = 1.0e0,
        init_MPI = true, finalize_MPI = false, figdir = nothing
    )

    # Physical domain ------------------------------------
    ni = nx, ny
    li = lx, ly
    origin = 0.0, 0.0
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_MPI)...)
    di = @. li / (nx_g(), ny_g())
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid
    dt = 1.0             # steady viscous solve; dt only scales the penalty fallback

    # Rheology -------------------------------------------
    η0 = 1.0e0           # matrix viscosity
    ηi = Δη * η0         # inclusion viscosity
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η0),)),
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = ηi),)),
        ),
    )

    # Phase ratios (static inclusion, no advection) ------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    circle = GGU.Circle((lx / 2, ly / 2), rc)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    stokes = StokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions: background pure shear ---------
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([-x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([ y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # Solve ----------------------------------------------
    !isnothing(figdir) && take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-8)
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
            verbose_PH = !isnothing(figdir),
            verbose_DR = false,
            iterMax = 100.0e3,
            total_iterMax = 2.0e6,
            nout = 100,
            linear_viscosity = true,
            viscosity_cutoff = (-Inf, Inf),
        )
    )
    tensor_invariant!(stokes.ε)
    tensor_invariant!(stokes.τ)

    if !isnothing(figdir)
        fig = Figure(size = (1200, 1000))
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\log_{10}(\eta)")
        ax2 = Axis(fig[1, 3], aspect = 1, title = L"P")
        ax3 = Axis(fig[2, 1], aspect = 1, title = L"V_x")
        ax4 = Axis(fig[2, 3], aspect = 1, title = L"\tau_{II}")
        h1 = heatmap!(ax1, xci..., Array(log10.(stokes.viscosity.η)), colormap = :batlow)
        h2 = heatmap!(ax2, xci..., Array(stokes.P), colormap = :vik)
        h3 = heatmap!(ax3, xvi[1], xci[2], Array(stokes.V.Vx[:, 2:(end - 1)]), colormap = :roma)
        h4 = heatmap!(ax4, xci..., Array(stokes.τ.II), colormap = :batlow)
        Colorbar(fig[1, 2], h1); Colorbar(fig[1, 4], h2)
        Colorbar(fig[2, 2], h3); Colorbar(fig[2, 4], h4)
        save(joinpath(figdir, "SolVi_DYREL.png"), fig)
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), stokes, iters
end

function multiple_solVi_DYREL(; Δη = 1.0e-3, lx = 1.0e1, ly = 1.0e1, rc = 1.0e0, εbg = 1.0e0, nrange::UnitRange = 4:8)
    L2_vx, L2_vy, L2_p = Float64[], Float64[], Float64[]
    for i in nrange
        nx = ny = 2^i - 1
        geometry, stokes, = solVi_DYREL(;
            Δη = Δη, nx = nx, ny = ny, lx = lx, ly = ly, rc = rc, εbg = εbg,
            init_MPI = false, finalize_MPI = false,
        )
        L2_vxi, L2_vyi, L2_pi = Li_error(geometry, stokes, Δη, εbg, rc; order = 2)
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
    ax.ylabel = "L2 norm"

    save(joinpath(@__DIR__, "SolVi_DYREL_error.png"), f)

    jldsave(joinpath(@__DIR__, "solvi_dyrel_error.jld2"); h, L2_vx, L2_vy, L2_p, Δη, nrange)

    return f
end

# single run, only when this file is executed directly (not when `include`d for its functions)
if abspath(PROGRAM_FILE) == @__FILE__
    solVi_DYREL(; nx = 255, ny = 255, figdir = "SolVi_DYREL")
end
