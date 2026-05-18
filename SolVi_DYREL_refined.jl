# using CUDA

using GeoParams, GLMakie
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
# @init_parallel_stencil(CUDA, Float64, 2)
@init_parallel_stencil(Threads, Float64, 2)

# const backend = CUDABackend
const backend = CPUBackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend
# const backend_JP = CUDABackend

using PoissonGrids

# HELPER FUNCTIONS ----------------------------------- ----------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, circle)
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, circle)
        x, y = xc[i], yc[j]
        p = GGU.Point(x, y)
        if GGU.inside(p, circle)
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

# MAIN SCRIPT --------------------------------------------------------------------
function main(xvi, igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    grid = Geometry(PTArray(backend), xvi...)
    (; xci, ni) = grid # nodes at the center and vertices of the cells

    # Physical properties using GeoParams ----------------
    τ_y = 1.6          # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30             # friction angle
    C = τ_y            # Cohesion
    η0 = 1.0           # viscosity
    G0 = 1.0           # elastic shear modulus
    Gi = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
    εbg = 1.0           # background strain-rate
    η_reg = 1.5e-2 #1e-2 #8.0e-3          # regularisation "viscosity"
    dt = η0 / G0 / 4.0     # assumes Maxwell time of 4
    ηinc = 10
    visc_mat = LinearViscous(; η = 1)
    visc_inc = LinearViscous(; η = ηinc)

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_mat, )),

        ),
        # High density phase
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_inc, )),
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.2
    origin = 0.0, 0.0
    circle = GGU.Circle(origin, radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    solvi        = solvi_solution(grid, ηinc, radius, εbg)
    stokes.V.Vx .= PTArray(backend)(solvi.vx)
    stokes.V.Vy .= PTArray(backend)(solvi.vy)
    
    # stokes.V.Vx .= PTArray(backend)([-x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    # stokes.V.Vy .= PTArray(backend)([ y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    @views stokes.V.Vx[2:(end - 1), 2:(end - 1)] .= 0
    @views stokes.V.Vy[2:(end - 1), 2:(end - 1)] .= 0
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-9)

    # IO -------------------------------------------------
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    tmax = 5
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]

    # Stokes solver ----------------
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
            verbose_DR = false,
            verbose_PH = true,
            iterMax = 50.0e3,
            nout = 1,
            rel_drop = 0.75,
            # λ_relaxation = 0,
            λ_relaxation_DR = 1,
            λ_relaxation_PH = 1,
            viscosity_relaxation = 1,
            viscosity_cutoff = (-Inf, Inf),
        )
    )

    tensor_invariant!(stokes.ε)
    tensor_invariant!(stokes.ε_pl)
    push!(τII, maximum(stokes.τ.xx))

    it += 1
    t += dt

    push!(sol, solution(εbg, t, G0, η0))
    push!(ttot, t)

    println("it = $it; t = $t \n")

    # visualisation
    th = 0:(pi / 50):(3 * pi)
    xunit = @. radius * cos(th) + 0.5
    yunit = @. radius * sin(th) + 0.5

    fig = Figure(size = (1600, 1600), title = "t = $t")
    ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
    ax2 = Axis(fig[2, 1], aspect = 1, title = L"E_{II}", titlesize = 35)
    ax3 = Axis(fig[1, 2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
    ax4 = Axis(fig[2, 2], aspect = 1)
    heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :lipari)
    # h1 = heatmap!(ax1, xci..., Array((stokes.viscosity.η_vep)) , colormap=:batlow)
    heatmap!(ax2, xci..., Array(stokes.EII_pl), colormap = :lipari)
    heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :lipari)
    lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
    lines!(ax4, ttot, τII, color = :black)
    lines!(ax4, ttot, sol, color = :red)
    hidexdecorations!(ax1)
    hidexdecorations!(ax3)
    # Colorbar(fig[1, 2], h1)
    fig
    save(joinpath(figdir, "$(it).png"), fig)

    fig = Figure(size = (1600, 1600), title = "t = $t")
    ax1 = Axis(fig[1, 1], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
    # heatmap!(ax1, xci..., Array(log10.(stokes.ε.II)), colormap = :lipari)
    ax1 = Axis(fig[1, 1], aspect = 1, title = L"$$ P ", titlesize = 35)
    heatmap!(ax1, xci..., Array(stokes.P), colormap = :lipari)

    for ui in xci[1]
        hlines!(ax1, ui; xmin = 0, xmax = 1, color = :black)
    end
    for ui in xci[2]
        vlines!(ax1, ui; ymin = 0, ymax = 1, color = :black)
    end
    fig
    save(joinpath(figdir, "refined_stress_$(it).png"), fig)

    # return iters
    return stokes, solvi, grid, iters
end

include("miniapps/benchmarks/stokes2D/solvi/vizSolVi.jl")

n  = 128
# n  = parse(Int, ARGS[1])
nx = n
ny = n
figdir = "Solvi_DYREL_refined"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end


xv  = LinRange(-1.0, 1.0, n + 1) # uniform grid 
xvi = xv, xv
@time stokes_reg, sol_reg, grid_reg, iters_reg=main(xvi, igg; figdir = figdir, nx = nx, ny = ny);

di_c  = [dx*dy for dx in grid_reg.di.vertex[1], dy in grid_reg.di.vertex[2]]
di_vx = [dx*dy for dx in grid_reg.di.center[1], dy in grid_reg.di.vertex[2]]
di_vy = [dx*dy for dx in grid_reg.di.vertex[1], dy in grid_reg.di.center[2]]

eP_reg  = sqrt(sum((abs2.(stokes_reg.P.-sol_reg.p)) .* di_c))
eVx_reg = sqrt(sum((abs2.(stokes_reg.V.Vx[2:end-1, 2:end-1].-sol_reg.vx[2:end-1, 2:end-1])) .* di_vx))
eVy_reg = sqrt(sum((abs2.(stokes_reg.V.Vy[2:end-1, 2:end-1].-sol_reg.vy[2:end-1, 2:end-1])) .* di_vy))

println("L2 error in P: $eP_reg")
println("L2 error in Vx: $eVx_reg")
println("L2 error in Vy: $eVy_reg")

M = window_monitor(2,  10, 2.0e-1, 0.0)
M = window_monitor(8, 10, 2.0e-1, 0.0)

xv_ref = solve_grid(-1.0, 1.0, M, n) # refined grid
xvi = xv_ref, xv_ref

@time stokes, sol, grid, iters=main(xvi, igg; figdir = figdir, nx = nx, ny = ny);

# di_c  = [dx*dy for dx in grid.di.vertex[1], dy in grid.di.vertex[2]]
# di_vx = [dx*dy for dx in grid.di.center[1], dy in grid.di.vertex[2]]
# di_vy = [dx*dy for dx in grid.di.vertex[1], dy in grid.di.center[2]]

di_c  == grid.di.vertex[1] * grid.di.vertex[2]'
di_vx == grid.di.center[1] * grid.di.vertex[2]'
di_vy == grid.di.vertex[1] * grid.di.center[2]'

eP  = sqrt(sum((abs2.(stokes.P.-sol.p)) .* di_c))
eVx = sqrt(sum((abs2.(stokes.V.Vx[2:end-1, 2:end-1].-sol.vx[2:end-1, 2:end-1])) .* di_vx))
eVy = sqrt(sum((abs2.(stokes.V.Vy[2:end-1, 2:end-1].-sol.vy[2:end-1, 2:end-1])) .* di_vy))

println("L2 error in P: $eP")
println("L2 error in Vx: $eVx")
println("L2 error in Vy: $eVy")

# f,ax,h=heatmap(grid.xci..., log10.(abs.(stokes.P .- sol.p)); colorrange=(-8, 0), colormap = :lipari)
# Colorbar(f[1, 2], h)
# f

# f_reg,ax,h=heatmap(grid_reg.xci..., log10.(abs.(stokes_reg.P .- sol_reg.p)); colorrange=(-8, 0),  colormap = :lipari)
# Colorbar(f_reg[1, 2], h)
# f_reg

# •  α: size of the monitor
#    increase inside the
#    window.
# •  κ: transition
#    sharpness. Larger
#    values produce steeper
#    window edges.
# •  b: half-width of the
#    refined window.
# •  c: center of the
#    refined window.
# M = window_monitor(2, 50 / 2, 2.0e-1, 0.0)
κ = [0, 10:10:50...]
# κ = [1, 10:12...]
its  = Float64[]
itsV = Float64[]
itsP = Float64[]
eP   = Float64[]
eVx  = Float64[]
eVy  = Float64[]
x    = Vector{Float64}[]
for κ in κ
    # M = window_monitor(0.5, κ, 2.0e-1, 0.0)
    M = window_monitor(8, κ, 2.0e-1, 0.0)
    println("Running with κ = $κ")
    xv_ref = solve_grid(-1.0, 1.0, M, n) # refined grid
    xvi = xv_ref, xv_ref
    # scatter((xv_ref[1:end-1]+xv_ref[2:end])./2, diff(xv_ref))
    @time stokes, sol, grid, iters=main(xvi, igg; figdir = figdir, nx = nx, ny = ny);

    di_c  == grid.di.vertex[1] * grid.di.vertex[2]'
    di_vx == grid.di.center[1] * grid.di.vertex[2]'
    di_vy == grid.di.vertex[1] * grid.di.center[2]'

    eP_i  = sqrt(sum((abs2.(stokes.P.-sol.p)) .* di_c))
    eVx_i = sqrt(sum((abs2.(stokes.V.Vx[2:end-1, 2:end-1].-sol.vx[2:end-1, 2:end-1])) .* di_vx))
    eVy_i = sqrt(sum((abs2.(stokes.V.Vy[2:end-1, 2:end-1].-sol.vy[2:end-1, 2:end-1])) .* di_vy))

    println("Done! $(length(iters.err_evo_V)) iterations recorded with κ=$(κ).")
    push!(itsV, iters.err_evo_V[end])
    push!(itsP, iters.err_evo_P[end])
    push!(its, length(iters.err_evo_V))
    push!(x, xv_ref)
    push!(eP, eP_i)
    push!(eVx, eVx_i)
    push!(eVy, eVy_i)
end

fig = Figure(size=(900,1200))
ax1 = Axis(fig[1,1], xlabel="κ", ylabel="Number of iterations")
ax2 = Axis(fig[2,1], xlabel="κ", ylabel="error", yscale = log10)
ax3 = Axis(fig[3,1], xlabel="x", ylabel="Δx")
ax4 = Axis(fig[4,1], xlabel="x", ylabel="growth rate of Δx")
scatterlines!(ax1, κ, its, markersize = 10)
scatterlines!(ax2, κ, eVx, markersize = 10, label = "V")
scatterlines!(ax2, κ, eP, markersize = 10, label = "P")
for (x, κ) in zip(x, κ)
    du = diff(x)
    lines!(ax3, (x[1:end-1] + x[2:end]) ./2, du, label="κ=$(κ)")
    # lines!(ax4, x[1:end-2], du[1:end-1] ./ du[2:end], label="κ=$(κ)")
    lines!(ax4,  du[1:end-1] ./ du[2:end], label="κ=$(κ)")
end
# axislegend(ax3, position = :rb)
axislegend(ax2, position = :rb)
# save(joinpath(figdir, "num_its_nx$(n).png"), fig)
fig

# M = window_monitor(8, 100, 0.2, 0.0)
# xv_ref = solve_grid(-1.0, 1.0, M, 64) # refined grid
# scatterlines!((xv_ref[1:end-1] + xv_ref[2:end]) ./ 2, diff(xv_ref))
