# Brittle-ductile transition with the Drucker-Prager tensile cap, solved with
# variational DYREL.
#
# Same model as `BrittleDuctile2D.jl` (Popov, Berlie and Kaus 2025, Sect. 4.5 and Fig. 9):
# a 100 x 25 km crustal section extended horizontally under a free surface, with a
# 20 K/km geotherm and a temperature- and stress-dependent quartzite rheology, so that
# the brittle-ductile transition falls inside the domain. `run_case(; with_cap = false)`
# is the control: the paper states the near-surface mode-I zones cannot form without a
# tensile yield surface, while the mode-II shear bands must survive its removal.
#
# What differs from the pseudo-transient version:
#   - The free surface is a marker chain with cut cells (`RockRatio`), so the air is
#     excluded from the momentum balance instead of being a soft sticky-air layer. This
#     is closer to the paper's free surface, and it removes the viscosity contrast that
#     forces the pseudo-transient run to use an unphysically stiff air.
#   - `solve_VariationalDYREL!` reports convergence directly, so a step that fails to
#     converge is recorded rather than inferred from residual norms.
#
# Still different from the paper, as in the pseudo-transient version: uniform grid
# instead of their variable-resolution triangles, temperature held at the initial
# geotherm, fixed timestep instead of their adaptive halving, and a plastic-strain seed
# that lives on the grid rather than on the particles.

# const isCUDA = false
const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using Pkg;
Pkg.activate("miniapps");

const backend = @static if isCUDA
    JustRelax.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC

const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

using GeoParams, GLMakie
using Random: MersenneTwister

include("BrittleDuctile_setup.jl")

# everything above the marker chain is air; the chain itself carries the free surface
function init_phases!(phases, particles, air_top)
    ni = size(phases)
    @parallel_indices (i, j) function _init!(phases, px, py, index, air_top)
        @inbounds for ip in cellaxes(phases)
            @index(index[ip, i, j]) == 0 && continue
            y = @index py[ip, i, j]
            @index phases[ip, i, j] = y > air_top ? 1.0 : 2.0
        end
        return nothing
    end
    return @parallel (@idx ni) _init!(phases, particles.coords..., particles.index, air_top)
end

# One snapshot of the run: where it is failing, how it is failing, and the free surface.
function snapshot(fig_dir, tag, it, t, stokes, ϕ, chain, xci)
    fig = Figure(size = (2100, 1000))
    kyr = round(t / yr / 1.0e3; digits = 2)
    x, y = xci[1] ./ 1.0e3, xci[2] ./ 1.0e3
    chain_x = Array(chain.cell_vertices) ./ 1.0e3
    chain_y = Array(chain.h_vertices) ./ 1.0e3

    rock = Array(ϕ.center)
    panels = (
        (1, 1, L"E_{II}^{pl}", Array(stokes.EII_pl), :batlow, true),
        (1, 2, L"E_{Vol}^{pl}", Array(stokes.EVol_pl), :batlow, true),
        (1, 3, L"\dot{\lambda}", Array(stokes.λ), :batlow, true),
        (2, 1, L"\tau_{II}\ \text{[MPa]}", Array(stokes.τ.II) ./ 1.0e6, :lipari, true),
        (2, 2, L"P\ \text{[MPa]}", Array(stokes.P) ./ 1.0e6, :vik, true),
        (2, 3, "rock fraction", rock, :grayC, false),
    )
    for (row, col, title, field, cmap, mask) in panels
        ax = Axis(fig[row, col], title = title, xlabel = "x [km]", ylabel = "y [km]")
        # a cut cell that is mostly air carries no meaningful solver state, so blank it
        # rather than draw it; the rock-fraction panel shows the cut cells themselves
        plotted = mask ? ifelse.(rock .≥ 0.5, field, NaN) : field
        hm = heatmap!(ax, x, y, plotted, colormap = cmap)
        Colorbar(fig[row, col][1, 2], hm)
        lines!(ax, chain_x, chain_y, color = :red, linewidth = 2)
    end
    Label(fig[0, 1:3], "t = $(kyr) kyr, step $(it)", fontsize = 20)
    take(fig_dir)
    return save(joinpath(fig_dir, "$(tag)_$(lpad(it, 4, '0')).png"), fig)
end

function write_vtk(vtkdir, tag, it, t, stokes, thermal, ϕ, chain, particles, pPhases, xci, xvi, Vx_v, Vy_v)
    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    data_v = (; Vx = Array(Vx_v), Vy = Array(Vy_v), Vel = Array(@. √(Vx_v^2 + Vy_v^2)))
    data_c = (;
        P = Array(stokes.P),
        tau_II = Array(stokes.τ.II),
        eps_II = Array(stokes.ε.II),
        eps_vol_pl = Array(stokes.ε_vol_pl),
        EII_pl = Array(stokes.EII_pl),
        EVol_pl = Array(stokes.EVol_pl),
        lambda = Array(stokes.λ),
        eta_vep = Array(stokes.viscosity.η_vep),
        T = Array(thermal.T[2:(end - 1), 2:(end - 1)]),
        rock_fraction = Array(ϕ.center),
    )
    base = joinpath(vtkdir, tag)
    save_vtk(
        joinpath(vtkdir, "$(tag)_" * lpad("$it", 6, "0")), xvi, xci, data_v, data_c,
        (Array(Vx_v), Array(Vy_v)); t = t, pvd = base
    )
    save_marker_chain(joinpath(vtkdir, "$(tag)_chain_" * lpad("$it", 6, "0")), chain; t = t, pvd = base * "_chain")
    return save_particles(
        particles, pPhases; fname = joinpath(vtkdir, "$(tag)_particles_" * lpad("$it", 6, "0")),
        t = t, pvd = base * "_particles"
    )
end

# MAIN SCRIPT ---------------------------------------------------------------------
function run_case(
        igg; with_cap = true, nx = 96, ny = 36, nsteps = 20, dt_yr = 1.0e3,
        thick_air = 5.0e3, seed = 1234, figdir = nothing, do_vtk = true, nplot = 1,
    )
    # lx = 100.0e3
    # ly_crust = 25.0e3

    lx = 60.0e3
    ly_crust = 12.0e3

    ly = ly_crust + thick_air
    ni = nx, ny
    li = lx, ly
    di = @. li / ni
    origin = 0.0, -ly_crust
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid
    grid_vxi = velocity_grids(xci, xvi, di)
    air_top = 0.0                       # free surface sits at y = 0

    rheology = rheology_setup(; with_cap = with_cap)
    dt = dt_yr * yr

    # particles carry the phases
    nxcell, max_xcell, min_xcell = 24, 36, 12
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, air_top)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # marker chain: the free surface, and the cut-cell fractions built from it
    nxcell_chain, max_xcell_chain, min_xcell_chain = 100, 200, 15
    chain = init_markerchain(
        backend_JP, nxcell_chain, min_xcell_chain, max_xcell_chain, xvi[1], air_top
    )
    ϕ = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ, chain, xvi, di)

    stokes = StokesArrays(backend, ni)

    thermal = ThermalArrays(backend, ni)
    @parallel (@idx ni) init_T!(thermal.T, xci[2], air_top)

    args = (; T = thermal.T, P = stokes.P, dt = dt)

    # lithostatic pressure of the rock column only, then undo the cut-cell weighting
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, args; air_phase = air_phase)
    ρg_rock = ρg[2] .* ϕ.center
    compute_lithostatic_pressure!(stokes.P, ρg_rock, di[2], igg)
    @. stokes.P = ifelse(ϕ.center > 0, stokes.P / ϕ.center, 0)

    stokes.EII_pl .= PTArray(backend)(plastic_strain_seed(xci, ni, lx, air_top; seed = seed))

    viscosity_cutoff = (1.0e18, 1.0e25)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase = air_phase)

    # horizontal extension at a constant background rate, free surface on top
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
        free_surface = false,
    )
    stokes.V.Vx .= PTArray(backend)([(x - 0.5lx) * εbg for x in xvi[1], _ in 1:(ny + 2)])
    fill!(stokes.V.Vy, 0.0)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; ϵ = 1.0e-6)

    # output: figures per step, and a ParaView time series next to them
    tag = with_cap ? "cap" : "nocap"
    vtkdir = figdir === nothing ? nothing : joinpath(figdir, "vtk")
    if vtkdir !== nothing && do_vtk
        take(vtkdir)
    end
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    t = 0.0
    unconverged = Int[]
    for it in 1:nsteps
        compute_ρg!(ρg[2], phase_ratios, rheology, args; air_phase = air_phase)
        compute_viscosity!(
            stokes, phase_ratios, args, rheology, viscosity_cutoff; air_phase = air_phase
        )

        result = solve_VariationalDYREL!(
            stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
            kwargs = (;
                air_phase = air_phase,
                iterMax = 50.0e3,
                total_iterMax = 50.0e3,
                viscosity_relaxation = 1.0e-2,
                nout = 50,
                rel_drop = 1e-2,
                free_surface = true,
                viscosity_cutoff = viscosity_cutoff,
                verbose = false,
            ),
        )
        result.converged || push!(unconverged, it)

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε_pl)

        # advect the material, then the free surface, then rebuild the cut cells
        advection!(particles, RungeKutta2(), @velocity(stokes), dt)
        move_particles!(particles, particle_args)
        semilagrangian_advection_markerchain!(
            chain, RungeKutta2(), @velocity(stokes), grid_vxi, xvi, dt
        )
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
        inject_particles_phase!(particles, pPhases, (), ())
        update_phase_ratios!(phase_ratios, particles, pPhases)
        compute_rock_fraction!(ϕ, chain, xvi, di)

        t += dt

        if figdir !== nothing && (it == 1 || rem(it, nplot) == 0)
            snapshot(figdir, tag, it, t, stokes, ϕ, chain, xci)
            do_vtk && write_vtk(
                vtkdir, tag, it, t, stokes, thermal, ϕ, chain, particles, pPhases,
                xci, xvi, Vx_v, Vy_v,
            )
        end

        println(
            "  it = ", it, "  t = ", round(t / yr / 1.0e3; digits = 2), " kyr",
            result.converged ? "" : "  (NOT CONVERGED)",
            "  err = ", round(result.err; sigdigits = 3),
            "  τII_max = ", round(maximum(Array(stokes.τ.II)) / 1.0e6; digits = 2), " MPa",
            "  EVol_max = ", round(maximum(Array(stokes.EVol_pl)); sigdigits = 3),
        )
    end

    r = diagnostics(stokes, xci, nx, air_top, nsteps, unconverged; with_cap = with_cap, ny = ny)
    figdir === nothing || save_figure(figdir, r, xci, with_cap ? "cap_VDYREL" : "nocap_VDYREL")
    return r
end

ny = 32
nx = 5 * ny
nsteps = 20
do_vtk = true
nplot = 1
figdir = "BrittleDuctile2D_VariationalDYREL"
with_cap = true
# once a global grid exists, `Geometry` takes its spacing from it, so `init_global_grid`
# has to be given the same `nx`, `ny` that `run_case` allocates with
igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !(JustRelax.MPI.Initialized()))...)
# for with_cap in (true, false)
# println(with_cap ? "with tensile cap" : "control: no tensile cap")
# push!(results, run_case(igg; with_cap, nx, ny, nsteps, figdir, do_vtk, nplot))
# finalize_global_grid(; finalize_MPI = false)

run_case(
    igg; with_cap = with_cap, nx = nx, ny = ny, nsteps = nsteps, dt_yr = 1.0e3,
    thick_air = 2.5e3, seed = 1234, figdir = figdir, do_vtk = do_vtk, nplot = nplot,
)