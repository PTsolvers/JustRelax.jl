# Brittle-ductile transition with the Drucker-Prager tensile cap.
#
# After Popov, Berlie and Kaus (2025), Sect. 4.5 and Fig. 9: a 100 x 25 km crustal
# section is extended horizontally under a free surface, with a 20 K/km geotherm and a
# temperature- and stress-dependent quartzite rheology, so that the brittle-ductile
# transition falls inside the domain. Mode-II shear bands develop in the brittle part,
# and mode-I (tensile) zones open near the free surface where the confining stress is
# smallest. The paper states the near-surface tensile zones are not reproducible without
# a tensile yield surface, so `run_case(; with_cap = false)` is the control: the shear
# bands must survive it and the tensile zones must not.
#
# Differences from the paper, all deliberate:
#   - Uniform staggered grid instead of their variable-resolution triangles (60 m in a
#     40 x 7 km window, 500 m elsewhere). `nx`/`ny` set the resolution; the default is
#     coarse enough to run on a laptop and too coarse to resolve their band widths.
#   - Free surface through a sticky-air layer, not a deforming boundary.
#   - The temperature field is the initial geotherm, held fixed: the paper's setup does
#     not depend on thermal evolution over 33.5 kyr.
#   - Fixed timestep. The paper halves dt when its global Newton fails to converge; this
#     script reports a step whose pseudo-transient solve did not converge instead.
#   - Localization is seeded by a random perturbation of the accumulated plastic strain
#     on the grid. It is not advected with the particles, so it marks where failure
#     starts rather than travelling with the material.

const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
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

using GeoParams, CairoMakie
using Random: MersenneTwister

include("BrittleDuctile_setup.jl")

# crust below the air interface; the sticky air above carries no strength
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

# MAIN SCRIPT ---------------------------------------------------------------------
function run_case(
        igg; with_cap = true, nx = 256, ny = 96, nsteps = 5, dt_yr = 1.0e3,
        thick_air = 5.0e3, seed = 1234, figdir = nothing,
    )
    lx = 100.0e3
    ly_crust = 25.0e3
    ly = ly_crust + thick_air
    ni = nx, ny
    li = lx, ly
    di = @. li / ni
    grid = Geometry(ni, li; origin = (0.0, -ly_crust))
    (; xci, xvi) = grid
    air_top = 0.0                       # free surface sits at y = 0

    rheology = rheology_setup(; with_cap = with_cap)
    dt = dt_yr * yr

    # particles carry the phases, so the sticky-air interface can move
    nxcell, max_xcell, min_xcell = 24, 36, 12
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, air_top)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-5, ϵ_rel = 1.0e-5, CFL = 0.75 / √2.1)

    thermal = ThermalArrays(backend, ni)
    @parallel (@idx ni) init_T!(thermal.T, xci[2], air_top)

    args = (; T = thermal.T, P = stokes.P, dt = dt)

    ρg = @zeros(ni...), @zeros(ni...)
    for _ in 1:2
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        compute_lithostatic_pressure!(stokes.P, ρg[2], di[2], igg)
    end

    # random seed of accumulated plastic strain in the central upper crust, as in the
    # paper's Sect. 4.3 setup, so that softening has somewhere to start
    seed_field = zeros(ni...)
    rng = MersenneTwister(seed)
    for j in axes(seed_field, 2), i in axes(seed_field, 1)
        x, y = xci[1][i], xci[2][j]
        if abs(x - 0.5lx) < 20.0e3 && -7.0e3 < y < air_top
            seed_field[i, j] = 0.02 * rand(rng)
        end
    end
    stokes.EII_pl .= PTArray(backend)(seed_field)

    viscosity_cutoff = (1.0e20, 1.0e24)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # horizontal extension at a constant background rate, free surface on top
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    stokes.V.Vx .= PTArray(backend)([(x - 0.5lx) * εbg for x in xvi[1], _ in 1:(ny + 2)])
    fill!(stokes.V.Vy, 0.0)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    t = 0.0
    unconverged = Int[]
    for it in 1:nsteps
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

        iters = solve!(
            stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, rheology, args, dt, igg;
            kwargs = (
                verbose = false, iterMax = 200.0e3, nout = 5.0e3,
                viscosity_cutoff = viscosity_cutoff, free_surface = true,
            )
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε_pl)

        converged = iters.norm_Rx[end] < 1.0e-4 && iters.norm_Ry[end] < 1.0e-4
        converged || push!(unconverged, it)

        advection!(particles, RungeKutta2(), @velocity(stokes), dt)
        move_particles!(particles, particle_args)
        inject_particles_phase!(particles, pPhases, (), ())
        update_phase_ratios!(phase_ratios, particles, pPhases)

        t += dt
        println(
            "  it = ", it, "  t = ", round(t / yr / 1.0e3; digits = 2), " kyr",
            "  PT iters = ", iters.iter, converged ? "" : "  (NOT CONVERGED)",
            "  τII_max = ", round(maximum(Array(stokes.τ.II)) / 1.0e6; digits = 2), " MPa",
            "  EVol_max = ", round(maximum(Array(stokes.EVol_pl)); sigdigits = 3),
        )
    end

    r = diagnostics(stokes, xci, nx, air_top, nsteps, unconverged; with_cap = with_cap, ny = ny)
    figdir === nothing || save_figure(figdir, r, xci, with_cap ? "cap_PT" : "nocap_PT")
    return r
end

function main(; nx = 256, ny = 96, nsteps = 5, figdir = "BrittleDuctile2D")
    results = NamedTuple[]
    for with_cap in (true, false)
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !(JustRelax.MPI.Initialized()))...)
        println(with_cap ? "with tensile cap" : "control: no tensile cap")
        push!(results, run_case(igg; with_cap, nx, ny, nsteps, figdir))
        finalize_global_grid(; finalize_MPI = false)
    end

    report(results)
    return results
end

main()
