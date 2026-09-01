# Kelvin-Helmholtz-style billows in a 3D box that is periodic in x and y, driven by two
# plates sliding past each other in ±x, solved with the DYREL Stokes solver.
#
# A note on the physics. The textbook Kelvin-Helmholtz instability is *inertial*: it feeds on
# the kinetic energy of the shear and needs a finite Reynolds number. JustRelax solves the
# Stokes equations, where inertia is dropped entirely, so a pure velocity jump across a
# neutrally buoyant interface will never roll up here - it just tilts. What does produce
# billows in creeping flow is a buoyantly *unstable* interface embedded in a shear: the
# Rayleigh-Taylor instability supplies the amplitude, and the background shear tilts and wraps
# the growing cusps into cat's-eye rolls. That is the setup below, and it is a real geodynamic
# configuration - entrainment of a dense layer by a shearing mantle.
#
# The look of the result is set by the RT growth rate against the shear rate. For two layers
# of equal viscosity the growth rate is σ ≈ Δρ g / (2 η k) with k = 2π nkx / lx, i.e. ≈ 8e-15
# 1/s here, so `εbg` is chosen to be comparable:
#   * εbg >> σ -> the interface is smeared into flat stripes before it can grow
#   * εbg << σ -> plain vertical Rayleigh-Taylor fingers, no roll-up
# `nkx` sets how many billows fit along x.
#
# A word on the convergence test. DYREL anchors its per-component reference norms `errV0` on
# the residuals of the first Powell-Hestenes iteration, and tests
#     errV_rel = min(errV / errV0, errV)
# On this problem one component of that reference is always degenerate: the initial pressure
# balances ρg column by column, so the *vertical* residual starts at ~1e-12 and `errV/errV0`
# for it is meaningless (O(1e10)). The test then falls back to its second, *absolute* branch,
# which means ϵ is compared against a residual carrying SI units. It converges - ~500
# iterations per step at the default resolution - but bear that in mind before rescaling the
# problem or tightening ϵ.

const isCUDA = false

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend_JR = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences3D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

using JustPIC

const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU
end

using GeoParams, CairoMakie, Printf

const Myr = 1.0e6 * 3600 * 24 * 365.25

## MODEL PARAMETERS -----------------------------------------------------------------
const η0 = 1.0e21     # viscosity of both layers [Pa s]
const ρ_light = 3200.0     # lower layer [kg/m³]
const ρ_dense = 3300.0     # upper layer [kg/m³] -> Rayleigh-Taylor unstable
const g0 = 9.81       # gravity [m/s²]
const εbg = 5.0e-15    # background simple-shear rate [1/s], Vx = 2 εbg (z - lz/2)
const A_pert = 0.1        # interface perturbation amplitude, as a fraction of lz
const nkx = 4          # number of billows along x
const nky = 1          # spanwise (y) mode of the perturbation

## MODEL SETUP ----------------------------------------------------------------------

function init_rheologies()
    visc = LinearViscous(; η = η0)
    return (
        # Name = "Light" (lower layer)
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = ρ_light),
            CompositeRheology = CompositeRheology((visc,)),
            Gravity = ConstantGravity(; g = g0),
        ),
        # Name = "Dense" (upper layer)
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = ρ_dense),
            CompositeRheology = CompositeRheology((visc,)),
            Gravity = ConstantGravity(; g = g0),
        ),
    )
end

# Dense layer on top of a light one, separated by a corrugated interface. Both perturbation
# modes are whole numbers of wavelengths across the box, so the initial condition is
# consistent with the periodic boundaries.
function init_phases!(phases, particles, lx, ly, lz, A, kx, ky)
    ni = size(phases)

    @parallel_indices (I...) function _init_phases!(phases, px, py, pz, index, lx, ly, lz, A, kx, ky)
        for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            z = @index pz[ip, I...]

            # perturbed interface height
            z_interface = lz * 0.5 +
                A * lz * cospi(2 * kx * x / lx) * (1 + 0.5 * cospi(2 * ky * y / ly))

            @index phases[ip, I...] = z > z_interface ? 2.0 : 1.0
        end
        return nothing
    end

    @parallel (@idx ni) _init_phases!(
        phases, particles.coords..., particles.index, lx, ly, lz, A, kx, ky
    )
    return nothing
end

## MAIN SCRIPT ----------------------------------------------------------------------

function main3D(igg; nx = 64, ny = 16, nz = 16, figdir = "KelvinHelmholtz3D", do_vtk = false, nt = 200)

    # Physical domain ------------------------------------
    lz = 100.0e3              # domain height
    lx = 4 * lz               # long enough to fit `nkx` billows
    ly = lz                   # spanwise extent
    ni = nx, ny, nz           # number of cells
    li = lx, ly, lz           # domain length
    di = @. li / ni           # grid steps
    origin = 0.0, 0.0, 0.0    # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid       # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    rheology = init_rheologies()

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 30, 10
    particles = init_particles(backend_JP, nxcell, max_xcell, min_xcell, grid.xi_vel...)
    # only the material phase is carried by the particles
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases,)

    init_phases!(pPhases, particles, lx, ly, lz, A_pert, nkx, nky)
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    stokes = StokesArrays(backend_JR, ni)
    # purely viscous, so there is no elastic timestep in the rheology
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = Inf)
    viscosity_cutoff = (1.0e19, 1.0e23)
    # ----------------------------------------------------

    # Buoyancy forces and lithostatic pressure
    ρg = ntuple(_ -> @zeros(ni...), Val(3))
    compute_ρg!(ρg[end], phase_ratios, rheology, args)
    # `compute_lithostatic_pressure!` integrates every column independently, so this initial
    # pressure balances ρg column by column and the initial *vertical* momentum residual
    # cancels to machine zero (~1e-12). That is the reference norm DYREL anchors `errV0[3]`
    # on - see the note on the convergence test at the top of the file.
    compute_lithostatic_pressure!(stokes.P, ρg[end], di[end], igg)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    # Boundary conditions --------------------------------
    # Periodic in x and y. The top and bottom faces carry no condition at all, which is how
    # `VelocityBoundaryConditions` lets the caller prescribe a velocity by hand: `flow_bcs!`
    # leaves such faces untouched and `compute_V!` only writes the interior, so whatever we
    # store on those planes here survives every pseudo-transient iteration.
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
        no_slip = (left = false, right = false, top = false, bot = false, front = false, back = false),
        periodic = (left = true, right = true, top = false, bot = false, front = true, back = true),
    )

    # Simple shear: Vx varies linearly with z. `zVx` includes the two ghost planes that sit
    # half a cell outside the box, so evaluating the exact linear profile on them puts the
    # plate velocity ∓εbg·lz right on the walls (averaging a linear function is exact).
    # Vy and Vz stay zero: the plates drag in x only, and they are impermeable.
    zVx = grid.xi_vel[1][3]
    Vx_profile = [2 * εbg * (z - lz * 0.5) for _i in 1:(nx + 1), _j in 1:(ny + 2), z in zVx]
    copyto!(stokes.V.Vx, Vx_profile)
    fill!(stokes.V.Vy, 0.0)
    fill!(stokes.V.Vz, 0.0)
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
    # ----------------------------------------------------

    # IO -------------------------------------------------
    local vtk_pvd
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        vtk_pvd = joinpath(vtk_dir, "KelvinHelmholtz3D")
        take(vtk_dir)
        # `save_vtk` appends to the collection, so drop a stale one from a previous run -
        # otherwise ParaView replays the old time series alongside the new one.
        rm(vtk_pvd * ".pvd"; force = true)
    end
    take(figdir)
    # ----------------------------------------------------

    local Vx_v, Vy_v, Vz_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
        Vz_v = @zeros(ni .+ 1...)
    end

    # DyRel solver options. `dt` only sizes the internal coefficients here, since the rheology
    # is purely viscous - but it must stay finite: passing `Inf` to `solve_DYREL!` NaNs out on
    # the first inner iteration.
    dt = 0.25 * min(di...) / (2 * εbg * lz)
    dyrel = DYREL(
        backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6, CFL = 0.99
    )

    # Time loop
    t, it = 0.0, 0
    while it < nt

        # Stokes solver ----------------
        t_stokes = @elapsed begin
            out = solve_DYREL!(
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
                    iterMax = 50.0e3,
                    total_iterMax = 50.0e3,
                    nout = 1,
                    rel_drop = 1.0e-2,
                    viscosity_relaxation = 1,
                    linear_viscosity = true,
                    viscosity_cutoff = viscosity_cutoff,
                )
            )
        end

        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("   Time/iteration:  $(t_stokes / out.iter) s")

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.τ)
        dt = compute_dt(stokes, di) * 0.5
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        # advect particles in memory, wrapping them around the periodic x and y faces
        move_particles!(particles, particle_args; periodic_1 = true, periodic_2 = true)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (), ())
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, pPhases)
        # the interface moved, so the buoyancy has to follow
        compute_ρg!(ρg[end], phase_ratios, rheology, args)

        it += 1
        t += dt
        @printf("it = %d, t = %.3f Myrs, dt = %.3f kyrs\n", it, t / Myr, dt / (Myr * 1.0e-3))

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 5) == 0

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
                data_c = (;
                    ρ = Array(ρg[end] ./ g0),
                    P = Array(stokes.P),
                    τII = Array(stokes.τ.II),
                    εII = Array(stokes.ε.II),
                    phase = [argmax(p) for p in Array(phase_ratios.center)],
                )
                velocity_v = (Array(Vx_v), Array(Vy_v), Array(Vz_v))
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    (;),
                    data_c,
                    velocity_v,
                    t = t,
                    pvd = vtk_pvd,
                )
            end

            ρ = Array(ρg[end] ./ g0)
            x_km, y_km, z_km = xci[1] .* 1.0e-3, xci[2] .* 1.0e-3, xci[3] .* 1.0e-3
            slice_j = max(1, ny >>> 1)   # x-z slice through the middle
            slice_k = max(1, nz >>> 1)   # x-y slice at the interface
            ar = lx / lz

            fig = Figure(size = (1400, 1000))
            ax1 = Axis(fig[1, 1], aspect = ar, ylabel = "z [km]", title = "ρ [kg/m³]   (x-z slice, t = $(round(t / Myr, digits = 2)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, ylabel = "z [km]", title = "log10(εII [1/s])   (x-z slice)")
            ax3 = Axis(fig[3, 1], aspect = lx / ly, xlabel = "x [km]", ylabel = "y [km]", title = "ρ [kg/m³]   (x-y slice at the interface)")
            h1 = heatmap!(ax1, x_km, z_km, ρ[:, slice_j, :], colormap = :roma)
            h2 = heatmap!(ax2, x_km, z_km, log10.(Array(stokes.ε.II)[:, slice_j, :]), colormap = :batlow)
            h3 = heatmap!(ax3, x_km, y_km, ρ[:, :, slice_k], colormap = :roma)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[3, 2], h3)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------
    end

    return nothing
end

## SCRIPT ENTRY POINT ---------------------------------------------------------------

do_vtk = true  # set to true to generate VTK files for ParaView
n = 16         # cells across the shear layer; n = 32 gives noticeably crisper billows
nx, ny, nz = n * 4, n, n
# NOTE: the global grid is deliberately *not* declared periodic. The x/y periodicity of this
# model is carried entirely by the velocity boundary conditions (`periodic_boundary!` copies
# the ghost planes every iteration) and by `move_particles!(...; periodic_1, periodic_2)`.
# Passing `periodx`/`periody` to `init_global_grid` instead makes `Geometry` divide `li` by
# the overlap-reduced global cell count that ImplicitGlobalGrid reports, which yields the
# wrong `di` and coordinate ranges that run *backwards* - for this grid `xvi[1]` comes out as
# 386666 -> 13333 instead of 0 -> lx. Everything seeded from those coordinates (particles,
# phases, buoyancy) is then inconsistent with the finite differences, and the solver stalls on
# an irreducible residual instead of converging. The consequence is that this miniapp runs on
# a single rank only.
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end

figdir = "KelvinHelmholtz3D_DYREL_$n"
main3D(igg; figdir = figdir, nx = nx, ny = ny, nz = nz, do_vtk = do_vtk)
