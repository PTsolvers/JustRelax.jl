using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend

# HELPER FUNCTIONS ---------------------------------------------------------------
function init_phases!(pPhases, particles, xvi, width, height, fault_thickness, buffer, fault_dip)
    ni      = size(particles.index)
    lx      = xvi[1][end] - xvi[1][1]

    fault   = GGU.Rectangle((0e0, height / 2), Inf, fault_thickness; θ = deg2rad(fault_dip))
    sample  = GGU.Rectangle((0e0, height / 2), width, height)
    buffer1 = GGU.Rectangle((xvi[1][1]   + buffer/2, height / 2), buffer, height)
    buffer2 = GGU.Rectangle((xvi[1][end] - buffer/2, height / 2), buffer, height)

    @parallel_indices (i, j) function init_phases!(pPhases, px, py, index, sample, fault, buffer1, buffer2)
        
        for ip in cellaxes(index)
            
            x        = @index px[ip, i, j]
            y        = @index py[ip, i, j]
            p        = GGU.Point(x, y)
            infault  = GGU.inside(p, fault)
            insample = GGU.inside(p, sample)
            inbuffer = GGU.inside(p, buffer1) || GGU.inside(p, buffer2)

            if insample
                @index pPhases[ip, i, j] = 1e0
            end
            if infault
                @index pPhases[ip, i, j] = 2e0
            end
            if inbuffer
                @index pPhases[ip, i, j] = 3e0
            end

        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(pPhases, particles.coords..., particles.index, sample, fault, buffer1, buffer2)

    return nothing
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # domain
    height          = 3e0
    width           = height / 3
    buffer          = 0.5
    fault_thickness = 0.2
    fault_dip       = 30
    # Physical domain ------------------------------------
    ly               = height              # domain length in y
    lx               = width + 2 * buffer  # domain length in x
    li               = lx, ly       # domain length in x- and y-
    # aspect_ratio     = ceil(Int, ly / lx)
    # aspect_ratio     = 1
    # ny               = nx * aspect_ratio # number of cells
    ni               = nx, ny       # number of cells
    di               = @. li / ni   # grid step in x- and -y
    origin           = -lx / 2, 0.0 # origin coordinates
    grid             = Geometry(ni, li; origin = origin)
    (; xci, xvi)     = grid # nodes at the center and vertices of the cells

    # Physical properties using GeoParams ----------------
    τ_y         = 1.6               # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ           = 30                # friction angle
    C           = τ_y               # Cohesion
    η0          = 1.0               # viscosity
    G0          = 1.0               # elastic shear modulus
    Gi          = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
    εbg         = 5.0               # background strain-rate
    η_reg       = 8.0e-3            # regularisation "viscosity"
    dt          = η0 / G0 / 4.0     # assumes Maxwell time of 4
    el_bg       = ConstantElasticity(; G = G0,      ν = 0.3)
    el_inc      = ConstantElasticity(; G = G0 / 10, ν = 0.3)
    visc_host   = LinearViscous(; η = η0)
    visc_gauge  = LinearViscous(; η = η0 / 5)
    visc_buffer = LinearViscous(; η = η0 / 1e3)
    # soft_C  = LinearSoftening((C/2, C), (0e0, 2e0))
    soft_C = NonLinearSoftening(; ξ₀ = C, Δ = C / 2)

    pl_host     = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C / cosd(ϕ),
        ϕ = 35,
        η_vp = η_reg,
        softening_C = soft_C,
        Ψ = 0
    )
    pl_gauge    = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C / cosd(ϕ) / 2,
        ϕ = 30,
        η_vp = η_reg,
        softening_C = soft_C,
        Ψ = 0
    )
    rheology = (
        # Host
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_host, el_bg, pl_host)),
            Elasticity = el_bg,
        ),
        # Fault
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_gauge, el_inc, pl_gauge)),
            Elasticity = el_inc,
        ),
        # Buffer
        SetMaterialParams(;
            Phase = 3,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc_buffer, el_inc, )),
            Elasticity = el_inc,
        ),
    )

    # Initialize particles -------------------------------
    nxcell    = 40
    max_xcell = 60
    min_xcell = 20
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase
    pPhases, = init_cell_arrays(particles, Val(1))

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, particles, xvi, width, height, fault_thickness, buffer, fault_dip)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-1, CFL = 0.95 / √2.1)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    take(figdir)

    # Time loop
    t, it = 0.0, 0
    tmax = 5
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]
    # while t < tmax
    for _ in 1:100

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            kwargs = (
                verbose = false,
                iterMax = 50.0e3,
                nout = 1.0e3,
                viscosity_cutoff = (-Inf, Inf),
            )
        )
        # dt = compute_dt(stokes, di) * 0.9

        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)
        push!(τII, stokes.τ.yy[n>>>1, end])

        # Advection --------------------
        # advect particles in space
        # advection_MQS!(particles, RungeKutta4(), @velocity(stokes), grid_vxi, dt)
        # # advect particles in memory
        # move_particles!(particles, xvi, ())
        # # check if we need to inject particles
        # # need stresses on the vertices for injection purposes
        # # center2vertex!(τxx_v, stokes.τ.xx)
        # # center2vertex!(τyy_v, stokes.τ.yy)
        # inject_particles_phase!(
        #     particles,
        #     pPhases,
        #     (),
        #     (),
        #     xvi
        # )

        # # update phase ratios
        # update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        it += 1
        t += dt

        push!(ttot, t)
        println("extrema ε_pl = $(extrema(stokes.ε_pl.II))")
        println("it = $it; t = $t \n")

        # visualisation
        ar = lx / ly
        # th = 0:(pi / 50):(3 * pi)
        # xunit = @. radius * cos(th) + 0.5
        # yunit = @. radius * sin(th) + 0.5
        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = ar, title = L"\tau_{II}", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = ar, title = L"\varepsilon_{II}^{pl}", titlesize = 35)
        # ax2 = Axis(fig[2, 1], aspect = ar, title = L"E_{II}^{pl}", titlesize = 35)
        ax3 = Axis(fig[1, 2], aspect = ar, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 2], aspect = ar, title = L"\tau_{II}")
        heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :batlow)
        # heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)
        heatmap!(ax2, xci..., Array(max.(0,log10.(stokes.ε_pl.II))), colormap = :batlow)
        # heatmap!(ax2, xci..., Array(stokes.EII_pl), colormap = :batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
        # lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        scatterlines!(ax4, ttot, τII, color = :black)
        # lines!(ax4, ttot, sol, color = :red)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

n  = 100
nx = n
ny = n * 2
figdir = "ShearBands2D"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);
