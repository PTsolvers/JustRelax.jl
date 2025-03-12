using CUDA
using JustRelax, JustRelax.JustRelax3D, JustRelax.DataIO

const backend_JR = CUDABackend
using Printf, GeoParams, GLMakie, CellArrays

using ParallelStencil
@init_parallel_stencil(CUDA, Float64, 3)

using JustPIC, JustPIC._3D
const backend = CUDABackend

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phases, particles, radius)
    ni = size(phases)
    origin = 0.5, 0.5, 0.5

    @parallel_indices (I...) function init_phases!(phases, index, xc, yc, zc, o_x, o_y, o_z)

        for ip in cellaxes(xc)
            (@index index[ip, I...]) || continue

            x = @index xc[ip, I...]
            y = @index yc[ip, I...]
            z = @index zc[ip, I...]

            if ((x - o_x)^2 + (y - o_y)^2 + (z - o_z)^2) > radius^2
                @index phases[ip, I...] = 1.0
            else
                @index phases[ip, I...] = 2.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.index, particles.coords..., origin...)
    return nothing
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, nz = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    lx = ly = lz = 1.0e0             # domain length in y
    ni = nx, ny, nz      # number of cells
    li = lx, ly, lz      # domain length in x- and y-
    di = @. li / ni      # grid step in x- and -y
    origin = 0.0, 0.0, 0.0   # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells    dt          = Inf

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30            # friction angle
    C = τ_y           # Cohesion
    η0 = 1.0           # viscosity
    G0 = 1.0           # elastic shear modulus
    Gi = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
    # Gi          = G0            # elastic shear modulus perturbation
    εbg = 1.0           # background strain-rate
    η_reg = 1.25e-2       # regularisation "viscosity"
    dt = η0 / G0 / 4.0     # assumes Maxwell time of 4
    dt /= 2
    el_bg = ConstantElasticity(; G = G0, ν = 0.5)
    el_inc = ConstantElasticity(; G = Gi, ν = 0.5)
    visc = LinearViscous(; η = η0)
    visc_inc = LinearViscous(; η = η0 / 10)
    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C,
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0.0,
    )
    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # Initialize phase ratios -------------------------------
    nxcell, max_xcell, min_xcell = 125, 150, 75
    particles = init_particles(backend, nxcell, max_xcell, min_xcell, xvi, di, ni)
    radius = 0.1
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    pPhases, = init_cell_arrays(particles, Val(1))
    # Assign particles phases anomaly
    init_phases!(pPhases, particles, radius)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-5, CFL = 0.5 / √3.1)
    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = Inf)
    # Rheology
    cutoff_visc = -Inf, Inf
    compute_viscosity!(stokes, phase_ratios, args, rheology, cutoff_visc)

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true, back = true, front = true),
        no_slip = (left = false, right = false, top = false, bot = false, back = false, front = false),
    )

    stokes.V.Vx .= PTArray(backend_JR)([ x * εbg for x in xvi[1], _ in 1:(ny + 2), _ in 1:(nz + 2)])
    stokes.V.Vz .= PTArray(backend_JR)([-z * εbg for _ in 1:(nx + 2), _ in 1:(ny + 2), z in xvi[3]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)
    Vz_v = @zeros(ni .+ 1...)

    # Time loop
    t, it = 0.0, 0
    tmax = 4
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]

    while t < tmax
        # Stokes solver ----------------
        solve!(
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
            kwargs = (;
                iterMax = 75.0e3,
                nout = 1.0e3,
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

        velocity2vertex!(Vx_v, Vy_v, Vz_v, @velocity(stokes)...)
        data_v = (;
            τII = Array(stokes.τ.II),
            εII = Array(stokes.ε.II),
            εII_pl = Array(stokes.ε_pl.II),
        )
        data_c = (;
            P = Array(stokes.P),
            η = Array(stokes.viscosity.η_vep),
        )
        velocity_v = (
            Array(Vx_v),
            Array(Vy_v),
            Array(Vz_v),
        )
        save_vtk(
            joinpath(figdir, "vtk_" * lpad("$it", 6, "0")),
            xvi,
            xci,
            data_v,
            data_c,
            velocity_v
        )

        # visualisation
        jslice = ni[2] >>> 1
        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = "τII")
        ax2 = Axis(fig[2, 1], aspect = 1, title = "η_vep")
        ax3 = Axis(fig[1, 2], aspect = 1, title = "log10(εxy)")
        ax4 = Axis(fig[2, 2], aspect = 1)
        heatmap!(ax1, xci[1], xci[3], Array(stokes.τ.II[:, jslice, :]), colormap = :batlow)
        heatmap!(ax2, xci[1], xci[3], Array(stokes.viscosity.η_vep[:, jslice, :]), colormap = :batlow)
        heatmap!(ax3, xci[1], xci[3], Array(stokes.ε_pl.II[:, jslice, :]), colormap = :batlow)
        lines!(ax4, ttot, τII, color = :black)
        lines!(ax4, ttot, sol, color = :red)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        for ax in (ax1, ax2, ax3)
            xlims!(ax, (0, 1))
            ylims!(ax, (0, 1))
        end
        save(joinpath(figdir, "$(it).png"), fig)

    end

    return nothing
end

n = 64
nx = ny = nz = n
figdir = "ShearBand3D"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, nz; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny, nz = nz);
