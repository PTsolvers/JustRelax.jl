using GeoParams
using JustRelax, JustRelax.JustRelax2D, GLMakie
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend_JR = CPUBackend

using JustPIC, JustPIC._2D

const backend = JustPIC.CPUBackend

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, xvi, radius)
    ni = size(phase_ratios.center)
    origin = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y, radius)
        x, y = xc[i], yc[j]
        if ((x - o_x)^2 + (y - o_y)^2) > radius^2
            @index phases[1, i, j] = 1.0
            @index phases[2, i, j] = 0.0

        else
            @index phases[1, i, j] = 0.0
            @index phases[2, i, j] = 1.0
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phase_ratios.center, xci..., origin..., radius)
    @parallel (@idx ni .+ 1) init_phases!(phase_ratios.vertex, xvi..., origin..., radius)
    return nothing
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 1.0e0          # domain length in y
    lx = ly           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    origin = 0.0, 0.0     # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt = Inf

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30            # friction angle
    C = τ_y           # Cohesion
    η0 = 1.0           # viscosity
    G0 = 1.0           # elastic shear modulus
    Gi = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
    εbg = 1.0           # background strain-rate
    η_reg = 8.0e-3          # regularisation "viscosity"
    dt = η0 / G0 / 4.0     # assumes Maxwell time of 4
    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = C / cosd(ϕ),
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0,
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
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # Initialize phase ratios -------------------------------
    radius = 0.1
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(phase_ratios, xci, xvi, radius)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ = 1.0e-6, CFL = 0.75 / √2.1)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni...), P = stokes.P, dt = dt)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend_JR)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend_JR)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    take(figdir)
    # ----------------------------------------------------

    # global array
    nx_v = (nx - 2) * igg.dims[1]
    ny_v = (ny - 2) * igg.dims[2]
    τII_v = zeros(nx_v, ny_v)
    η_vep_v = zeros(nx_v, ny_v)
    εII_v = zeros(nx_v, ny_v)
    τII_nohalo = zeros(nx - 2, ny - 2)
    η_vep_nohalo = zeros(nx - 2, ny - 2)
    εII_nohalo = zeros(nx - 2, ny - 2)
    Vxv_v = zeros(nx_v, ny_v)
    Vyv_v = zeros(nx_v, ny_v)
    Vx_nohalo = zeros(nx - 2, ny - 2)
    Vy_nohalo = zeros(nx - 2, ny - 2)
    xci_v = LinRange(0, 1, nx_v), LinRange(0, 1, ny_v)

    local Vx, Vy
    Vx = @zeros(ni...)
    Vy = @zeros(ni...)

    # Time loop
    t, it = 0.0, 0
    tmax = 3.5
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
            kwargs = (
                verbose = false,
                iterMax = 50.0e3,
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

        igg.me == 0 && println("it = $it; t = $t \n")

        # visualisation
        th = 0:(pi / 50):(3 * pi)
        xunit = @. radius * cos(th) + 0.5
        yunit = @. radius * sin(th) + 0.5

        # Gather MPI arrays
        velocity2center!(Vx, Vy, @velocity(stokes)...)
        @views Vx_nohalo .= Array(Vx[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views Vy_nohalo .= Array(Vy[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        gather!(Vx_nohalo, Vxv_v)
        gather!(Vy_nohalo, Vyv_v)

        @views τII_nohalo .= Array(stokes.τ.II[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views η_vep_nohalo .= Array(stokes.viscosity.η_vep[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        @views εII_nohalo .= Array(stokes.ε.II[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        gather!(τII_nohalo, τII_v)
        gather!(η_vep_nohalo, η_vep_v)
        gather!(εII_nohalo, εII_v)

        if igg.me == 0
            fig = Figure(size = (1600, 1600), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = 1, title = "τII")
            ax2 = Axis(fig[2, 1], aspect = 1, title = "η_vep")
            ax3 = Axis(fig[1, 3], aspect = 1, title = "log10(εII)")
            ax4 = Axis(fig[2, 3], aspect = 1)
            heatmap!(ax1, xci_v..., Array(τII_v), colormap = :batlow)
            heatmap!(ax2, xci_v..., Array(log10.(η_vep_v)), colormap = :batlow)
            heatmap!(ax3, xci_v..., Array(log10.(εII_v)), colormap = :batlow)
            lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
            lines!(ax4, ttot, τII, color = :black)
            lines!(ax4, ttot, sol, color = :red)
            hidexdecorations!(ax1)
            hidexdecorations!(ax3)
            save(joinpath(figdir, "MPI_$(it).png"), fig)

        end
    end

    return nothing

end

N = 30
n = N
nx = n * 2  # if only 2 CPU/GPU are used nx = 67 - 2 with N =128
ny = n * 2
figdir = "ShearBands2D_MPI"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true, select_device = false)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);
