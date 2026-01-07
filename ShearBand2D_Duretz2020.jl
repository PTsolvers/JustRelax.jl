# Setup from Duretz et al 2020 https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2019GL086027

# using CUDA

using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using ParallelStencil
# @init_parallel_stencil(CUDA, Float64, 2)
@init_parallel_stencil(Threads, Float64, 2)

# const backend = CUDABackend
const backend = CPUBackend

using JustPIC, JustPIC._2D
import JustPIC._2D.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPUBackend
# const backend_JP = CUDABackend

# HELPER FUNCTIONS ----------------------------------- ----------------------------
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

const year = 3.1536e7

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx = 64, ny = 64, figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 30e3            # domain length in y
    lx = 100e3           # domain length in x
    ni = nx, ny          # number of cells
    li = lx, ly          # domain length in x- and y-
    di = @. li / ni      # grid step in x- and -y
    origin = -lx/2, -ly  # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    # Physical properties using GeoParams ----------------
    dt       = 10e3 * year     # assumes Maxwell time of 4
    el_crust = ConstantElasticity(; G = 1e10, ν = 0.499)
    el_inc   = ConstantElasticity(; G = 1e10, ν = 0.499)
    
    disl_crust = DislocationCreep(A = 3.1623e-26, n = 3.3, E = 186.5e3, V = 0e0, r = 0.0, R = 8.3145)
    disl_incl  = DislocationCreep(A = 1e-20,      n = 1,   E = 0e0,     V = 0e0, r = 0.0, R = 8.3145)
    
    pl_crust = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = 50e6,
        ϕ = atand(0.6),
        η_vp = 1e21,
        Ψ = 0
    )
        
    pl_incl  = DruckerPrager_regularised(;
        # non-regularized plasticity
        C = 0.1e6,
        ϕ = 0e0,
        η_vp = 1e21,
        Ψ = 0
    )

    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 2.7e3),
            Gravity = ConstantGravity(; g = 10.0),
            CompositeRheology = CompositeRheology((disl_crust, el_crust, pl_crust)),
            Elasticity = el_crust,

        ),
        # High density phase
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 2.7e3),
            Gravity = ConstantGravity(; g = 10.0),
            CompositeRheology = CompositeRheology((disl_incl, el_inc, pl_incl)),
            Elasticity = el_inc,
        ),
    )

    # perturbation array for the cohesion
    perturbation_C = @zeros(ni...)

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 2e3
    origin = 0e0, -ly/2
    circle = GGU.Circle(origin, radius)
    init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-2, ϵ_rel = 1.0e-9, CFL = 0.95 / √2.1)

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    T_1D = LinRange(466, 20, ny + 1)
    thermal.T[2:end-1, :] .= PTArray(backend)([ T_1D[j] for i in 1:(nx + 1), j in 1:(ny + 1)])
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    stokes.P        .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (1e18, 1e23)
    )
    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    εbg = 1e-15
    stokes.V.Vx .= PTArray(backend)([         x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-(y + ly/2) * εbg for _ in 1:(nx + 2), y in xvi[2]])
    # @views stokes.V.Vx[2:end-1, :] .= 0
    # @views stokes.V.Vy[:, 2:end-1] .= 0
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, di, dt; ϵ=1e-3)
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    # IO -------------------------------------------------
    vtk_dir      = joinpath(figdir, "vtk")
    take(figdir)
    take(vtk_dir)

    # Time loop
    t, it = 0.0, 0
    tmax = 5
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]
    # while t < tmax
    for _ in 1:100

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
                iterMax = 150.0e3,
                nout = 2.0e3,
                λ_relaxation = 1,
                viscosity_relaxation = 1,
                viscosity_cutoff = (1e18, 1e23),
            )
        )
        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)

        it += 1
        t += dt

        println("it = $it; t = $t \n")

        velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
        data_v = (;
            T = Array(thermal.T[2:(end - 1), :]),
            τxy = Array(stokes.τ.xy),
            εxy = Array(stokes.ε.xy),
            Vx = Array(Vx_v),
            Vy = Array(Vy_v),
        )
        data_c = (;
            P = Array(stokes.P),
            τxx = Array(stokes.τ.xx),
            τyy = Array(stokes.τ.yy),
            εxx = Array(stokes.ε.xx),
            εyy = Array(stokes.ε.yy),
            η = Array(stokes.viscosity.η_vep),
        )
        velocity_v = (
            Array(Vx_v),
            Array(Vy_v),
        )
        save_vtk(
            joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
            xvi,
            xci,
            data_v,
            data_c,
            velocity_v,
            t = t
        )

        # visualisation
        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax3 = Axis(fig[1, 2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 2], aspect = 1)
        heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :lipari)
        # h1 = heatmap!(ax1, xci..., Array((stokes.viscosity.η_vep)) , colormap=:batlow)
        heatmap!(ax2, xci..., Array(stokes.ε_pl.II), colormap = :lipari)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :lipari)
        heatmap!(ax4, xci..., Array(log10.(stokes.viscosity.η_vep)), colormap = :lipari)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        # Colorbar(fig[1, 2], h1)
        fig
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

# n  = 2
# nx = n * 201
# ny = n * 61
n  = 60
nx = n * 3 
ny = n
figdir = "ShearBands2D_Duretz2020_APT"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
@time main(igg; figdir = figdir, nx = nx, ny = ny);

# 7.174e-07, Rp=1.402e-08] 
# itPH = 98 iter = 025101 iter/nx = 196, err = 9.671e-07 norm[Rx=9.671e-07, Ry=6.793e-07, Rp=1.327e-08] 
# it = 15; t = 3.75 

# 170.963150 seconds (110.81 M allocations: 39.591 GiB, 4.11% gc time, 0.00% compilation time: 100% of which was recompilation)