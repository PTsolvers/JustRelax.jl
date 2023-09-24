# using CUDA
# CUDA.allowscalar(false)

using Printf, GeoParams, GLMakie, CellArrays, CSV, DataFrames
using JustRelax, JustRelax.DataIO

# setup ParallelStencil.jl environment
model  = PS_Setup(:cpu, Float64, 3)
environment!(model)

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, radius)
    ni      = size(phase_ratios.center)
    origin  = 0.5, 0.5, 0.5
    
    @parallel_indices (i, j, k) function init_phases!(phases, xc, yc, zc, o_x, o_y, o_z)
        x, y, z = xc[i], yc[j], zc[k]
        if ((x-o_x)^2 + (y-o_y)^2 + (z-o_z)^2) > radius
            JustRelax.@cell phases[1, i, j, k] = 1.0
            JustRelax.@cell phases[2, i, j, k] = 0.0
        
        else
            JustRelax.@cell phases[1, i, j, k] = 0.0
            JustRelax.@cell phases[2, i, j, k] = 1.0

        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phase_ratios.center, xci..., origin...)
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, nz=64, figdir="model_figs")

    # Physical domain ------------------------------------
    lx =ly = lz = 1e0             # domain length in y
    ni          = nx, ny, nz      # number of cells
    li          = lx, ly, lz      # domain length in x- and y-
    di          = @. li / ni      # grid step in x- and -y
    origin      = 0.0, 0.0, 0.0   # origin coordinates
    xci, xvi    = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    dt          = Inf 

    # Physical properties using GeoParams ----------------
    τ_y         = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ           = 30            # friction angle
    C           = τ_y           # Cohesion
    η0          = 1.0           # viscosity
    G0          = 1.0           # elastic shear modulus
    Gi          = G0/(6.0-4.0)  # elastic shear modulus perturbation
    εbg         = 1.0           # background strain-rate
    η_reg       = 8e-3 * 0          # regularisation "viscosity"
    # η_reg   = 1.25e-2       # regularisation "viscosity"
    dt          = η0/G0/4.0     # assumes Maxwell time of 4
    el_bg       = ConstantElasticity(; G=G0, ν=0.5)
    el_inc      = ConstantElasticity(; G=Gi, ν=0.5)
    visc        = LinearViscous(; η=η0) 
    pl          = DruckerPrager_regularised(;  # non-regularized plasticity
        C    = C,
        ϕ    = ϕ, 
        η_vp = η_reg,
        Ψ    = 0.0,
    ) 
    rheology    = (
        # Low density phase
        SetMaterialParams(;
            Phase             = 1,
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity        = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity        = el_inc,
        ),
    )

    # Initialize phase ratios -------------------------------
    radius       = 0.01
    phase_ratios = PhaseRatio(ni, length(rheology))
    init_phases!(phase_ratios, xci, radius)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes       = StokesArrays(ni, ViscoElastic)
    pt_stokes    = PTStokesCoeffs(li, di; ϵ = 1e-4,  CFL = 0.75 / √3.1)
    # Buoyancy forces
    ρg           = @zeros(ni...), @zeros(ni...)
    args         = (; T = @zeros(ni...), P = stokes.P, dt = dt)
    # Rheology
    η            = @ones(ni...)
    η_vep        = similar(η) # effective visco-elasto-plastic viscosity
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
    )

    # Boundary conditions
    flow_bcs      = FlowBoundaryConditions(; 
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip   = (left = false, right = false, top = false, bot=false),
    )
    stokes.V.Vx  .= PTArray([ x*εbg for x in xvi[1], _ in 1:ny+2, _ in 1:nz+2])
    stokes.V.Vz  .= PTArray([-z*εbg for _ in 1:nx+2, _ in 1:nx+2, z in xvi[3]])
    
    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it      = 0.0, 0
    tmax       = 3
    τII        = Float64[]
    sol        = Float64[]
    ttot       = Float64[]
    iterations = Int64[]
    # while it < 10
    while t < tmax

        # Stokes solver ----------------
        λ, iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            phase_ratios,
            rheology,
            args,
            dt,
            igg;
            iterMax          = 150e3,
            nout             = 1e3,
            viscosity_cutoff = (-Inf, Inf)
        )
        @show maximum(stokes.τ.II)
        push!(τII, maximum(stokes.τ.xx))

        it += 1
        t  += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)
        push!(iterations, iters)

        println("it = $it; t = $t \n")

        # visualisation
        th    = 0:pi/50:3*pi;
        xunit = @. 0.1 * cos(th) + 0.5;
        yunit = @. 0.1 * sin(th) + 0.5;

        i_slice = div(nx, 2) |> Int
        fig     = Figure(resolution = (1600, 1600), title = "t = $t")
        ax1     = Axis(fig[1,1], aspect = 1, title = "τII")
        ax2     = Axis(fig[2,1], aspect = 1, title = "η_vep")
        ax3     = Axis(fig[1,2], aspect = 1, title = "λ")
        ax4     = Axis(fig[2,2], aspect = 1)
        heatmap!(ax1, xci..., Array(stokes.τ.II) , colormap=:batlow)
        heatmap!(ax2, xci..., Array(η_vep) , colormap=:batlow)
        heatmap!(ax3, xci..., Array(λ .!= 0) , colormap=:batlow)
        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII, color = :black) 
        lines!(ax4, ttot, sol, color = :red) 
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
        # display(fig)
        # fig

    end

    df = DataFrame(
        t = ttot,
        τII = τII,
        sol = sol,
        iterations = iterations
    )

    CSV.write(joinpath(figdir, "data_$(nx).csv"), df)

    return nothing
end

n            = 64 + 2
nx = ny = nz = n - 2
figdir       = "ShearBand_vertex_DP_$n"
igg          = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)
else
    igg
end
# main(igg; figdir = figdir, nx = nx, ny = ny, nz = nz);
