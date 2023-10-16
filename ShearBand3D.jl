using CUDA
CUDA.allowscalar(false)

using Printf, GeoParams, GLMakie, CellArrays, CSV, DataFrames
using JustRelax, JustRelax.DataIO

# setup ParallelStencil.jl environment
model  = PS_Setup(:gpu, Float64, 3)
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
    η_reg       = 8e-3          # regularisation "viscosity"
    η_reg       = 1.25e-2       # regularisation "viscosity"
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
            # CompositeRheology = CompositeRheology((visc, el_bg)),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity        = el_bg,

        ),
        # High density phase
        SetMaterialParams(;
            Density           = ConstantDensity(; ρ = 0.0),
            Gravity           = ConstantGravity(; g = 0.0),
            # CompositeRheology = CompositeRheology((visc, el_inc)),
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
    ρg           = @zeros(ni...), @zeros(ni...), @zeros(ni...)
    args         = (; T = @zeros(ni...), P = stokes.P, dt = dt)
    # Rheology
    η            = @ones(ni...)
    η_vep        = similar(η) # effective visco-elasto-plastic viscosity
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, @strain(stokes)..., args, rheology, (-Inf, Inf)
    )

    # Boundary conditions
    flow_bcs     = FlowBoundaryConditions(;
        free_slip   = (left = true , right = true , top = true , bot = true , back = true , front = true),
        no_slip     = (left = false, right = false, top = false, bot = false, back = false, front = false),
        periodicity = (left = false, right = false, top = false, bot = false, back = false, front = false),
    )
    stokes.V.Vx .= PTArray([ x*εbg   for x in xvi[1], _ in 1:ny+2, _ in 1:nz+2])
    # stokes.V.Vy .= PTArray([ y*εbg   for _ in 1:nx+2, y in xvi[2], _ in 1:nz+2])
    stokes.V.Vz .= PTArray([-z*εbg   for _ in 1:nx+2, _ in 1:nx+2, z in xvi[3]])
    
    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    tmax  = 3
    τII   = Float64[]
    sol   = Float64[]
    ttot  = Float64[]

    while t < tmax
        # Stokes solver ----------------
        λ = solve!(
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
        @parallel (JustRelax.@idx ni) JustRelax.Stokes3D.tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        push!(τII, maximum(stokes.τ.xx))

        it += 1
        t  += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

        data_v = (;)
        data_c = (; 
            τII = Array(stokes.τ.II),
            εII = Array(stokes.ε.II),
            logεII = Array(log10.(stokes.ε.II)),
            η   = Array(η_vep),
        )
        save_vtk(
            joinpath(figdir, "vtk_" * lpad("$it", 6, "0")),
            xvi,
            xci, 
            data_v, 
            data_c
        )
        heatmap(Array(stokes.τ.II[:,32,:]))

        # # visualisation
        # fig     = Figure(resolution = (1600, 1600), title = "t = $t")
        # ax1     = Axis3(fig[1,1], aspect =  (1, 1, 1), title = "τII")
        # ax2     = Axis3(fig[2,1], aspect =  (1, 1, 1), title = "η_vep")
        # ax3     = Axis3(fig[1,2], aspect =  (1, 1, 1), title = "log10(εxy)")
        # ax4     = Axis(fig[2,2], aspect = 1)
        # volume!(ax1, xci..., Array(stokes.τ.II) , colormap=:batlow)
        # volume!(ax2, xci..., Array(η_vep) , colormap=:batlow)
        # volume!(ax3, xci..., Array(log10.(stokes.ε.xy)) , colormap=:batlow)
        # lines!(ax4, ttot, τII, color = :black) 
        # lines!(ax4, ttot, sol, color = :red) 
        # hidexdecorations!(ax1)
        # hidexdecorations!(ax3)
        # for ax in (ax1, ax2, ax3)
        #     xlims!(ax, (0, 1))
        #     ylims!(ax, (0, 1))
        #     zlims!(ax, (0, 1))
        # end
        # save(joinpath(figdir, "$(it).png"), fig)
        # fig
    end

    df = DataFrame(
        t = ttot,
        τII = τII,
        sol = sol,
        # iterations = iterations
    )

    CSV.write(joinpath(figdir, "data_$(nx).csv"), df)

    return nothing
end

# N           = 64
N            = parse(Int, ARGS[1]) 
n            = N + 2
nx = ny = nz = n - 2
figdir       = "ShearBand3D/ShearBand_vertex_hiReg_$n"
# figdir       = "ShearBand3D/ShearBand_vertex_hiReg_$n"
igg          = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny, nz = nz);
