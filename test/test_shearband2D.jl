push!(LOAD_PATH, "..")

using Test, Suppressor
using GeoParams, CellArrays
using JustRelax, JustRelax.DataIO
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

# setup ParallelStencil.jl environment
model  = PS_Setup(:Threads, Float64, 2)
environment!(model)

# HELPER FUNCTIONS ----------------------------------- ----------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, radius)
    ni      = size(phase_ratios.center)
    origin  = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y, radius)
        x, y = xc[i], yc[j]
        if ((x-o_x)^2 + (y-o_y)^2) > radius^2
            JustRelax.@cell phases[1, i, j] = 1.0
            JustRelax.@cell phases[2, i, j] = 0.0

        else
            JustRelax.@cell phases[1, i, j] = 0.0
            JustRelax.@cell phases[2, i, j] = 1.0
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phase_ratios.center, xci..., origin..., radius)
end

# MAIN SCRIPT --------------------------------------------------------------------
function ShearBand2D()
    n      = 32
    nx     = n
    ny     = n
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg    = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    # Physical domain ------------------------------------
    ly           = 1e0          # domain length in y
    lx           = ly           # domain length in x
    ni           = nx, ny       # number of cells
    li           = lx, ly       # domain length in x- and y-
    di           = @. li / ni   # grid step in x- and -y
    origin       = 0.0, 0.0     # origin coordinates
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    dt           = Inf

    # Physical properties using GeoParams ----------------
    τ_y     = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ       = 30            # friction angle
    C       = τ_y           # Cohesion
    η0      = 1.0           # viscosity
    G0      = 1.0           # elastic shear modulus
    Gi      = G0/(6.0-4.0)  # elastic shear modulus perturbation
    εbg     = 1.0           # background strain-rate
    η_reg   = 8e-3          # regularisation "viscosity"
    dt      = η0/G0/4.0     # assumes Maxwell time of 4
    el_bg   = ConstantElasticity(; G=G0, Kb=4)
    el_inc  = ConstantElasticity(; G=Gi, Kb=4)
    visc    = LinearViscous(; η=η0)
    pl      = DruckerPrager_regularised(;  # non-regularized plasticity
        C    = C,
        ϕ    = ϕ,
        η_vp = η_reg,
        Ψ    = 0
    )
    rheology = (
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
    radius       = 0.1
    phase_ratios = PhaseRatio(ni, length(rheology))
    init_phases!(phase_ratios, xci, radius)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL = 0.75 / √2.1)

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    args      = (; T = @zeros(ni...), P = stokes.P, dt = dt)

    # Rheology
    η         = @ones(ni...)
    η_vep     = similar(η) # effective visco-elasto-plastic viscosity
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, args, rheology, (-Inf, Inf)
    )

    # Boundary conditions
    flow_bcs     = FlowBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip   = (left = false, right = false, top = false, bot=false),
    )
    stokes.V.Vx .= PTArray([ x*εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(stokes.V.Vx, stokes.V.Vy)

    # Time loop
    t, it      = 0.0, 0
    tmax       = 3.5
    τII        = Float64[]
    sol        = Float64[]
    ttot       = Float64[]
    local iters, τII, sol

    while t < tmax

        # Stokes solver ----------------
        iters = solve!(
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
            verbose          = false,
            iterMax          = 50e3,
            nout             = 1e2,
            viscosity_cutoff = (-Inf, Inf)
        )
        @parallel (JustRelax.@idx ni) JustRelax.Stokes2D.tensor_invariant!(stokes.ε.II, @strain(stokes)...)
        push!(τII, maximum(stokes.τ.xx))

        @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
        @parallel (@idx ni) multi_copy!(
            @tensor_center(stokes.τ_o), @tensor_center(stokes.τ)
        )

        it += 1
        t  += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)

        println("it = $it; t = $t \n")

    end

    finalize_global_grid(; finalize_MPI = true)

    return iters, τII, sol


end

@testset "ShearBand2D" begin
    @suppress begin
        iters, τII, sol = ShearBand2D()
        @test passed = iters.err_evo1[end] < 1e-6
        @test τII[end] ≈ 1.4170633311147924 rtol = 1e-6
        @test sol[end] ≈ 1.939605233155363 rtol = 1e-6
    end

end
