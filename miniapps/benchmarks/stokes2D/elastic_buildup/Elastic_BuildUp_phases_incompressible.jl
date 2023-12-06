using CUDA
CUDA.allowscalar(false)

using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.DataIO

# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases on the particles
function init_phases!(phase_ratios, xci, radius)
    ni = size(phase_ratios.center)
    origin = 0.5, 0.5

    @parallel_indices (i, j) function init_phases!(phases, xc, yc, o_x, o_y)
        JustRelax.@cell phases[1, i, j] = 1.0

        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phase_ratios.center, xci..., origin...)
end

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, figdir="model_figs")

    # Physical domain ------------------------------------
    ly = 1e0          # domain length in y
    lx = ly           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / ni   # grid step in x- and -y
    origin = 0.0, 0.0     # origin coordinates
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    dt = Inf

    # Physical properties using GeoParams ----------------
    η0 = 1e22           # viscosity
    G0 = 10^10           # elastic shear modulus
    εbg = 1e-14           # background strain-rate
    dt = 1e11
    el_bg = SetConstantElasticity(; G=G0, ν=0.5)
    visc = LinearViscous(; η=η0)
    rheology = (
        # Low density phase
        SetMaterialParams(;
            Phase=1,
            Density=ConstantDensity(; ρ=2700.0),
            Gravity=ConstantGravity(; g=9.81),
            CompositeRheology=CompositeRheology((visc, el_bg)),
            Elasticity=el_bg,
        ),
    )

    # Initialize phase ratios -------------------------------
    radius = 0.1
    phase_ratios = PhaseRatio(ni, length(rheology))
    init_phases!(phase_ratios, xci, radius)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6, CFL=0.75 / √2.1)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    ρg[2] .= rheology[1].Density[1].ρ.val .* rheology[1].Gravity[1].g.val
    args = (; T=@zeros(ni...), P=stokes.P, dt=dt)

    # Rheology
    η = @ones(ni...)
    η_vep = similar(η) # effective visco-elasto-plastic viscosity
    @parallel (@idx ni) compute_viscosity!(
        η,
        1.0,
        phase_ratios.center,
        stokes.ε.xx,
        stokes.ε.yy,
        stokes.ε.xy,
        args,
        rheology,
        (-Inf, Inf),
    )

    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(;
        free_slip=(left=true, right=true, top=true, bot=true),
        no_slip=(left=false, right=false, top=false, bot=false),
    )
    stokes.V.Vx .= PTArray([x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    tmax = 1.0e13
    τII = Float64[0.0]
    sol = Float64[0.0]
    ttot = Float64[0.0]
    P = Float64[0.0]

    while t < tmax

        # Stokes solver ----------------
        solve!(
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
            verbose=false,
            iterMax=500e3,
            nout=1e3,
            viscosity_cutoff=(-Inf, Inf),
        )
        @parallel (JustRelax.@idx ni) JustRelax.Stokes2D.tensor_invariant!(
            stokes.ε.II, @strain(stokes)...
        )
        push!(τII, maximum(stokes.τ.xx))

        if !isinf(dt)
            @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
            @parallel (@idx ni) multi_copy!(
                @tensor_center(stokes.τ_o), @tensor_center(stokes.τ)
            )
        end

        it += 1
        t += dt

        push!(sol, solution(εbg, t, G0, η0))
        push!(ttot, t)
        push!(P, maximum(stokes.P))

        println("it = $it; t = $t \n")

        fig = Figure(; size=(1600, 1600), title="t = $t")
        ax1 = Axis(fig[1, 1]; aspect=1, title="τII")
        ax2 = Axis(fig[2, 1]; aspect=1, title="Pressure")
        lines!(ax1, ttot, τII ./ 1e6; color=:black, label="τII")
        lines!(ax1, ttot, sol ./ 1e6; color=:red, label="sol")
        lines!(ax2, ttot, P; color=:black, label="P")
        Legend(fig[1, 2], ax1)
        Legend(fig[2, 2], ax2)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

N = 32
n = N + 2
nx = n - 2
ny = n - 2
figdir = "ElasticBuildUp_incompressible"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 0; init_MPI=true)...)
else
    igg
end
main(igg; figdir=figdir, nx=nx, ny=ny);
