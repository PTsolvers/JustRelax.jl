# Visco-elastic stress build-up under constant pure-shear strain rate, solved with DYREL.
# A homogeneous Maxwell material relaxes toward the viscous stress 2·η·εbg with the
# analytic transient τ(t) = 2·εbg·η·(1 - exp(-G·t/η)). DYREL carries the elastic stress
# memory (τ_o) across successive solves, so the buildup accumulates over the time loop.

using GeoParams, CairoMakie
using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC, JustPIC._2D

const backend_JP = JustPIC.CPUBackend

# Analytical solution
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

function main(igg; nx = 63, ny = 63, lx = 100.0e3, ly = 100.0e3, endtime = 500, η0 = 1.0e22, εbg = 1.0e-14, G = 1.0e10, figdir = "ElasticBuildUp_DYREL")

    # Physical domain ------------------------------------
    ni = nx, ny
    li = lx, ly
    di = @. li / ni
    origin = 0.0, 0.0
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid

    # Time discretisation --------------------------------
    yr = 365.25 * 3600 * 24
    kyr = 1.0e3 * yr
    ttot = endtime * kyr

    # Rheology (single Maxwell phase) --------------------
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = η0), ConstantElasticity(; G = G, ν = 0.49))),
            Elasticity = ConstantElasticity(; G = G, ν = 0.49),
        ),
    )

    # Phase ratios (single homogeneous phase) ------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    phase_ratios.center.data .= 1.0
    phase_ratios.vertex.data .= 1.0

    # STOKES ---------------------------------------------
    stokes = StokesArrays(backend, ni)
    ρg = @zeros(ni...), @zeros(ni...)
    dt = 0.05 * kyr
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)
    compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))

    # Boundary conditions: background pure shear ---------
    pureshear_bc!(stokes, xci, xvi, εbg, backend)
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    take(figdir)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, grid.di, dt; ϵ = 1.0e-6)

    # Time loop ------------------------------------------
    t = 0.0
    it = 0
    av_τyy = Float64[]
    sol_τyy = Float64[]
    tt = Float64[]
    while t < ttot
        dt = t < 10 * kyr ? 0.05 * kyr : 1.0 * kyr
        args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

        solve_DYREL!(
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
                iterMax = 150.0e3,
                nout = 1000,
                viscosity_cutoff = (-Inf, Inf),
            )
        )

        t += dt
        it += 1
        println("Iteration $it => t = $(t / kyr) kyrs")

        push!(av_τyy, maximum(abs.(stokes.τ.yy)))
        push!(sol_τyy, solution(εbg, t, G, η0))
        push!(tt, t / kyr)
    end

    # Visualisation --------------------------------------
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1]; xlabel = "kyrs", ylabel = "Stress [MPa]")
    scatter!(ax, tt, sol_τyy ./ 1.0e6; label = "analytic")
    lines!(ax, tt, av_τyy ./ 1.0e6; label = "DYREL", linewidth = 3, color = :black)
    axislegend(ax; position = :rb)
    save(joinpath(figdir, "ElasticBuildUp_DYREL.png"), fig)

    return (; ni, xci, xvi, li), stokes, av_τyy, sol_τyy, tt
end

nx = ny = 63
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; nx = nx, ny = ny);
