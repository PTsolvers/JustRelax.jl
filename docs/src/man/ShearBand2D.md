# Shear-band localization

This two-dimensional visco-elasto-plastic model localizes deformation around
a circular inclusion with lower elastic shear modulus than the background.
It uses a Drucker-Prager material model and compares the simulated stress
evolution with the analytical response of a Maxwell material.

A higher-resolution run shows the localized shear-band evolution:

![Shear-band evolution](../assets/movies/DP_nx2058_2D.gif)

Run this miniapp from the repository root with

```sh
julia --project=miniapps --startup-file=no miniapps/benchmarks/stokes2D/shear_band/ShearBand2D.jl
```

The simulation writes one figure per time step to `ShearBands2D/`.

## Imports and backends

The model uses the threaded CPU backend for both JustRelax and JustPIC.
`ParallelStencil` supplies the device-agnostic phase-initialization kernel,
`GeoParams` defines the material behavior, and CairoMakie writes the figures.

````julia
using GeoParams, CairoMakie
using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

using JustPIC
import JustPIC.GridGeometryUtils as GGU

const backend_JP = JustPIC.CPU
````

## Helper functions

`solution` is the analytical normal-stress response used for comparison in
the final time-history panel.

````julia
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))
````

Initialize the particle phases from a circular inclusion. The kernel fills
the phase-ratio fields at both cell centers and vertices.

````julia
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
````

## Model setup and solution

````julia
function main(igg; nx = 64, ny = 64, figdir = "model_figs")
````

### Model domain

The unit-square domain is discretized with `nx × ny` cells. `Geometry`
provides the staggered center and vertex coordinates used below.

````julia
    ly = 1.0e0          # domain length in y
    lx = ly           # domain length in x
    ni = nx, ny       # number of cells
    li = lx, ly       # domain length in x- and y-
    di = @. li / ni   # grid step in x- and -y
    origin = 0.0, 0.0     # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
````

### Material properties

The background and inclusion use the same viscous and plastic laws, but
the inclusion has a lower elastic shear modulus. The material model is a
linear viscosity, elasticity, and regularized Drucker-Prager plasticity.

````julia
    τ_y = 1.6           # yield stress. If do_DP=true, τ_y stand for the cohesion: c*cos(ϕ)
    ϕ = 30            # friction angle
    C = τ_y           # Cohesion
    η0 = 1.0           # viscosity
    G0 = 1.0           # elastic shear modulus
    Gi = G0 / (6.0 - 4.0)  # elastic shear modulus perturbation
    εbg = 1.0           # background strain-rate
    η_reg = 8.0e-3          # regularisation "viscosity"
    dt = η0 / G0 / 6.0     # assumes Maxwell time of 4
    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
````

non-regularized plasticity

````julia
        C = C / cosd(ϕ),
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0
    )

    rheology = (
````

Low density phase

````julia
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,

        ),
````

High density phase

````julia
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )
````

### Phase anomaly

Allocate a cohesion perturbation and initialize the circular inclusion.
`PhaseRatios` carries phase fractions on the staggered grid; the circle
occupies phase 2 and the background is phase 1.

````julia
    perturbation_C = @zeros(ni...)
````

Initialize phase ratios -------------------------------

````julia
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.1
    origin = 0.5, 0.5
    circle = GGU.Circle(origin, radius)
    init_phases!(phase_ratios, xci, xvi, circle)
````

### Stokes state and boundary conditions

`StokesArrays` stores the velocity, pressure, stress, and strain-rate
fields. `PTStokesCoeffs` allocates the pseudo-transient solver state.

````julia
    stokes = StokesArrays(backend, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-6, CFL = 0.95 / √2.1)
````

The phases have zero density, so buoyancy is zero in this benchmark.

````julia
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt, perturbation_C = perturbation_C)
````

Initialize the viscosity from the phase ratios and material parameters.

````julia
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )
````

The imposed velocity is extension in `x` and compression in `y`; all
boundaries are free slip. Halo updates make the fields consistent when
the model is distributed across MPI ranks.

````julia
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)
````

### Time integration and visualization

Each time step solves Stokes to the requested residual tolerance, then
records the maximum normal stress with the analytical Maxwell response.
The four-panel figure shows stress, plastic strain, strain rate, and the
stress history, and is written to `figdir`.

````julia
    take(figdir)
````

Time loop

````julia
    t, it = 0.0, 0
    tmax = 5
    τII = Float64[]
    sol = Float64[]
    ttot = Float64[]
````

while t < tmax

````julia
    for _ in 1:19
````

Stokes solver ----------------

````julia
        iters = solve!(
            stokes,
            pt_stokes,
            grid,
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

        println("it = $it; t = $t \n")
````

visualisation

````julia
        th = 0:(pi / 50):(3 * pi)
        xunit = @. radius * cos(th) + 0.5
        yunit = @. radius * sin(th) + 0.5
        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax3 = Axis(fig[1, 2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 2], aspect = 1)
        heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :batlow)
````

heatmap!(ax2, xci..., Array(log10.(stokes.viscosity.η_vep)) , colormap=:batlow)

````julia
        heatmap!(ax2, xci..., Array(stokes.EII_pl), colormap = :batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII, color = :black)
        lines!(ax4, ttot, sol, color = :red)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

nx = 128
ny = 256
figdir = "ShearBands2D"
igg = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

