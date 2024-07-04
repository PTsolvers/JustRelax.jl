# JustRelax.jl
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptsolvers.github.io/JustRelax.jl/dev/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10212422.svg)](https://doi.org/10.5281/zenodo.10212422)
![CI](https://github.com/PTSolvers/JustRelax.jl/actions/workflows/ci.yml/badge.svg)
[![Build status](https://badge.buildkite.com/6b970b1066dc828a56a75bccc65a8bc896a8bb76012a61fe96.svg)](https://buildkite.com/julialang/justrelax-dot-jl)
[![codecov](https://codecov.io/gh/PTsolvers/JustRelax.jl/graph/badge.svg?token=4ZJO7ZGT8H)](https://codecov.io/gh/PTsolvers/JustRelax.jl)


:warning: This Package is still under active development
- The API is still subject to change.
- The benchmarks and miniapps are working and provide the user with an insight into the capabilities of the package.

Need to solve a very large multi-physics problem on many GPUs in parallel? Just Relax!

`JustRelax.jl` is a collection of accelerated iterative pseudo-transient solvers using MPI and multiple CPU or GPU backends. It's part of the [PTSolvers organisation](https://ptsolvers.github.io) and
developed within the [GPU4GEO project](https://www.pasc-ch.org/projects/2021-2024/gpu4geo/). Current publications, outreach and news can be found on the [GPU4GEO website](https://ptsolvers.github.io/GPU4GEO/).

The package relies on other packages as building blocks and parallelisation tools:

* [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
* [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl)
* [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl)
* [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl)


The package serves several purposes:

  * It provides a collection of solvers to be used in quickly developing new applications
  * It provides some standardization so that application codes can

     - more easily handle local material properties through the use of [GeoParams.jl]((https://github.com/JuliaGeodynamics/GeoParams.jl))
     - more easily switch between a pseudo-transient solver and another solvers (e.g. an explicit thermal solvers)

  * It provides a natural repository for contributions of new solvers for use by the larger community

We provide several miniapps, each designed to solve a well-specified benchmark problem, in order to provide

  - examples of usage in high-performance computing
  - basis on which to build more full-featured application codes
  - cases for reference and performance tests


## Installation

`JustRelax.jl` is a registered package and can be added as follows:

```julia
using Pkg; Pkg.add("JustRelax")
```
However, as the API is changing and not every feature leads to a new release, one can also do `add JustRelax#main` which will clone the main branch of the repository.
After installation, you can test the package by running the following commands:

```julia
using JustRelax
julia> ]
  pkg> test JustRelax
```
The test will take a while, so grab a :coffee: or :tea:

## Example: shear band localisation (2D)

![ShearBand2D](docs/src/assets/movies/DP_nx2058_2D.gif)

This example displays how the package can be used to simulate shear band localisation. The example is based on the [ShearBands2D.jl](miniapps/benchmarks/stokes2D/shear_band/ShearBand2D.jl).

```julia
using GeoParams, CellArrays, GLMakie
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

const backend = CPUBackend

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

```
The function `init_phases!` initializes the phases within cell arrays. The function is parallelized with the `@parallel_indices` macro from [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl). In this case, this is the only function that needs to be tailored to the specific problem, everything else is handled by `JustRelax.jl` itself.

```julia
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
```

# Initialize packages

Load JustRelax necessary modules and define backend.
```julia
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CPUBackend
```

For this specific example we use particles to define the material phases, for which we rely on [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl). As in `JustRelax.jl`, we need to set up the environment of `JustPIC.jl`. This is done by running/including the following commands:

```julia
  using JustPIC
  using JustPIC._2D

  const backend = CPUBackend    # Threads is the default backend
  const backend = CUDABackend   # to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script
  const backend = AMDGPUBackend # and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
```

We will also use `ParallelStencil.jl` to write some device-agnostic helper functions:
```julia
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)
```
and will use [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl/tree/main) to define and compute physical properties of the materials:
```julia
using GeoParams
```

For the initial setup, you will need to specify the number of nodes in x- and y- direction `nx` and `ny` as well as the directory where the figures are stored (`figdir`). The initialisation of the global grid and MPI environment is done with `igg = IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)`:

```julia
N      = 128
n      = N + 2
nx     = n - 2
ny     = n - 2
figdir = "ShearBands2D"
igg    = IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)
```

Initialisation of the physical domain and the grid. As `JustRelax.jl` relies on [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl), the grid can be `MPIAWARE`. This makes it a global grid and the grid steps are automatically distributed over the MPI processes.

```julia
# Physical domain ------------------------------------
ly           = 1.0                      # domain length in y
lx           = ly                       # domain length in x
ni           = nx, ny                   # number of cells
li           = lx, ly                   # domain length in x- and y-
di           = @. li / ni               # grid step in x- and -y
origin       = 0.0, 0.0                 # origin coordinates
grid         = Geometry(ni, li; origin = origin)
(; xci, xvi) = grid                     # nodes at the center and vertices of the cells
dt           = Inf
```
Initialisation of the rheology with [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl). The rheology can be tailored to the specific problem with different creep laws and material parameters (see [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl)) or the miniapps in the [convection folder](miniapps/convection).

```julia
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
    Ψ    = 0,
) # viscoplasticity model from e.g. Duretz, T., Räss, L., de Borst, R., & Hageman, T. (2023). A comparison of plasticity regularization approaches for geodynamic modeling. Geochemistry, Geophysics, Geosystems, 24, e2022GC010675. https://doi.org/101029/2022GC010675
rheology = (
    # Matrix phase
    SetMaterialParams(;
        Phase             = 1,
        Density           = ConstantDensity(; ρ = 0.0),
        Gravity           = ConstantGravity(; g = 0.0),
        CompositeRheology = CompositeRheology((visc, el_bg, pl)),
        Elasticity        = el_bg,
    ),
    # Inclusion phase
    SetMaterialParams(;
        Density           = ConstantDensity(; ρ = 0.0),
        Gravity           = ConstantGravity(; g = 0.0),
        CompositeRheology = CompositeRheology((visc, el_inc, pl)),
        Elasticity        = el_inc,
    ),
)
```
Initialisation of the Stokes arrays and the necessary allocations. The rheology is computed with `compute_viscosity!` which is a function from `JustRelax.jl` and computes the viscosity according to the strain rate and the phase ratios.

```julia
# Initialize phase ratios -------------------------------
radius       = 0.1
phase_ratios = PhaseRatio(backend_JR, ni, length(rheology))
init_phases!(phase_ratios, xci, radius)
# STOKES ---------------------------------------------
# Allocate arrays needed for every Stokes problem
stokes = StokesArrays(backend_JR, ni)
pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL = 0.75 / √2.1)
# PT coefficients after Räss, L., Utkin, I., Duretz, T., Omlin, S., and Podladchikov, Y. Y.: Assessing the robustness and scalability of the accelerated pseudo-transient method, Geosci. Model Dev., 15, 5757–5786, https://doi.org/10.5194/gmd-15-5757-2022, 2022.
# Buoyancy forces
ρg               = @zeros(ni...), @zeros(ni...)
η                = @ones(ni...)
args             = (; T = thermal.Tc, P = stokes.P, dt = Inf)
compute_ρg!(ρg[2], phase_ratios, rheology, args)
compute_viscosity!(stokes, 1.0, phase_ratios, args, rheology, (-Inf, Inf))
```

Define pure shear velocity boundary conditions or displacement boundary conditions. The boundary conditions are applied with `flow_bcs!` and the halo is updated with `update_halo!`.
```julia
# Boundary conditions
flow_bcs     = VelocityBoundaryConditions(;
    free_slip = (left = true, right = true, top = true, bot = true),
    no_slip   = (left = false, right = false, top = false, bot=false),
)
stokes.V.Vx .= PTArray([ x*εbg for x in xvi[1], _ in 1:ny+2])
stokes.V.Vy .= PTArray([-y*εbg for _ in 1:nx+2, y in xvi[2]])
flow_bcs!(stokes, flow_bcs) # apply boundary conditions
update_halo!(@velocity(stokes)...)
```
```julia
# Boundary conditions
flow_bcs     = DisplacementBoundaryConditions(;
free_slip = (left = true, right = true, top = true, bot = true),
no_slip   = (left = false, right = false, top = false, bot=false),
)
stokes.U.Ux .= PTArray(backend)([ x*εbg*lx*dt for x in xvi[1], _ in 1:ny+2])
stokes.U.Uy .= PTArray(backend)([-y*εbg*ly*dt for _ in 1:nx+2, y in xvi[2]])
flow_bcs!(stokes, flow_bcs) # apply boundary conditions
displacement2velocity!(stokes, dt)   # convert displacement to velocity
update_halo!(@velocity(stokes)...)
```
Pseudo-transient Stokes solver and visualisation of the results. The visualisation is done with [GLMakie.jl](https://github.com/MakieOrg/Makie.jl).

```julia
# if it does not exist, make a folder where figures are stored
take(figdir)

# Time loop
t, it, tmax    = 0.0, 0,  3.5
τII, sol, ttot = Float64[], Float64[], Float64[]

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
          iterMax          = 150e3,
          nout             = 200,
          viscosity_cutoff = (-Inf, Inf),
          verbose          = true
      )
  )
    # Compute second invariant of the strain rate tensor
    tensor_invariant!(stokes.ε)
    push!(τII, maximum(stokes.τ.xx))

    it += 1
    t  += dt

    push!(sol, solution(εbg, t, G0, η0))
    push!(ttot, t)

    println("it = $it; t = $t \n")

    # visualisation of high density inclusion
    th    = 0:pi/50:3*pi;
    xunit = @. radius * cos(th) + 0.5;
    yunit = @. radius * sin(th) + 0.5;

    fig   = Figure(size = (1600, 1600), title = "t = $t")
    ax1   = Axis(fig[1,1], aspect = 1, title = L"\tau_{II}", titlesize=35)
    ax2   = Axis(fig[2,1], aspect = 1, title = L"E_{II}", titlesize=35)
    ax3   = Axis(fig[1,2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize=35)
    ax4   = Axis(fig[2,2], aspect = 1)
    heatmap!(ax1, xci..., Array(stokes.τ.II) , colormap=:batlow)
    heatmap!(ax2, xci..., Array(log10.(stokes.EII_pl)) , colormap=:batlow)
    heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)) , colormap=:batlow)
    lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
    lines!(ax4, ttot, τII, color = :black)
    lines!(ax4, ttot, sol, color = :red)
    hidexdecorations!(ax1)
    hidexdecorations!(ax3)
    save(joinpath(figdir, "$(it).png"), fig)
    fig
end
```

## Miniapps

Currently there are 4 convection miniapps with particles and 4 corresponding miniapps without. The miniapps with particles are:

  * [Layered_convection2D.jl](miniapps/convection/Particles2D/Layered_convection2D.jl)
  * [Layered_convection2D_nonDim.jl](miniapps/convection/Particles2D_nonDim/Layered_convection2D.jl)
  * [Layered_convection3D.jl](miniapps/convection/Particles3D/Layered_convection3D.jl)
  * [WENO_convection2D.jl](miniapps/convection/WENO5/WENO_convection2D.jl)

The miniapps without particles are:
  * [GlobalConvection2D_Upwind.jl](miniapps/convection/GlobalConvection2D_Upwind.jl)
  * [GlobalConvection2D_WENO5.jl](miniapps/convection/GlobalConvection2D_WENO5.jl)
  * [GlobalConvection2D_WENO5_MPI.jl](miniapps/convection/GlobalConvection2D_WENO5_MPI.jl)
  * [GlobalConvection3D_Upwind.jl](miniapps/convection/GlobalConvection3D_Upwind.jl)

## Benchmarks

Current (Blankenback2D, Stokes 2D-3D, thermal diffusion, thermal stress) and future benchmarks can be found in the [Benchmarks](miniapps/benchmarks).

## Funding
The development of this package is supported by the [GPU4GEO](https://ptsolvers.github.io/GPU4GEO/) [PASC](https://www.pasc-ch.org) project.
