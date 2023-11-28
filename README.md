# JustRelax.jl

![CI](https://github.com/PTSolvers/JustRelax.jl/actions/workflows/ci.yml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10212423.svg)](https://doi.org/10.5281/zenodo.10212423)


:warning: This Package is still under active development. The benchmarks and miniapps are working and provide the user with an insight into the capabilities of the package. However, the API is still subject to change. :warning: 

Need to solve a very large multi-physics problem on a GPU cluster? Just Relax!


Pseudo-transient accelerated iterative solvers, ready for extreme-scale, multi-GPU computation.

`JustRelax.jl` is a collection of pseudo-transient relaxation solvers
for multi-physics problems on regular, staggered, parallel grids,
using MPI and multiple CPU or GPU backends. It's part of the [PTSolvers project](https://ptsolvers.github.io) and
the [GPU4GEO project](https://www.pasc-ch.org/projects/2021-2024/gpu4geo/). Current publications, outreach and news can be found on the [GPU4GEO website](https://ptsolvers.github.io/GPU4GEO/).

The package relies on other packages as building blocks and parallelisation tools:

* [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
* [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl)
* [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl)
* [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl)


The package serves several purposes:

  * It provides a collection of solvers to be used in quickly developing new applications
  * It provides some standardization so that application codes can

     - more easily handle local material propoerties through the use of [GeoParams.jl]((https://github.com/JuliaGeodynamics/GeoParams.jl))
     - more easily switch between a psuedo-transient solver and another solver (e.g. an explicit thermal solvers)

  * It provides a place to describe performance benchmarks for the solver routines
  * It provides a natural location for contributions of new solvers for use by the larger community

We include several miniapps, each designed to solve a well-specified benchmark problem, in order to provide

  - examples of high-performance usage,
  - bases on which to build more full-featured application codes
  - cases for reference and performance tests


## Installation

`JustRelax.jl` is not yet registered (we are in the process), however it can be installed by cloning the repository. If you have navigated the terminal to the directory you cloned `JustRelax.jl`, you can test the package by running the following commands:

```julia
using JustRelax
julia> ] 
  pkg> activate .
  pkg> instantiate
  pkg> test JustRelax
```
The test will take a while, so grab a coffee or tea. 

## Usage

For the advection of physical fields, this package relies on the Particles-In-Cell method implemented in [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl). Depending on where you want to run `JustRelax.jl` (CPU or GPU) you will need to specify the Backend. Run this command in the REPL and restart Julia:

```julia
  set_backend("Threads_Float64_2D")       #running on the CPU
  set_backend("CUDA_Float64_2D")          #running on an NVIDIA GPU
  set_backend("AMDGPU_Float64_2D")        #running on an AMD GPU
```
After you have run your desired line and restarted Julia, there should be a file called `LocalPreferences.toml` in the directory together with your `Project.toml` and `Manifest.toml`. This file contains the information about the backend you want to use. If you want to change the backend, you can run the command again. The default backend is `Threads_Float64_2D`.


As stated above, the parallelisation is done using ParallelStencil.jl. Therefore, `JustRelax.jl` has as an environment already setup with which you can specify the dimension of your problem (2D or 3D) and the backend (CPU or GPU). The `PS_setup` variables are `PS_Setup(device, precission, dimensions)`. If you therefore want to run a 3D code, change the `dimensions` to 3 in the available commands below.

```julia
  model = PS_Setup(:Threads, Float64, 2)  #running on the CPU in 2D
  environment!(model)

  model = PS_Setup(:CUDA, Float64, 2)     #running on an NVIDIA GPU in 2D
  environment!(model)

  model = PS_Setup(:AMDGPU, Float64, 2)   #running on an AMD GPU in 2D
  environment!(model)
```

## Shear Band 2D example

This example benchmark test displays how the package can be used to simulate shear bands. The example is based on the [ShearBands2D.jl](miniapps/benchmarks/stokes2D/shear_band/ShearBand2D.jl). 

![ShearBand2D](miniapps/benchmarks/stokes2D/shear_band/movies/DP_nx2058_2D.gif)

The `CUDA.allowscalar(false)` disallows scalar indexing. This could be regarded as a safety measure, as the scalar indexing would run super slow on the CPU with CUDA and will actually crash with AMD. 

```julia
using CUDA
CUDA.allowscalar(false)

using GeoParams, GLMakie, CellArrays
using JustRelax, JustRelax.DataIO

# setup ParallelStencil.jl environment
model  = PS_Setup(:CUDA, Float64, 2)
environment!(model)

# HELPER FUNCTIONS ---------------------------------------------------------------
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

```
The function `init_phases!` initializes the phases within cell arrays. The function is parallelized with the `@parallel_indices` macro from [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl). In this case, this is the only function that needs to be taylored to the specific problem, everything else is handled by `JustRelax.jl` itself. 

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
Initialisation of the physical domain and the grid. As `JustRelax.jl` relies on [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl), the grid can be `MPIAWARE` through setting the grid steps in x- and y- direction to ` di = @. li / (nx_g(),ny_g())`. This makes it a global grid and the grid steps are automatically distributed over the MPI processes. 

```julia

# MAIN SCRIPT --------------------------------------------------------------------
function main(igg; nx=64, ny=64, figdir="model_figs")

    # Physical domain ------------------------------------
    ly       = 1e0          # domain length in y
    lx       = ly           # domain length in x
    ni       = nx, ny       # number of cells
    li       = lx, ly       # domain length in x- and y-
    di       = @. li / ni   # grid step in x- and -y
    origin   = 0.0, 0.0     # origin coordinates
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    dt       = Inf 
```
Initialisation of the rheology with [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl). The rheology can be taylored to the specific problem with different creep laws and material parameters (see [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl)) or the miniapps in the [convection folder](miniapps/convection).

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
    ) # Duretz, T., Räss, L., de Borst, R., & Hageman, T. (2023). A comparison of plasticity regularization approaches for geodynamic modeling. Geochemistry, Geophysics, Geosystems, 24, e2022GC010675. https://doi.org/10.1029/2022GC010675
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
Initialisation of the Stokes arrays and the necesseary allocations. The rheology is computed with `compute_viscosity!` which is a function from `JustRelax.jl` and computes the viscosity according to the strain rate and the phase ratios.

```julia
    # Initialize phase ratios -------------------------------
    radius       = 0.1
    phase_ratios = PhaseRatio(ni, length(rheology))
    init_phases!(phase_ratios, xci, radius)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL = 0.75 / √2.1) 
    #PT coefficients after Räss, L., Utkin, I., Duretz, T., Omlin, S., and Podladchikov, Y. Y.: Assessing the robustness and scalability of the accelerated pseudo-transient method, Geosci. Model Dev., 15, 5757–5786, https://doi.org/10.5194/gmd-15-5757-2022, 2022.

    # Buoyancy forces
    ρg        = @zeros(ni...), @zeros(ni...)
    args      = (; T = @zeros(ni...), P = stokes.P, dt = dt)
    
    # Rheology
    η         = @ones(ni...)
    η_vep     = similar(η) # effective visco-elasto-plastic viscosity
    @parallel (@idx ni) compute_viscosity!(
        η, 1.0, phase_ratios.center, stokes.ε.xx, stokes.ε.yy, stokes.ε.xy, args, rheology, (-Inf, Inf)
    )
```
Apply the velocity boundery conditions, in this case a pure shear flow. 

```julia
    # Boundary conditions
    flow_bcs     = FlowBoundaryConditions(; 
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip   = (left = false, right = false, top = false, bot=false),
    )
    stokes.V.Vx .= PTArray([ x*εbg for x in xvi[1], _ in 1:ny+2])
    stokes.V.Vy .= PTArray([-y*εbg for _ in 1:nx+2, y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
```
Pseudo-transient Stokes solver and visualisation of the results. The visualisation is done with [GLMakie.jl](https://github.com/MakieOrg/Makie.jl). 

```julia    
    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it      = 0.0, 0
    tmax       = 3.5
    τII        = Float64[]
    sol        = Float64[]
    ttot       = Float64[]

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
            verbose          = false,
            iterMax          = 500e3,
            nout             = 1e3,
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

        # visualisation
        th    = 0:pi/50:3*pi;
        xunit = @. radius * cos(th) + 0.5;
        yunit = @. radius * sin(th) + 0.5;

        fig   = Figure(resolution = (1600, 1600), title = "t = $t")
        ax1   = Axis(fig[1,1], aspect = 1, title = "τII")
        ax2   = Axis(fig[2,1], aspect = 1, title = "η_vep")
        ax3   = Axis(fig[1,2], aspect = 1, title = "log10(εII)")
        ax4   = Axis(fig[2,2], aspect = 1)
        heatmap!(ax1, xci..., Array(stokes.τ.II) , colormap=:batlow)
        heatmap!(ax2, xci..., Array(log10.(η_vep)) , colormap=:batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)) , colormap=:batlow)
        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII, color = :black) 
        lines!(ax4, ttot, sol, color = :red) 
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)

    end

    return nothing
end

```
For the initial setup, you will need to specify the number of nodes in x- and y- direction `nx` and `ny` as well as the directory where the figures are stored `figdir`. The initialisation of the global grid is done with `igg = IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)`. The `init_MPI = true` is needed to initialise the MPI environment.

```julia

N      = 128
n      = N + 2
nx     = n - 2
ny     = n - 2
figdir = "ShearBands2D"
igg  = if !(JustRelax.MPI.Initialized())
    IGG(init_global_grid(nx, ny, 0; init_MPI = true)...)
else
    igg
end
main(igg; figdir = figdir, nx = nx, ny = ny);
```

## Miniapps

Currenlty there are 3 convection miniapps with particles and 3 corresponding miniapps without. The miniapps with particles are:

  * [Layered_convection2D.jl](miniapps/convection/Particles2D/Layered_convection2D.jl)
  * [Layered_convection3D.jl](miniapps/convection/Particles3D/Layered_convection3D.jl)
  * [WENO_convection2D.jl](miniapps/convection/WENO5/WENO_convection2D.jl)

The miniapps without particles are:
  * [GlobalConvection2D_Upwind.jl](miniapps/convection/GlobalConvection2D_Upwind.jl)
  * [GlobalConvection3D_Upwind.jl](miniapps/convection/GlobalConvection3D_Upwind.jl)
  * [GlobalConvection2D_WENO5.jl](miniapps/convection/GlobalConvection2D_WENO5.jl)

  ## Benchmarks

  We got multiple benchmark tests for Stokes in 2D and 3D as well as thermal diffusion. The benchmarks are updated to be up to date to the current Solvers and version. The benchmarks can be found in the [Benchmarks](miniapps/benchmarks).