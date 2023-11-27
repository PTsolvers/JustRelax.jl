# JustRelax.jl

![CI](https://github.com/PTSolvers/JustRelax.jl/actions/workflows/ci.yml/badge.svg)

:warning: This Package is still under active development. The benchmarks and miniapps are working and provide the user with an insight into the capabilities of the package. However, the API is still subject to change. :warning: 

Need to solve a very large multi-physics problem on a GPU cluster? Just Relax!

Pseudo-transient accelerated iterative solvers, ready for extreme-scale, multi-GPU computation.

JustRelax.jl is a collection of pseudo-transient relaxation solvers
for multi-physics problems on regular, staggered, parallel grids,
using MPI and multiple CPU or GPU backends. It's part of the [PTSolvers project](https://ptsolvers.github.io) and
the [GPU4GEO project](https://www.pasc-ch.org/projects/2021-2024/gpu4geo/). Current publications, outreach and news can be found on the [GPU4GEO website](https://ptsolvers.github.io/GPU4GEO/).

The package relies on other packages as building blocks and parallelisation tools:

* [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
* [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl)
* [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl)
* [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl)


The package serves several purposes:

  * It reduces code duplication between several applications, e.g. [PseudoTransientStokes.jl](https://github.com/PTsolvers/PseudoTransientStokes.jl). 
  * It provides a collection of solvers to be used in quickly developing new applications
  * It provides some standardization so that application codes can

     - more easily "add more physics" through the use of [GeoParams.jl]((https://github.com/JuliaGeodynamics/GeoParams.jl))
     - more easily switch between a psuedo-transient solver and another solver (e.g. an explicit thermal solvers)

  * It provides a place to describe performance benchmarks for the solver routines
  * It provides a natural location for contributions of new solvers for use by the larger community

We include several miniapps, each designed to solve a well-specified benchmark problem, in order to provide

  - examples of high-performance usage,
  - bases on which to build more full-featured application codes
  - cases for reference and performance tests


JustRelax.jl is used in the following applications:

  * TODO link to all applications using the package here (crucial for early development)

## Installation

JustRelax.jl is not yet registered (we are in the process), however it can be installed by cloning the repository. If you have navigated the terminal to the directory you cloned JustRelax.jl, you can test the package by running the following commands:

```julia
using JustRelax
julia> ] 
  pkg> activate .
  pkg> instantiate
  pkg> test JustRelax
```
The test will take a while, so grab a coffee or tea. 

## Usage

The package relies on [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl) for the particle in cell method. Depending on where you want to run JustRelax.jl (CPU or GPU) you will need to specify the Backend. Run this command in the REPL and restart Julia:

```julia
  set_backend("Threads_Float64_2D")       #running on the CPU
  set_backend("CUDA_Float64_2D")          #running on an NVIDIA GPU
  set_backend("AMDGPU_Float64_2D")        #running on an AMD GPU
```
After you have run your desired line and restarted Julia, there should be a file called `LocalPreferences.toml` in the directory together with your `Project.toml` and `Manifest.toml`. This file contains the information about the backend you want to use. If you want to change the backend, you can run the command again. 


As stated above, the parallelisation is done using ParallelStencil.jl. Therefore, JustRelax.jl has as an environment already setup with which you can specify the dimension of your problem (2D or 3D) and the backend (CPU or GPU). The following commands are available:

```julia
  model = PS_Setup(:Threads, Float64, 2)  #running on the CPU in 2D
  environment!(model)

  model = PS_Setup(:CUDA, Float64, 2)     #running on an NVIDIA GPU in 2D
  environment!(model)

  model = PS_Setup(:AMDGPU, Float64, 2)   #running on an AMD GPU in 2D
  environment!(model)
```

## Miniapps

Currenlty there are 3 convection miniapps with particles and 3 corresponding miniapps without. The miniapps with particles are:

  * [Layered_convection2D.jl](miniapps/convection/Particles2D/Layered_convection2D.jl)
  * [Layered_convection3D.jl](miniapps/convection/Particles3D/Layered_convection3D.jl)
  * [WENO_convection2D.jl](miniapps/convection/Particles2D/WENO_convection2D.jl)

The miniapps without particles are:
  * [GlobalConvection2D_Upwind.jl](miniapps/convection/GlobalConvection2D_Upwind.jl)
  * [GlobalConvection3D_Upwind.jl](miniapps/convection/GlobalConvection3D_Upwind.jl)
  * [GlobalConvection2D_WENO5.jl](miniapps/convection/GlobalConvection2D_WENO5.jl)

  ## Benchmarks

  We got multiple benchmark tests for Stokes in 2D and 3D as well as thermal diffusion. The benchmarks are updated to be up to date to the current Solvers and version. The benchmarks can be found in the [Benchmarks](miniapps/benchmarks).