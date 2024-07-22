```@meta
CurrentModule = JustRelax
```# JustRelax.jl

Need to solve a very large multi-physics problem on many GPUs in parallel? Just Relax!

`JustRelax` is a collection of accelerated iterative pseudo-transient solvers using MPI and multiple CPU or GPU backends. It's part of the [PTSolvers organisation](https://ptsolvers.github.io) and
developed within the [GPU4GEO project](https://www.pasc-ch.org/projects/2021-2024/gpu4geo/). Current publications, outreach and news can be found on the [GPU4GEO website](https://ptsolvers.github.io/GPU4GEO/).

The package relies on other packages as building blocks and parallelisation tools:

* [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) - device agnostic parallel kernels
* [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl) - (CUDA-aware) MPI communication
* [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) - Material physics
* [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl) - Particle-in-cell advection


The package serves several purposes:

  * It provides a collection of solvers to be used in quickly developing new applications
  * It provides some standardization so that application codes can

     - more easily handle local material properties through the use of [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl)
     - more easily switch between a pseudo-transient solver and another solvers (e.g. an explicit thermal solvers)

  * It provides a natural repository for contributions of new solvers for use by the larger community

We provide several miniapps, each designed to solve a well-specified benchmark problem, in order to provide

  - examples of usage in high-performance computing
  - basis on which to build more full-featured application codes
  - cases for reference and performance tests
