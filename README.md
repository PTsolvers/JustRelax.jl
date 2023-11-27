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
