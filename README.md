<h1> <img src="./docs/src/assets/logo.png" alt="JustRelax.jl" width="50"> JustRelax.jl </h1>

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptsolvers.github.io/JustRelax.jl/dev/)
[![Ask us anything](https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg)](https://github.com/PTsolvers/JustRelax.jl/discussions/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10212422.svg)](https://doi.org/10.5281/zenodo.10212422)
![CI](https://github.com/PTSolvers/JustRelax.jl/actions/workflows/ci.yml/badge.svg)
[![Build status](https://badge.buildkite.com/6b970b1066dc828a56a75bccc65a8bc896a8bb76012a61fe96.svg?branch=main)](https://buildkite.com/julialang/justrelax-dot-jl)
[![codecov](https://codecov.io/gh/PTsolvers/JustRelax.jl/graph/badge.svg?token=4ZJO7ZGT8H)](https://codecov.io/gh/PTsolvers/JustRelax.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img src="./docs/src/assets/logo.png" alt="JustRelax.jl" width="200"></p>

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

## Miniapps

Available miniapps can be found in the [miniapps folder](miniapps) and will be updated regularly. The miniapps are designed to be simple and easy to understand, while still providing a good basis for more complex applications. The miniapps are designed to be run on a single node, but can be easily extended to run on multiple nodes using [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl) and [MPI.jl](https://github.com/JuliaParallel/MPI.jl).


## Benchmarks

Current (Blankenback2D, Stokes 2D-3D, thermal diffusion, thermal stress) and future benchmarks can be found in the [Benchmarks](miniapps/benchmarks).

## Funding
The development of this package is supported by the [GPU4GEO](https://ptsolvers.github.io/GPU4GEO/) [PASC](https://www.pasc-ch.org) project.
