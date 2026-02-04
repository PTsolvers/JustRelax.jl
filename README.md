<h1> <img src="./docs/src/assets/logo.png" alt="JustRelax.jl" width="50"> JustRelax.jl </h1>

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ptsolvers.github.io/JustRelax.jl/dev/)
[![Ask us anything](https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg)](https://github.com/PTsolvers/JustRelax.jl/discussions/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10212422.svg)](https://doi.org/10.5281/zenodo.10212422)
[![CPU Unit Tests](https://img.shields.io/github/actions/workflow/status/PTSolvers/JustRelax.jl/ci.yml?label=CPU%20Unit%20Tests)](https://github.com/PTSolvers/JustRelax.jl/actions/workflows/ci.yml)
[![GPU Unit Tests](https://img.shields.io/buildkite/6b970b1066dc828a56a75bccc65a8bc896a8bb76012a61fe96/main?label=GPU%20Unit%20Tests)](https://buildkite.com/julialang/justrelax-dot-jl)
[![CSCS CI](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/282716251344384/8101554320689785/badges/main/pipeline.svg?ignore_skipped=true)](https://gitlab.com/cscs-ci/ci-testing/webhook-ci/mirrors/282716251344384/8101554320689785/-/pipelines)
[![codecov](https://codecov.io/gh/PTsolvers/JustRelax.jl/graph/badge.svg?token=4ZJO7ZGT8H)](https://codecov.io/gh/PTsolvers/JustRelax.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FJustRelax&query=total_requests&suffix=%2Fmonth&label=Downloads)](http://juliapkgstats.com/pkg/JustRelax)
[![SQAaaS badge shields.io](https://img.shields.io/badge/sqaaas%20software-silver-lightgrey)](https://api.eu.badgr.io/public/assertions/gXEVz2XaS4iE-zi4lMY5pA "SQAaaS silver badge achieved")
[![DOI](https://joss.theoj.org/papers/10.21105/joss.09365/status.svg)](https://doi.org/10.21105/joss.09365)

<p align="center"><img src="./docs/src/assets/logo.png" alt="JustRelax.jl" width="200"></p>

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

(@v1.xx) pkg> test JustRelax
```

The test will take a while, so grab a :coffee: or :tea:

:warning: If you plan on developing JustRelax.jl and/or modifying the source coude, you can test your local version by running the testing framework again
```julia
julia> ]
(@v1.xx) pkg> test JustRelax
```

## Miniapps

Available examples and [Benchmark](miniapps/benchmarks) miniapps can be found in the [miniapps folder](miniapps). The miniapps are simple and easy to understand, while still providing a good basis for more complex applications. The miniapps are designed to be run on a single node, but can be easily extended to run on multiple nodes using [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl) and [MPI.jl](https://github.com/JuliaParallel/MPI.jl). To get started, instantiate the miniapps folder and run our favourite one!

## Funding
The development of this package is supported by the [GPU4GEO & δGPU4GEO](https://gpu4geo.org/) [PASC](https://www.pasc-ch.org) project, and the European Research Council through the MAGMA project, ERC Consolidator Grant #771143.
