```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: JustRelax.jl Docs
  text: Solving multi-physics problems
  tagline: A collection of iterative accelerated pseudo-transient solvers using MPI for distributed computing on GPUs or CPUs.
  actions:
    - theme: brand
      text: Getting Started
      link: /man/installation
    - theme: alt
      text: API Reference 📚
      link: /man/listfunctions
    - theme: alt
      text: View on GitHub
      link: https://github.com/PTsolvers/JustRelax.jl
  image:
    src: /logo.png
    alt: JustRelax.jl

features:
  - icon: 🚀
    title: Backend Agnostic
    details: Effortlessly execute your code on CPUs and GPUs with ParallelStencils.jl.
    link: /concepts/architectures

  - icon: 🛠️
    title: Framework
    details: A collection of solvers to be used in prototyping new applications.
    link: /getting_started/introduction

  - icon: ⚡
    title: Standardization
    details: Allow applications to handle local material properties and switch between solving strategies.
    link: /concepts/distributed

  - icon: 🧩
    title: Extensibility
    details: Provides a natural repository for contributions of new solvers for use by the larger community.
    link: /concepts/kernels
---
```

## What is JustRelax.jl?

[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) is **a collection of accelerated iterative pseudo-transient solvers** using MPI for distributed memory parallelisation  and running on GPUs or CPUs.

The package relies on other packages as building blocks and parallelisation tools:

* [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) - device agnostic parallel kernels
* [ImplicitGlobalGrid.jl](https://github.com/omlins/ImplicitGlobalGrid.jl) - (GPU-aware) distributed parallelisation (MPI)
* [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) - Material physics
* [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl) - Particle-in-cell advection

Current publications, outreach and news can be found on the [GPU4GEO website](https://GPU4GEO.org).

## How to Install JustRelax.jl?

To install JustRelax.jl, one can simply add it using the Julia package manager by running the following command in the Julia REPL:

```julia
julia> using Pkg

julia> Pkg.add("JustRelax")
```

After the package is installed, one can load the package by using:

```julia
julia> using JustRelax
```

If you want to use the latest unreleased version of JustRelax.jl, you can run the following command:

```julia
julia> using Pkg

julia> Pkg.add(url="https://github.com/PTsolvers/JustRelax.jl")
```

## Funding

The development of this package is supported by the [GPU4GEO](https://pasc-ch.org/projects/2021-2024/gpu4geo/index.html) and ∂GPU4GEO PASC projects. More information about the GPU4GEO project can be found on the [GPU4GEO website](https://GPU4GEO.org/).
