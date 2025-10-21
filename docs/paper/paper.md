---
title: '`JustRelax.jl`: A Julia package for geodynamic modeling with matrix-free solvers'
tags:
  - Julia
  - Geosciences
  - Geodynamics
  - Tectonics
  - Geophysics
  - Computational geosciences
authors:
  - name: Albert de Montserrat
    orcid: 0000-0003-1694-3735
    affiliation: 1
  - name: Pascal S. Aellig
    orcid: 0009-0008-9039-5646
    affiliation: 2
  - name: Christian Schuler
    orcid: 0009-0004-9873-9774
    affiliation: 2
  - name: Ivan Navarrete
    orcid: 0009-0004-9272-511X
    affiliation: 3
  - name: Ludovic Räss
    orcid: 0000-0002-1136-899X
    affiliation: 4
  - name: Lukas Fuchs
    orcid: 0000-0002-9165-6384
    affiliation: 5
  - name: Boris J.P. Kaus
    orcid: 0000-0002-0247-8660
    affiliation: 2 # (Multiple affiliations must be quoted)
  - name: Hugo Dominguez
    orcid: 0009-0001-8425-0503
    affiliation: 2

affiliations:
 - name: ETH Zürich, Switzerland
   index: 1
 - name: Johannes Gutenberg-University Mainz, Germany
   index: 2
 - name: École Normale Supérieure - PSL University, France
   index: 3
 - name: University of Lausanne, Switzerland
   index: 4
 - name: Frankfurt University, Germany
   index: 5

date: 21 August 2025
bibliography: paper.bib
---

# Summary

`JustRelax.jl` is an open source, highly portable, and high-performance [Julia](https://julialang.org/) [@bezanson2017julia] package designed for geodynamic modeling. It employs the Accelerated Pseudo-Transient (APT) method —a matrix-free, embarrassingly parallel iterative method— to solve the Stokes and diffusion equations, making it well-suited to exploit Graphics Processing Unit (GPU) hardware accelerators.

`JustRelax.jl` incorporates a wide range of features critical to computational geodynamics. These include complex and highly non-linear rheologies, free surface, and a particle-in-cell method to track and advect material information. Several of the features available in  `JustRelax.jl` are are outsourced to specialized external packages, reducing the core code base, and improving maintainability and reusability.

# Statement of Need

Simulating Earth's thermo-mechanical evolution requires solving coupled, nonlinear Stokes and heat‑diffusion problems across large domains with sharp material contrasts, complex rheologies and long time scales. These calculations are computationally demanding and traditionally rely on monolithic, CPU‑oriented codes built around matrix‑based linear solvers. Such designs are hard to port efficiently to modern heterogeneous HPC systems (e.g. multi‑GPU nodes).

`JustRelax.jl` addresses this gap by providing a compact, modular, GPU‑ready toolkit implemented in Julia. It implements a matrix‑free Accelerated Pseudo‑Transient (APT) solver that reduces global linear‑algebra bottlenecks and ports efficiently to accelerators, while outsourcing local material physics and advection (particle‑in‑cell) to specialized external packages. This keeps the core solver as contained as possible, makes the software more flexible. Julia’s high‑level syntax and package manager simplify installation, scripting and extension, lowering the barrier for students and researchers to prototype and test new physics.

In short, `JustRelax.jl` delivers a high‑performance, portable alternative to legacy geodynamics codes: it (i) exploits modern GPUs without extensive rewrites; (ii) avoids costly matrix assembly via matrix‑free algorithms; and (iii) promotes modularity, reproducibility and rapid development. Together with CI, scalable I/O and checkpointing, these features make `JustRelax.jl` practical for both exploratory studies and production‑scale simulations on modern HPC platforms.

# Methods

`JustRelax.jl` solves the (in)compressible Stokes equations, described by the equations of conservation of momentum and mass, as well as the conservation of energy equation. This system of equations is solved using the APT method [@Raess2022], which transforms the elliptic PDEs into damped wave equations by augmenting them with a second order pseudo-time derivative that includes a damping term. These pseudo-time derivative should vanish upon convergence, thus recovering the original form of the PDE. For an in-depth description of this method, we refer the reader to @Raess2022.

# Package summary

`JustRelax.jl` features:

- **High-performance and scalable matrix-free solver**: The package implements a matrix-free APT method for (in)compressible Stokes and diffusion problems to circumvent the need for computationally expensive linear algebra operations and direct solvers, significantly improving computational efficiency for large-scale simulations. The embarrassingly parallel nature of the APT method makes it an excellent solver to exploit hardware accelerators such as GPUs. The weak scaling curve of the 3D Stokes solver is shown in Fig. \ref{fig:scaling}, where the parallel efficiency is the wall-time of any simulation normalized against the wall-time of a simulation with a single process. Distributed parallelism across multiple CPU/GPU nodes is achieved with ImplicitGlobalGrid.jl [@Omlin2024].

![GPU weak scaling performance of `JustRelax.jl` of the three-dimensional backend, demonstrating efficient parallelization and scalability of the Stokes solver. \label{fig:scaling}](figs/efficiency_Stokes3D.png){width=50%}

- **Advanced non-linear rheology**: The package supports a comprehensive suite of geologically relevant rheological models, such as visco-elasto-plastic, and non-Newtonian constitutive laws. This allows for simulations with complex Earth-like materials that are essential for modeling geological processes at a wide range of scales. The material physics calculations that are local to a single grid point are computed in the external package [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl).

- **Portability**: `JustRelax.jl` is designed to run efficiently on multiple hardware architectures, including CPUs, GPUs (CUDA and AMD), and on multi-node clusters. This portability is achieved through Julia's advanced meta-programming capabilities, which generate the code for the specific target hardware during compile or parse time. This abstraction of the hardware backend is implemented in ParallelStencil.jl [@Omlin2024]. 

- **Modular design**: `JustRelax.jl` is designed with a highly modular design that separates physics, numerics, and I/O components. This allows users to extend the code with, for example, custom rheological models, advection schemes, or other tools without having to modify the core solver infrastructure, facilitating both research flexibility and code maintenance. The core dependencies of `JustRelax.jl` are shown in Fig. \ref{fig:dependencies}.

![Core dependencies of `JustRelax.jl`. \label{fig:dependencies}](figs/dependencies.png){width=75%}

- **Continuous integration (CI) and testing**: The package is tested and validated against a suite of benchmarks and model examples to ensure correctness and performance on various hardware. The default CI/CD pipeline is implemented using GitHub Actions, automatically running tests on every commit and pull request. Single GPU CI is run on using JuliaGPU Buildkite and multi-GPU CI executes on the Swiss National Supercomputing Centre (CSCS) ALPS supercomputer. This ensures that the package remains robust and reliable across different hardware architectures and Julia versions.
  
# Examples

An extensive set of benchmarks and model examples are available in the GitHub repository of [`JustRelax.jl`](https://github.com/PTsolvers/`JustRelax.jl`). Some examples such as [shear band localization](https://ptsolvers.github.io/`JustRelax.jl`/dev/man/ShearBands), [2D subduction](https://ptsolvers.github.io/`JustRelax.jl`/dev/man/subduction2D/subduction2D) or the rise of a [3D plume](https://ptsolvers.github.io/`JustRelax.jl`/dev/man/plume3D/plume3D) are described in the [documentation](https://ptsolvers.github.io/`JustRelax.jl`/dev/). Here we limit ourselves to show some snapshots of the results of these examples in Fig. \ref{fig:examples}.

![Model examples from the documentation: a) 2D shear band localization of a visco-elasto-viscoplastic body ($10240 \times 10240$ cells), b) 2D subduction ($512 \times 512$ cells), and c) raising of a hot plume in 3D ($128 \times 128 \times 128$ cells). All models were run on one NVIDIA GH200 Grace Hopper GPU. \label{fig:examples}](figs/models_JOSS.png)

# Acknowledgments
We acknowledge funding by the Swiss Platform for Advanced Scientific Computing (PASC) as part of the [GPU4GEO & $\partial$GPU4GEO](https://gpu4geo.org/) project, and the European Research Council through the MAGMA project, ERC Consolidator Grant #771143.

# References
