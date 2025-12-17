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
    affiliation: 2
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

date: 21 October 2025
bibliography: paper.bib
---

# Summary

`JustRelax.jl` is an open-source, highly portable, and high-performance [Julia](https://julialang.org/) [@bezanson2017julia] package designed for geodynamic modeling. It employs the Accelerated Pseudo-Transient (APT) [@Raess2022] method to solve the Stokes and diffusion equations, making it well-suited to exploit Graphics Processing Units (GPUs).

`JustRelax.jl` incorporates a wide range of features critical to computational geodynamics, including complex and highly non-linear rheologies, free surface, and a particle-in-cell method to advect material information. Several of the features available in  `JustRelax.jl` are outsourced to specialized external packages, reducing the core code base, and improving maintainability and reusability.

# Statement of Need

<!-- Simulating Earth's thermo-mechanical evolution requires solving coupled, non-linear Stokes and heat‑diffusion problems across large domains with sharp material contrasts, complex rheologies, and long time scales. These calculations are computationally demanding and traditionally rely on monolithic, CPU‑oriented codes built around matrix‑based linear solvers. Such designs are hard to port efficiently to modern heterogeneous HPC systems (e.g. multi‑XPU nodes). -->

Simulating Earth's thermo-mechanical evolution requires solving coupled, non-linear Stokes and heat‑diffusion problems across large domains with sharp material contrasts, complex rheologies, and long time scales. These calculations are computationally demanding and traditionally rely on CPU‑oriented codes often built around matrix‑based linear solvers (e.g. I3ELVIS [@gerya2015plate], StagYY [@stagyy], [CITCOM-S](https://github.com/geodynamics/citcoms) [@citcom], [Underworld](https://github.com/underworldcode/underworld2) [@underworld]), with designs that may be hard to port efficiently to modern heterogeneous HPC systems (e.g. multi‑XPU nodes). While other geodynamics code, such as [ASPECT](https://github.com/geodynamics/aspect) [@aspect], [TerraNeo](https://terraneo.fau.de/), and [LaMEM](https://github.com/UniMainzGeo/LaMEM/) [@lamem2], do support some matrix-free solvers, they remain CPU-oriented codes.

With `JustRelax.jl` we aim to provide a compact, modular, and highly portable (CPU/GPU) toolkit written in Julia. It implements a matrix‑free Accelerated Pseudo‑Transient (APT) solver that reduces global linear‑algebra bottlenecks and can be ported efficiently to hardware accelerators, while outsourcing some physical calculations, parallelization, and I/O to other packages, as shown in \ref{fig:dependencies}. This keeps the core solver as contained as possible and makes the software more flexible. Julia’s high‑level syntax and package manager simplify installation and scripting, lowering the barrier for students and researchers to prototype and test new physics.

![Sketch of the main dependencies of `JustRelax.jl`: a) physics packages: material properties and rheology ([GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) [@GeoParams]), Particles-in-Cell based advection ([JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl) [@JustPIC], where data structures are handled by [CellArrays.jl](https://github.com/omlins/CellArrays.jl)); b) parallelization: domain decomposition and distributed parallelism are handled by [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) [@Omlin2024] and [MPI.jl](https://github.com/JuliaParallel/MPI.jl) [Byrne2021], and backend abstraction is implemented with [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) [@Omlin2024], which supports Julia's native multi-threading ([Threads.jl](https://docs.julialang.org/en/v1/manual/multi-threading/)), and NVIDIA ([CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) [besard2018juliagpu]) and [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) GPUs; and c) I/O and data visualization: HDF5 binary files ([HDF5.jl](https://github.com/JuliaIO/HDF5.jl) and [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)), and VTK files for visualization ([WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl) [@WriteVTK]). \label{fig:dependencies}](figs/dependencies.png){width=75%}

In short, `JustRelax.jl` delivers a high‑performance, portable alternative to legacy geodynamics codes: it (i) exploits modern GPUs without having to write device-specific code; (ii) avoids costly matrix assembly and memory limits in matrix-vector multiplication through matrix‑free algorithms; and (iii) promotes modularity, reproducibility, and rapid development. Together with CI, scalable I/O and checkpointing, these features make `JustRelax.jl` practical for both exploratory studies and production‑scale simulations on modern multi-XPU platforms.

# Methods

`JustRelax.jl` solves the (in)compressible Stokes equations, described by the equations of conservation of momentum and mass, as well as the conservation of energy equation. This system of equations is solved using the APT method [@Raess2022], which transforms the PDEs into damped wave equations by augmenting them with a second-order pseudo-time derivative, which should vanish upon convergence, thus recovering the original form of the PDE. For an in-depth description of this method, we refer the reader to @Raess2022.

# Package summary

`JustRelax.jl` features:

- **High-performance and scalable matrix-free solver**: `JustRelax.jl` implements the APT method for (in)compressible Stokes and diffusion problems to circumvent the need for computationally expensive linear algebra operations and direct solvers, significantly improving computational efficiency for large-scale simulations. The embarrassingly parallel nature of the APT method makes it an excellent solver to exploit hardware accelerators. The weak scaling curve of the 3D Stokes solver is shown in Fig. \ref{fig:scaling}, where the parallel efficiency is the wall-time of any simulation normalized by the wall-time of a single process simulation ($t_{\text{parallel}}/t_{\text{series}}$). Distributed parallelism across multiple CPU/GPU nodes is achieved with [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) [@Omlin2024].

![GPU weak scaling performance of `JustRelax.jl` of the three-dimensional backend, demonstrating efficient parallelization and scalability of the Stokes solver. \label{fig:scaling}](figs/efficiency_Stokes3D.png){width=50%}

- **Advanced non-linear rheology**: The package supports a comprehensive suite of geologically relevant rheology models, such as visco-elasto-plastic and non-Newtonian constitutive laws, allowing for simulations with complex Earth-like materials, essential for modeling geological processes at a wide range of scales. All the local material physics calculations are computed by [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl) [@GeoParams].

- **Portability**: `JustRelax.jl` is designed to run efficiently on multiple hardware architectures, including CPUs, GPUs (CUDA and AMD), and on multi-node clusters. This portability is achieved through Julia's advanced meta-programming capabilities, which generate the code for the specific target hardware at compile or parse time. This abstraction of the hardware backend is implemented in [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) [@Omlin2024].

- **Continuous integration (CI) and testing**: The package is tested and validated against a suite of benchmarks and model examples to ensure correctness and performance on various hardware. The default CI/CD pipeline is implemented using GitHub Actions, automatically running tests on every commit and pull request. Single GPU CI is run on JuliaGPU Buildkite and multi-GPU CI executes on the Swiss National Supercomputing Centre (CSCS) ALPS supercomputer. This ensures that the package remains robust and reliable across different hardware architectures and Julia versions.
  
# Examples

An extensive set of benchmarks and model examples are available in the GitHub repository of [`JustRelax.jl`](https://github.com/PTsolvers/JustRelax.jl). Some examples such as [shear band localization](https://ptsolvers.github.io/JustRelax.jl/dev/man/ShearBands), [2D subduction](https://ptsolvers.github.io/JustRelax.jl/dev/man/subduction2D/subduction2D), or the rise of a [3D plume](https://ptsolvers.github.io/JustRelax.jl/dev/man/plume3D/plume3D), are described in the [documentation](https://ptsolvers.github.io/JustRelax.jl/dev/). Here, we limit ourselves to showing some snapshots of the results of these examples in Fig. \ref{fig:examples}.

![Model examples from the documentation: a) Second invariant of the plastic strain tensor ($\text{E}_{\text{II}}^{\text{pl}}$) heatmap of 2D shear band localization of a visco-elasto-viscoplastic body ($10240 \times 10240$ cells), b) 2D subduction ($512 \times 512$ cells), and c) rise of a hot plume in 3D ($128 \times 128 \times 128$ cells). All models were run on one NVIDIA GH200 Grace Hopper GPU. \label{fig:examples}](figs/models_JOSS.png)

# Acknowledgments
We acknowledge funding from the Swiss Platform for Advanced Scientific Computing (PASC) as part of the [GPU4GEO & $\partial$GPU4GEO](https://gpu4geo.org/) project, and the European Research Council through the MAGMA project, ERC Consolidator Grant #771143.

# References
