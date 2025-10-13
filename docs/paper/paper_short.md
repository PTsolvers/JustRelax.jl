---
title: 'JustRelax.jl: A Julia package for geodynamic modeling with matrix-free solvers'
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

JustRelax.jl is an open source, highly portable, and high-performance Julia package designed for geodynamic modeling. It employs the Accelerated Pseudo-Transient (APT) method to solve the Stokes and diffusion equations to simulate complex geodynamic processes. The APT is a matrix-free and embarrassingly parallel iterative method, making it especially well-suited to exploit Graphics Processing Unit (GPU) hardware accelerators available in distributed systems of most HPC centers.

JustRelax.jl incorporates a wide range of features critical to computational geodynamics. These include complex and highly non-linear rheologies, free surface and its good spatial locality (compared to e.g. matrix-matrix or matrix-vector multiplication) particle-in-cell method to track and advect material phases and properties. Several of the features JustRelax.jl is built upon are outsourced to specialized packages. This modular design considerably reduces the size of the core code base, making the entire package infrastructure easier to maintain. This also improves code reusability, as the external packages can be re-used either as stand-alone packages or as building blocks for some other package.

Most of JustRelax.jl's core dependencies are fully written in the [Julia programming language](https://julialang.org/) [@bezanson2017julia], which facilitates contributions from users without extensive experience in programming, compared to other compiled languages common in HPC-focused software (C/C++/Fortran). The interactive environment of Julia allows users to prototype and implement new code in a single high-level language, improving the overall development experience in similar fashion to Python and MATLAB.

# Statement of Need

Computational geodynamics is an important tool for simulating and investigating Earth's complex multi-physics processes across different temporal and spatial scales. Running these simulations is computationally expensive which requires numerical methods that can efficiently solve the governing equations and resolving large and sharp contrasts in physical properties, and highly non-linear rheologies. In the past decades, the computational geodynamics community has developed an extensive set of software tools that have undoubtedly improved our understanding of Earth's dynamics. Most of these codes employ different implementations of either staggered grid finite difference methods (e.g. I3ELVIS [@gerya2015plate], StagYY [@stagyy], [LaMEM](https://github.com/UniMainzGeo/LaMEM/) [@lamem]) or finite element methods (e.g. [CITCOM-S](https://github.com/geodynamics/citcoms) [@citcom], [ASPECT](https://github.com/geodynamics/aspect) [@aspect], [pTatin3D](https://bitbucket.org/ptatin/ptatin3d/src/master/) [@ptatin], [Underworld](https://github.com/underworldcode/underworld2) [@underworld]) to solve the Stokes equations. Nonetheless, these existing geodynamic modeling tools face several critical challenges:

- Most codes are optimized for CPUs and distributed memory architectures, and would require from substantial-modifications to full-rewrites to be able to execute on hardware accelerators such as GPUs. This CPU-only design limits the capability of these codes to take advantage of the latest high-performance computing resources, which are evolving rapidly towards multi-GPU systems. Beyond offloading parts of the computational routines onto GPUs, existing codes mostly build upon matrix-based solvers which may scale poorly on GPUs. It is likely that different solution strategies may need to be implemented in order to deliver better scaling on GPUs and accelerators.

- Traditionally, geodynamic codes are written (or at least their critical bits) in statically compiled languages (to our knowledge, only C/C++ or Fortran), with very large and complex codebases that have evolved over several years or decades, which poses a steep learning curve for new users and developers. In particular for students and those without significant experience in software engineering or even basic programming skills. The static nature of these languages also limits the flexibility of code development workflows, which often involve rapid prototyping and interactive development in other interpreted languages.

With JustRelax.jl, we intend to address these fundamental limitations by introducing the first HPC-ready geodynamic software fully written in Julia:

- We leverage Julia's metaprogramming capabilities to generate hardware-specific code. With this, any script written by the user runs efficiently across different architectures and hardware with minimal changes.

- We use a solution method which is particularly well-suited for GPU accelerators.

- Compared to some legacy codes written in C/C++/Fortran, Julia's built-in package manager makes the installation of JustRelax.jl trivial, as it automatically downloads and installs all the required dependencies for any compatible operating system.

- Julia is a higher-level language, with a syntax that may be familiar to Python and MATLAB users. This significantly facilitates contributions from users, while offering a performance comparable to statically compiled languages.

# Methods

JustRelax.jl solves the compressible Stokes equations, given by the conservation of momentum and mass equations, as well as the conservation of energy equation, (e.g. [@lamem], [@aspect]). These system of equations is solved using the APT method [@Raess2022], which transforms the elliptic PDEs into damped wave equations by augmenting them with a second order pseudo-time derivative including a damping term. These pseudo-time derivative should vanish upon convergence, thus recovering the original form of the PDE. For an in-depth description of this method, we refer the reader to @Raess2022.

Advection of material fields is solved in a decoupled manner using a Particle-in-Cell (PiC) [@Harlow1965] method to advect information carried by a set of Lagrangian particles, which is implemented in the [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl) package. PiC is particularly well-suited for this task because it can accurately handle the advection of these fields, and it is extensively used to simulate global and regional scale geodynamic processes, e.g. StaggYY [@stagyy], LaMEM [@lamem], I3ELVIS [@i3elvis], ASPECT [@aspect], amongst others.

### Parallelization

The APT method in JustRelax.jl is parallelised using a hybrid shared-distributed memory architecture fashion, mainly leveraging two Julia packages:

- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) [@Omlin2024] handles the backend abstraction, allowing the user to write (shared memory) parallel kernels that seamlessly run on CPUs and GPUs (currently supported: CUDA, AMDGPU, and Mac M-series chips).

- [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) [@Omlin2024] handles Message Passing Interface (MPI) domain decomposition and communication, distributing the computational domain across multiple (CPU/GPU) compute nodes.

# Package summary

JustRelax.jl features:

- **High-performance and scalable matrix-free solver**: The package implements a matrix-free APT method for (in)compressible Stokes and diffusion problems to circumvent the need for computationally expensive linear algebra operations and direct solvers, significantly improving computational efficiency for large-scale simulations. The embarrassingly parallel nature of the APT method makes it an excellent solver to exploit hardware accelerators such as GPUs. The weak scaling curve of the 3D Stokes solver is shown in Fig. \ref{fig:scaling}, where the parallel efficiency is the wall-time of any simulation normalized against the wall-time of a simulation with a single process.

![GPU weak scaling performance of JustRelax.jl of the three-dimensional backend, demonstrating efficient parallelization and scalability of the Stokes solver. \label{fig:scaling}](figs/efficiency_Stokes3D.png){width=50%}

- **Portability**: JustRelax.jl is designed to run efficiently on multiple hardware architectures, including CPUs, GPUs (CUDA and AMD), and on multi-node clusters. This portability is achieved through Julia's advanced meta-programming capabilities, which generate the code for the specific target hardware during compile or parse time. This abstraction of the hardware backend is implemented in the ParallelStencil.jl package. In the code snippet below we show how one can switch the backend to be used by simply switching on or off the `isCUDA` flag, defined at the top-level or any given script:

<!-- ```julia
const isCUDA = true # or `false` to use the CPU backend
# Conditional loading of CUDA.jl
@static if isCUDA
    using CUDA
end
# Load JustRelax.jl and its 2D backend
using JustRelax, JustRelax.JustRelax2D
# Define the `backend` variable that is passed to instantiate JustRelax.jl
# objects on the correct hardware backend. Options:
#     (1) CPUBackend; (2) CUDABackend; (3) AMDGPUBackend
const backend = @static if isCUDA
    CUDABackend
else
    JustRelax.CPUBackend
end
``` -->

- **Advanced non-linear rheology**: The package supports a comprehensive suite of geologically relevant rheological models, including visco-elastic, visco-elasto-plastic, and non-Newtonian temperature- and pressure-dependent constitutive laws. This allows for simulations with complex Earth-like materials that are essential for modeling geological processes at a wide range of scales. All the material physics calculations that are local to a single grid point are computed in the external package [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl).

- **Particle-in-cell method**: The package employs an efficient particle-in-cell approach for tracking material properties and deformation history. This method prevents numerical diffusion of compositional boundaries and allows for accurate representation of complex geological structures, rheological interfaces, and strain localization features. This is implemented in [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl).

- **Modular architecture**: JustRelax.jl is designed with a highly modular design that separates physics, numerics, and I/O components. This architecture allows users to extend the code with, for example, custom rheological models, advection schemes, or I/O tools without having to modify the core solver infrastructure, facilitating both research flexibility and code maintenance. The core dependencies of JustRelax.jl are shown in Fig. \ref{fig:dependencies}.

![Main Julia dependencies of JustRelax.jl. \label{fig:dependencies}](figs/dependencies.png){width=75%}

- **Interactive development environment**: As a Julia package, JustRelax.jl takes full advantage of the language's dynamic nature, allowing for interactive sessions, real-time debugging, and rapid prototyping. This significantly enhances the quality of life of the users and developers compared to traditional compiled languages, while maintaining high performance thanks to Julia's JIT/JAOT compilation.

- **Continuous integration (CI) and testing**: The package is continuously tested and validated against a suite of benchmarks and model examples to ensure correctness and performance. This includes unit tests, regression tests, and performance benchmarks on various hardware configurations. The CI/CD pipeline is implemented using GitHub Actions, which automatically runs tests on every commit and pull request. Additionally, we test the package on its GPU compatibility using Buildkite and the Swiss National Supercomputing Centre (CSCS) GPU cluster Alps.Daint. This ensures that the package remains robust and reliable across different hardware architectures and Julia versions.

# Examples

An extensive set of benchmarks and model examples are available in the GitHub repository of [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl). Some examples such as [shear band localization](https://ptsolvers.github.io/JustRelax.jl/dev/man/ShearBands), [2D subduction](https://ptsolvers.github.io/JustRelax.jl/dev/man/subduction2D/subduction2D) or the rise of a [3D plume](https://ptsolvers.github.io/JustRelax.jl/dev/man/plume3D/plume3D) are described in the [documentation](https://ptsolvers.github.io/JustRelax.jl/dev/). Here we limit ourselves to show some snapshots of the results of these examples in Fig. \ref{fig:examples}.

![Model examples from the documentation: a) 2D shear band localization of a visco-elasto-viscoplastic body ($10240 \times 10240$ cells), b) 2D subduction ($512 \times 512$ cells), and c) raising of a hot plume in 3D ($128 \times 128 \times 128$ cells). All models were run in one  NVIDIA GH200 Grace Hopper GPU. \label{fig:examples}](figs/models_JOSS.png)


# Acknowledgments
We acknowledge funding by the Swiss Platform for Advanced Scientific Computing (PASC) as part of the [GPU4GEO & $\partial$GPU4GEO](https://gpu4geo.org/) project, and the European Research Council through the MAGMA project, ERC Consolidator Grant #771143.

# References
