---
title: 'JustRelax.jl: A Julia package for geodynamic modeling with matrix-free solvers'
tags:
  - julia
  - geosciences
  - geodynamics
  - tectonics
  - geophysics
  - computational geosciences
authors:
  - name: Albert de Montserrat
    orcid: 0000-0003-1694-3735
    affiliation: 1
  - name: Pascal Aellig
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

date: 14 March 2024
bibliography: joss.bib
---

# Summary

JustRelax.jl is a open source, highly portable, and high-performance Julia package designed for geodynamic modeling. 
It employs the Accelerated Pseudo-Transient (APT) method to solve the compressible Stokes and heat diffusion equations to simulate complex geodynamic processes. The APT is a matrix-free and embarrassingly parallel iterative method, making it especially well-suited to exploit Graphical Processing Unit (GPU) hardware accelerators available in distributed systems of modern HPC centers.

JustRelax.jl incorporates a wide range of features critical to computational geodynamics. These include highly non-linear rheologies, free surface and a particles-in-cell method to track and advect material phases and properties. Several of these features that JustRelax.jl is built upon are outsourced to smaller specialized packages. This modular design considerably reduces the size of the core code base and enhances the maintenance of the whole package infrastructure. This also improves code reusability, as the external packages can be re-used either as stand-alone packages or as building blocks for some other package.

At last, JustRelax.jl, and most of its core dependencies, are fully written in the [Julia programming language](https://julialang.org/), which lowers the entry bar for users without extensive experience in programming, compared to other compiled languages common in HPC-focused software (C/C++/Fortran). The interactive environment of Julia also, similar to Python and MATLAB, lets the users prototype and implement new code in the same language, improving the quality of life of the whole developing experience.

<!-- JustRelax.jl is a high-performance Julia package for geodynamic modeling that implements matrix-free solvers based on the Accelerated Pseudo-Transient (APT) method. The package provides efficient and scalable solutions for simulating complex geophysical processes across multiple spatial and temporal scales, from crustal deformation to mantle dynamics. JustRelax.jl solves the coupled system of Stokes equations for compressible flow and heat diffusion equations with non-linear rheologies, enabling realistic simulations of Earth's lithosphere and mantle.

A key feature of JustRelax.jl is portability, being able to run on diverse operative systems (Windows, MacOS, and any Linux distro supported by the Julia programming language) computing architectures, including workstation-level CPUs, GPUs from multiple vendors, and High Performance Computing (HPC) clusters, without requiring code modifications and with a seamless installation. This portability is achieved through Julia's metaprogramming capabilities, which can automatically generate hardware-specific code at parse time. The matrix-free nature of the APT approach eliminates the need for expensive and not-so-scalable linear algebra operations, making it particularly suitable for large-scale simulations that are typically memory-bound on modern computing architectures.

The package implements a wide range of geological features essential for geodynamic modeling, including a complex and highly non-linear rheologies, a free-surface, and an efficient and scalable marker-in-cell methods for tracking and advecting material properties. JustRelax.jl is designed with a modular architecture that allows researchers to easily customize parts of the code such as rheological models, material models, or model setup generation without modifying the core package infrastructure, as these features are outsourced to other specialized packages. This flexibility, combined with Julia's interactive development environment, enables rapid prototyping and testing of new physical models while maintaining computational efficiency. -->

# Statement of Need

Computational geodynamics is an important tool for simulating and investigating Earth's complex multi-physics process across different temporal and spatial scales. Running these simulations is computationally expensive, requiring numerical methods that can efficiently solve the governing equations that can also handle, at least, large and sharp contrasts of physical properties, and highly non-linear rheologies. In the past decades, the computational geodynamics community has developed an extensive set of software tools that have undoubtedly improved our understanding of Earth's dynamics. Most of these codes employ different implementations of either staggered grid finite difference methods (e.g. **I3ELVIS** [@gerya2015plate], **StagYY** [@stagyy], **LaMEM** [@lamem]) or finite element methods (e.g. **CITCOM-S** [@citcom], **ASPECT** [@aspect], **pTatin3D** [@ptatin], **Underworld** [@underworld]) to solve the Stokes equations. Nonetheless, these existing geodynamic modeling tools face several critical challenges:

1) Most codes are optimized for CPUs and distributed (or hybrid shared-distributed) memory architectures (as in classic CPU-based HPC clusters) and would require substantial modifications to full rewrites to be able to execute on hardware accelerators such as GPUs and exploit the extra computational power. This CPU-only design 
limits the capability of these codes to take advantage of the latest high-performance computing resources, which are evolving rapidly towards multi-GPU systems. Apart from making these codes GPU-friendly, they commonly rely on iterative matrix-based methods which may scale poorly on GPUs, so that it is likely they would require new solution methods with better scaling on this kind of hardware.

2) Traditionally, geodynamic codes are written (or at least their critical bits) in statically compiled languages (to our knowledge, only C/C++ or Fortran), with complex codebases that have evolved over several years or decades. This leads to highly complex codebases with a steep learning curve for new users and developers, in particular for students and those without significant experience in software engineering or basic programming skills. The latter being the most common user in the geodynamics community. The static nature of these languages also limits the flexibility of code development workflows, which often involve rapid prototyping and interactive development in interpreted languages such as MATLAB or Python, instead of prototyping directly in the code base.

3) Often, these codes depend on external libraries for performing linear algebra operations, or I/O, for example. This often leads to the so-called _dependencies hell_, where installation is not so trivial, in particular for newcomers, as compatibility issues, version conflicts, and other difficulties may occur. 

With JustRelax.jl, we intend to address these fundamental limitations by introducing the first HPC-ready geodynamic software written in Julia:

- We leverage Julia's metaprogramming capabilities to generate hardware-specific code. With this, any script written by the user runs efficiently across different architectures and hardware with minimal changes.

- We use a solution method which is particularly well-suited for GPU accelerators. 

- Julia's built-in package manager makes the installation of JustRelax.jl trivial, as it automatically downloads and installs all the required dependencies for any compatible operative system (Windows, MacOs, and most of the mainstream Linux distros).

- Julia is a higher-level language, with a syntax that may be familiat to Python and MATLAB users. This significantly lowers the entry bar to new users, while still offering a performance comparable to static languages.

# Methods

As previously stated, JustRelax.jl employs the APT method to solve the compressible Stokes and heat diffusion equations @rass2022. Within each APT iteration, the pressure-dependent equation of state, non-Newtonian viscosity, plastic stress corrections, and conservation of mass and momentum are solved simultaneously in a fully coupled manner. In the sections below, we briefly summarize the governing equations and their respective APT implementation, and discuss the parallelization strategy and the advection schemes used to track material properties and deformation history.

## Thermo-mechanical modeling
The Stokes equations for compressible viscous flow are:

$$
\begin{align}
    \nabla\cdot\mathbf{\tau} - \nabla p = \mathbf{f} \\
    \nabla\cdot\mathbf{v} + \beta \frac{\partial p}{\partial t} + \alpha \frac{\partial T}{\partial t} = 0
\end{align}
$$

where $\nabla$ is the nabla operator, $\mathbf{\tau}$ is the deviatoric stress tensor, $p$ is the total pressure, $\mathbf{f}$ are the body forces, usually being the buoyancy forces $\mathbf{f} = \rho \mathbf{g}$ where $\rho$ is density and $\mathbf{g} = (0, -g_y)$ is the gravitational force vector, $\mathbf{v}$ is the velocity field, $\beta$ is the inverse of the bulk modulus, and $\alpha$ is the thermal expansivity coefficient. The previous system of equations is closed with the constitutive equation that relates the deviatoric strain tensor $\dot{\mathbf{\varepsilon}}$ with $\mathbf{\tau}$. In the simple case of a linear isotropic rheology, the constitutive equation is

$$
\begin{align}
    \mathbf{\tau} = 2\eta\dot{\mathbf{\varepsilon}}
\end{align}
$$

where $\eta$ is the viscosity. Since the evolution of the temperature $T$  field is highly relevant in geodynamics, and most of the material properties such as density and rheology laws are also temperature dependent, we also solve the equation of conservation of energy:

$$
\begin{align}
    \rho C_p \frac{\partial T}{\partial t} = - \nabla q + \mathbf{\tau}:\mathbf{\dot\varepsilon} + \alpha T (\mathbf{v} \cdot \nabla P) + Q \\
    q = - \kappa \nabla T
\end{align}
$$

where $C_p$ is the heat capacity, $\kappa$ is the heat conductivity, $\mathbf{\tau}:\mathbf{\dot\varepsilon}$ is the shear heating (heating generated by viscous dissipation), $\alpha T (\mathbf{v} \cdot \nabla P)$ is the adiabatic heating, and $Q$ is the sum of any other source terms.

### Pseudo-Transient formulation

The APT method consists of transforming the elliptic PDEs into damped wave equations by augmenting them with a damped pseudo-time derivative. Upon convergence, the pseudo-time derivative term should vanish to machine precision, hence recovering the original PDE. We employ the APT method because it is embarrassingly parallel as it is matrix-free and requires only explicit updates, which also results in a numerical implementation much simpler than other solution methods typically used to solve the Stokes equations (Multigrid, Conjugate Gradients, GMRES...). The APT formulation of the compressible Stokes equations and continuity equation is then:
$$
\begin{align}
    \widetilde{\rho}\frac{\partial \mathbf{v}}{\partial \psi} + \nabla\cdot\mathbf{\tau} - \nabla p = \mathbf{f}  \\
    \frac{1}{\widetilde{K}}\frac{\partial p}{\partial \psi} + \nabla\cdot\mathbf{v} = \beta \frac{\partial p}{\partial t} + \alpha \frac{\partial T}{\partial t} \\
    \frac{1}{2\widetilde{G}} \frac{\partial\mathbf{\tau}}{\partial\psi} + \frac{\mathbf{\tau}}{2\eta} = \dot{\mathbf{\varepsilon}} \\
\end{align}
$$

where the wide tilde denotes the effective damping coefficients and $\psi$ is the pseudo-time step. These are defined as in [@Raess2022]:
$$
\begin{align}
    \widetilde{\rho} = Re\frac{\eta}{\widetilde{V}L}, \qquad
    \widetilde{G} = \frac{\widetilde{\rho} \widetilde{V}^2}{r+2}, \qquad
    \widetilde{K} = r \widetilde{G}
\end{align}
$$
with
$$
\begin{align}
    \widetilde{V} = \sqrt{ \frac{\widetilde{K} +2\widetilde{G}}{\widetilde{\rho}}}, \qquad
    r = \frac{\widetilde{K}}{\widetilde{G}}, \qquad
    Re = \frac{\widetilde{\rho}\widetilde{V}L}{\eta}
\end{align}
$$
where the P-wave $\widetilde{V} = V_p = \widetilde{C} \frac{\Delta x}{\Delta \tau}$ is the characteristic velocity scale, and $Re$ is the Reynolds number. We refer the reader to [@Raess2022] for further details on these damping coefficients. Similarly to the Stokes equations, the PT heat diffusion equation is given by:
$$
\begin{align}
    \widetilde{\rho}\frac{\partial T}{\partial \psi} + \rho C_p \frac{\partial T}{\partial t} = -\nabla q \\
    \widetilde{\theta}\frac{\partial q}{\partial \psi} + q  = -\kappa\nabla T \\
\end{align}
$$

where
$$
\begin{align}
  \widetilde{\rho} = R_e \frac{\kappa}{\widetilde{V} L}, \qquad
  \theta_r = \frac{L}{R_e \widetilde{V}}, \qquad
  Re = \pi + \sqrt{\pi^2 + \frac{\rho C_p}{\kappa}L^2}
\end{align}
$$

The APT equations are solved by discretizing the pseudo-time derivatives, either explicitly or implicitly, and iterating until the residual form of the equations is below a given tolerance. This means that the wave component of the equation has completely damped out and the pseudo-time derivative is approximately zero. In JustRelax.jl, the APT equations are discretized implicitly, as this yields better convergence rates.

### Advection

The advection equation in a decoupled manner using a Particle-in-Cell (PiC) method to advect the temperature, composition, or any other information carried by the particles. This method is particularly well-suited for this task because it can accurately handle the advection of these fields @dominguez2024, and it is extensively used to simulate global and regional scale geodynamic processes, e.g. StaggYY [@stagyy], LaMEM [@lamem], I3ELVIS [@i3elvis], ASPECT [@aspect], amongst others. This is implemented in the publicly available [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl) package, where different time integrators and velocity interpolation schemes are available.

### Parallelization

The APT method in JustRelax.jl is parallelised in a hybrid shared-distributed memory architecture manner, i.e. multithreading combined with MPI for CPUs, and GPU-aware MPI for multi-GPU architectures. Parallelisation is implemented with two Julia packages:

- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) [Omlin2024] handles the backend abstraction, allowing the user to write (shared memory) parallel kernels that seamlessly run on CPUs and GPUs (currently supported: CUDA, AMDGPU, and Mac M-series chips). It automatically generates Julia code based on Base.Threads.jl (Julia's native multi-threading library) and the third-party GPU libraries CUDA.jl (Nvidia GPUs) [@CUDA.jl], AMDGPU.jl (AMD GPUs), and Metal.jl (Mac M-series chips) depending on the device backend of choice of the end-user.

- [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) handles Message Passing Interface (MPI) domain decomposition and communication, distributing the computational domain across multiple (CPU/GPU) compute nodes. MPI communication is handled at the lowest level by MPI.jl, the Julia wrapper of the MPI standard.

# Package summary

JustRelax.jl features:

- **High-performance matrix-free solver**: The package implements a matrix-free APT method for Stokes and heat diffusion problems to circumvent the need for computationally expensive linear algebra operations and direct solvers, significantly improving computational efficiency for large-scale simulations. The embarrassingly parallel nature of the APT method makes it an excellent solver to exploit hardware accelerators such as GPUs. 

<!-- The weak scaling curve of the 3D Stokes solver is shown in Fig. \ref{fig:scaling}. -->

<!-- <figure>
    <img src="figs/weak_scaling_JR.png"
         alt="weak scaling">
    <figcaption>Figure 1: GPU weak scaling performance of JustRelax.jl in the two- and three-dimensional backends, demonstrating efficient parallelization and scalability for large-scale geodynamic simulations</figcaption>
</figure> -->

<!-- ![GPU weak scaling performance of JustRelax.jl in the two- and three-dimensional backends, demonstrating efficient parallelization and scalability for large-scale geodynamic simulations. \label{fig:scaling}](figs/weak_scaling_JR.png) -->


- **Portability**: JustRelax.jl is designed to run efficiently on multiple hardware architectures, including CPUs, GPUs (CUDA and AMD), and on multi-node clusters. This portability is achieved through Julia's advanced meta-programming capabilities, which generate the code for the specific target hardware during compile or parse time. This abstraction of the hardware backend is implemented in the ParallelStencil.jl package. In the code snippet below we show how one can switch the backend to be used by simply switching on or off the `isCUDA` flag, defined at the top-level or any given script:

```julia
const isCUDA = true # or `false` to use the CPU backend
# conditional loading of CUDA.jl
@static if isCUDA
    using CUDA
end
# load JustRelax.jl and its 2D backend
using JustRelax, JustRelax.JustRelax2D
# define the `backend` variable that is passed to instantiate JustRelax.jl
# objects on the correct hardware backend.
# Options:
#     CPUBackend, CUDABackend, AMDGPUBackend
const backend = @static if isCUDA
    CUDABackend
else
    JustRelax.CPUBackend
end
```

- **Advanced non-linear rheology**: The package supports a comprehensive suite of geologically relevant rheological models, including visco-elastic, visco-elasto-plastic, and non-Newtonian temperature- and pressure-dependent constitutive laws. This allows for simulations with complex Earth-like materials that are essential for modeling geological processes at a wide range of scales. All the material physics calculations that are local to a single grid point are computed in the external package [GeoParams.jl](https://github.com/JuliaGeodynamics/GeoParams.jl).

- **Particles-in-cell method**: The package employs an efficient particles-in-cell approach for tracking material properties and deformation history. This method prevents numerical diffusion of compositional boundaries and allows for accurate representation of complex geological structures, rheological interfaces, and strain localization features. This is implemented in [JustPIC.jl](https://github.com/JuliaGeodynamics/JustPIC.jl).

- **Free-surface implementation**: JustRelax.jl uses a variational Stokes [@larionov2017variational] approach to implement a free-surface boundary condition. This feature is critical for modeling surface processes such as mountain building, rifting, and subduction, where the interaction between internal deformation and surface topography plays a fundamental role in the tectonic evolution.

- **Modular architecture**: JustRelax.jl is designed with a highly modular design that separates physics, numerics, and I/O components. This architecture allows users to extend the code with, for example, custom rheological models, advection schemes, or I/O tools without having to modify the core solver infrastructure, facilitating both research flexibility and code maintenance. The core dependencies of JustRelax.jl are shown in Fig. \ref{fig:dependencies}.

<!-- <figure>
    <img src="figs/dependencies.png"
         alt="dependencies">
    <figcaption> Figure 2: Main Julia dependencies of JustRelax.jl. </figcaption>
</figure> -->

![Main Julia dependencies of JustRelax.jl. \label{fig:dependencies}](figs/dependencies.png)

<!-- - **Distributed I/O support**: The package implements efficient parallel input/output operations for handling large datasets common in 3D geodynamic simulations. This includes parallel writing and reading of solution fields, checkpoint/restart capabilities, and compatibility with standard visualization formats (VTK) for scientific data analysis and post-processing. -->

- **Interactive development environment**: As a Julia package, JustRelax.jl takes full advantage of the language's dynamic nature, allowing for interactive sessions, real-time debugging, and rapid prototyping. This significantly enhances the quality of life of the users and developers compared to traditional compiled languages, while maintaining high performance thanks to Julia's JIT/JAOT compilation.

# Examples

An extensive set of benchmarks and model examples are available in the GitHub repository of [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl). More complex examples such as [2D subduction](https://ptsolvers.github.io/JustRelax.jl/dev/man/subduction2D/subduction2D) or the rise of a [3D plume](https://ptsolvers.github.io/JustRelax.jl/dev/man/plume3D/plume3D) are described in the [documentation](https://ptsolvers.github.io/JustRelax.jl/dev/). Here we limit ourselves to show some snapshots of the results of these examples in Fig. \ref{fig:examples}.

![Model examples from the documentation: a) 2D subduction, and b) raising of a hot plume in 3D. \label{fig:examples}](figs/models_JOSS.png)


<!-- <figure>
    <img src="figs/models_JOSS.png"
         alt="models">
    <figcaption> Figure 3: Model examples from the documentation: a) 2D subduction, and b) raising of a hot plume in 3D. </figcaption>
</figure> -->


# Acknowledgments
We acknowledge funding by the Swiss Platform for Advanced Scientific Computing (PASC) as part of the [GPU4GEO](https://gpu4geo.org/) project, and the European Research Council through the MAGMA project, ERC Consolidator Grant #771143.

# References
