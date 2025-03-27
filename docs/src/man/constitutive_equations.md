# Constitutive equation

In the more general case, JustRelax.jl implements a elasto-visco-elastoplastic rheology. Where the constitutive equation is:

$\begin{align}
\boldsymbol{\dot\varepsilon} =
\boldsymbol{\dot\varepsilon}^{\text{viscous}} +
\boldsymbol{\dot\varepsilon}^{\text{elastic}} +
\boldsymbol{\dot\varepsilon}^{\text{plastic}}
\end{align}$

or

$\begin{align}
\boldsymbol{\dot\varepsilon} =
\frac{1}{2\eta_{\text{eff}}}\boldsymbol{\tau} +
\frac{1}{2G} \frac{D\boldsymbol{\tau}}{Dt}  + \dot\lambda\frac{\partial Q}{\partial \boldsymbol{\tau}_{II}}
\end{align}$

where $\eta_{\text{eff}}$ is the effective viscosity, $G$ is the elastic shear modulus, $\dot\lambda$ is the plastic multiplier, and $Q$ is the plastic flow potential is:

$\begin{align}
Q = \boldsymbol{\tau}_{II} - p\sin{\psi}
\end{align}$

where $\psi$ is the dilation angle.

## Effective viscosity

The effective viscosity of a non-Newtonian Maxwell body is defined as:

$\begin{align}
\eta_{\text{eff}} = \frac{1}{\frac{1}{\eta^{\text{diff}}} + \frac{1}{\eta^{\text{disl}}}}
\end{align}$

where $\eta^{\text{diff}}$ and $\eta^{\text{disl}}$ are the diffusion and dislocation creep viscosities. These are computed from their respective strain rate equations:

$\begin{align}
\dot{ε}_{II}^{\text{diff}} = A^{\text{diff}} τ_{II}^{n^{\text{diff}}} d^{p} f_{H_2O}^{r^{\text{diff}}} \exp \left(- {{E^{\text{diff}} + PV^{\text{diff}}} \over RT} \right) \\
\dot{ε}_{II}^{\text{disl}} = A^{\text{disl}} τ_{II}^{n^{\text{disl}}} f_{H_2O}^{r^{\text{disl}}} \exp \left(- {{E^{\text{disl}} + PV^{\text{disl}}} \over RT} \right)
\end{align}$

where $A$ material specific parameter, $n$ is the stress powerlaw exponent, $p$ is the negative defined grain size exponent, $f$ is the water fugacity, $r$ is the water fugacity exponent, $E$ is the activation energy, $PV$ is the activation volume, and $R$ is the universal gas constant.
## Elastic stress

### Method (1): Jaumann derivative
$\begin{align}
\frac{D\boldsymbol{\tau}}{Dt} =
\boldsymbol{v}\frac{\partial\boldsymbol{\tau}}{\partial t} +
\boldsymbol{\omega}\boldsymbol{\tau} -
\boldsymbol{\tau}\boldsymbol{\omega}^T
\end{align}$

where $\boldsymbol{\omega}$ is the vorticity tensor

$\begin{align}
\boldsymbol{\omega} =
\frac{1}{2} \left(\nabla\boldsymbol{v} - \nabla^T \boldsymbol{v} \right)
\end{align}$

### Method (2): Euler-Rodrigues rotation

1. Compute unit rotation axis
$\begin{align}
    \alpha = \sqrt{\boldsymbol{\omega} \cdot \boldsymbol{u}} \\
    \boldsymbol{n} = \alpha \boldsymbol{I} \frac{1}{\boldsymbol{\omega}}
\end{align}$

where $\boldsymbol{u}$ is a unit vector of length $\mathbb{R}^n$.

2. Integrate rotation angle

$\begin{align}
    \theta = \frac{\alpha}{2\Delta t}
\end{align}$

3. Euler-Rodrigues rotation matrix
$\begin{align}
    \boldsymbol{c} = \boldsymbol{n}\sin{\theta} \\
    \boldsymbol{R_1} =
    \begin{bmatrix}
        \cos{\theta}\sin{\theta} & -c_3 &  c_2 \\
        c_3 &  \cos{\theta}\sin{\theta}  & -c_1 \\
       -c_2 &  c_1 &  \cos{\theta}\sin{\theta} \\
    \end{bmatrix} \\
    \boldsymbol{R_2} = (1-\cos{\theta}) \boldsymbol{n} \boldsymbol{n}^T \\
    \boldsymbol{R}=\boldsymbol{R_1}+\boldsymbol{R_2}
\end{align}$

4. Rotate stress tensor

$\begin{align}
    \boldsymbol{\tau}^{\star}=\boldsymbol{R}\boldsymbol{\tau}\boldsymbol{R}^T
\end{align}$

## Plastic formulation

JustRelax.jl implements the regularised plasticity model from [Duretz et al 2021](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GC009675). In this formulation, the yield function is given by

$\begin{align}
F = \tau_{II} - \left( P \sin{\phi} + C \cos{\phi} + \dot\lambda \eta_{\text{reg}}\right) \geq 0
\end{align}$

where $\eta_{\text{reg}}$ is a regularization term and

$\begin{align}
\dot\lambda = \frac{F^{\text{trial}}}{\eta + \eta_{\text{reg}} + K \Delta t \sin{\psi}\sin{\phi}}
\end{align}$

Note that since we are using an iterative method to solve the APT Stokes equation, the non-linearities are dealt by the iterative scheme. Othwersie, one would need to solve a non-linear problem to compute $\dot\lambda$, which requires to compute $\frac{\partial F}{\partial \dot\lambda}$

# Selecting the constitutive model in JustRelax.jl

All the local calculations corresponding to the effective rheology are implemented in GeoParams.jl. The composite rheology is implemented using the `CompositeRheology` object. An example of how to set up the a visco-elasto-viscoplastic rheology is shown below:

```julia
# elasticity
el    = ConstantElasticity(;
    G = 40e9, # shear modulus [Pa]
    ν = 0.45, # Poisson coefficient
)
# Olivine dislocation law from Hirth and Kohlstedt 2003
disl_wet_olivine  = SetDislocationCreep(Dislocation.wet_olivine1_Hirth_2003)
# Olivine diffusion law from Hirth and Kohlstedt 2003
diff_wet_olivine  = SetDiffusionCreep(Diffusion.wet_olivine_Hirth_2003)
# plasticity
ϕ     = 30   # friction angle
Ψ     = 5    # dilation angle
C     = 1e6  # cohesion [Pa]
η_reg = 1e16 # viscosity regularization term [Pa s]
pl    = DruckerPrager_regularised(; C = C, ϕ = ϕ, η_vp=η_reg, Ψ=Ψ)
# composite rheology
rheology = CompositeRheology(
    (el, disl_wet_olivine, diff_wet_olivine, pl)
)
```

`rheology` then needs to be passed into a `MaterialParams` object with the help of `SetMaterialParams`:

```julia
rheology = (
    SetMaterialParams(;
        Phase             = 1,
        CompositeRheology = rheology,
        # other material properties here
    ),
)
```

### Multiple rheology phases

It is common in geodynamic models to have multiple rheology phases. In this case, we just need to build a tuple containing every single material phase properties:
```julia
rheology = (
    SetMaterialParams(;
        Phase             = 1,
        CompositeRheology = rheology_one,
        # other material properties here
    ),
        SetMaterialParams(;
        Phase             = 2,
        CompositeRheology = rheology_two,
        # other material properties here
    ),
)
```

### Computing the effective viscosity

The effective viscosity is computed internally during the Stokes solver. However, it can also be computed externally with the `compute_viscosity!` function as follows:

```julia
# Rheology
args             = (T=T, P=P, dt = dt) # or (T=thermal.Tc, P=stokes.P, dt=dt)
viscosity_cutoff = (1e18, 1e23)
compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
```

where `T` and `P` are the temperature and pressure fields defined at the **cell centers**, `dt` is the time step, and `phase_ratios` is an object containing the phase ratios corresponding to each material phase, of each cell.

### Elastic stress rotation

The elastic stress rotation is done on the particles.

1. Allocate stress tensor on the particles:
```julia
pτ = StressParticles(particles)
```

2. Since JustPIC.jl requires the fields to be defined at the cell vertices, we need to allocate a couple of buffer arrays where we will interpolate the normal compononents of the stress tensor:
```julia
τxx_v = @zeros(ni.+1...)
τyy_v = @zeros(ni.+1...)
```

3. During time stepping:
```julia
# 1. interpolate stress back to the grid
stress2grid!(stokes, pτ, xvi, xci, particles)
# 2. solve Stokes equations....
#
# 3. rotate stresses
rotate_stress!(pτ, stokes, particles, xci, xvi, dt)
# 4. advection step
    # advect particles in space
advection!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
    # advect particles in memory
move_particles!(particles, xvi, particle_args)
    # check if we need to inject particles
    # need stresses on the vertices for injection purposes
center2vertex!(τxx_v, stokes.τ.xx)
center2vertex!(τyy_v, stokes.τ.yy)
inject_particles_phase!(
        particles,
        pPhases,
        pτ,
        (τxx_v, τyy_v, stokes.τ.xy, stokes.ω.xy),
        xvi
)
```

Note that `rotate_stress!` rotates the stress tensor using the Euler-Rodrigues rotation matrix.
