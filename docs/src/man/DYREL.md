# Using the APT method with auto tuned damping coefficients

> [!WARNING]
> This solver is still work-in-progress/experimental. It is 2D only; a 3D version is coming up soon. Both the standard and the variational (free surface) Stokes problems are supported.

Instead of using the Accelerated Pseudo-Transient where the damping coefficients are constant throughout the PT iterations (as in [Räss et al, 2022](https://gmd.copernicus.org/articles/15/5757/2022/)), we can use a self-tuning version of the APT method based on the approached described in [Duretz et al, 2025](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-5641/).

# Usage

To use this solver, only two changes are needed with respect to the scripts using the APT solver described in previous examples:

1. The `PTStokesCoeffs` object containing the arrays needed for the standard APT solver is not needed anymore, and needs to be replaced by the `DYREL` object that contains all the new arrays that are needed for the self-tuned APT method. This means we need to change this line
```julia
pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-6, ϵ_rel = 1.0e-6, CFL = 0.95 / √2)
```
by this one
```julia
dyrel = DYREL(backend, stokes, rheology, phase_ratios, di, dt; ϵ=1e-6)
```
where `ϵ` is the convergence tolerance and the optional `γfact` (default `20.0`) scales the
Powell-Hestenes penalty as $\gamma_{\text{num}} = \gamma_{\text{fact}} \cdot \eta$. `γfact` is the dominant control on
how many iterations a time step costs, and its optimum is problem- and resolution-dependent, so
it is worth tuning per model.

> [!NOTE]
> Note that the `DYREL` arrays need effective viscosity of the model, so it needs to be instantiated *after* having a effective viscosity guess.

2. The last change requires changing the solver function call to the following:
```julia
solve_DYREL!(
    stokes,
    ρg,
    dyrel,
    flow_bcs,
    phase_ratios,
    rheology,
    args,
    grid,
    dt,
    igg;
    kwargs = (;
        iterMax              = 50.0e3,
        total_iterMax        = 50.0e3,
        nout                 = 10,
        rel_drop             = 0.1,
        λ_relaxation_PH      = 1,
        λ_relaxation_DR      = 1,
        verbose_PH           = false,
        verbose_DR           = false,
        viscosity_relaxation = 1,
        linear_viscosity     = true,
        free_surface         = false,
        viscosity_cutoff     = (-Inf, Inf),
    )
);
```
where `grid` is the `Geometry` object (a tuple of grid spacings `di` is also accepted), and the
solver keyword arguments are:
- `iterMax` $\rightarrow$ maximum number of dynamic relaxation iterations per Powell-Hestenes sweep.
- `total_iterMax` $\rightarrow$ maximum number of dynamic relaxation iterations summed over all the Powell-Hestenes sweeps of one time step. This is what actually caps the cost of a time step, so a step that stops short of `ϵ` has usually hit this rather than `iterMax`.
- `nout` $\rightarrow$ damping coefficients are re-computed every `nout` iterations.
- `rel_drop` $\rightarrow$ the tolerance for the inner dynamic relaxation loop is $error(P^n) \text{rel_drop}$ where $n$ is the inner Powell-Hesteness iteration counter.
- `λ_relaxation_PH` $\rightarrow$ relaxation coefficient for the plastic multiplier ($\cdot\lambda$) during the inner Powell-Hesteness loop. `λ_relaxation_PH=1` means no relaxation.
- `λ_relaxation_DR` $\rightarrow$ relaxation coefficient for the plastic multiplier ($\cdot\lambda$) during the innes Dynamic Relaxation loop. `λ_relaxation_DR=1` means no relaxation.
- `verbose_PH` $\rightarrow$ # print solver metrics during  inner Powell-Hesteness loop.
- `verbose_DR` $\rightarrow$ # print solver metrics during  innes Dynamic Relaxation loop.
- `viscosity_relaxation` $\rightarrow$ relaxation coefficient for the viscosity. `viscosity_relaxation=1` means no relaxation.
- `linear_viscosity` $\rightarrow$ if the rheology is linear (viscosity will not be updated during the solver iterations).
- `free_surface` $\rightarrow$ adds the free-surface stabilization (FSSA) term $V_y \frac{\partial (\rho g)}{\partial y} \Delta t$ to the vertical momentum residual, and puts a lower bound on the penalty so that it stays commensurate with that term.
- `viscosity_cutoff` $\rightarrow$ viscosity is clamped so that $\text{viscosity_cutoff}_1 \leq \eta \leq \text{viscosity_cutoff}_2$.

# Variational (free surface) Stokes

## Formulation

The rock occupies a sub-domain $\Omega$ whose upper boundary $\partial\Omega_s$ is a free
surface, so the Stokes problem is solved on $\Omega$ alone, subject to a traction-free condition
there:

$\begin{align}
\nabla\cdot\boldsymbol{\tau} - \nabla P + \rho \boldsymbol{g} = \boldsymbol{0}, \qquad
\nabla\cdot\boldsymbol{v} = -\frac{P - P^0}{K \Delta t} + \frac{Q}{\Delta t}
\quad \text{in } \Omega
\end{align}$

$\begin{align}
\boldsymbol{\sigma}\cdot\boldsymbol{n} = (\boldsymbol{\tau} - P\boldsymbol{I})\cdot\boldsymbol{n} = \boldsymbol{0}
\quad \text{on } \partial\Omega_s
\end{align}$

Rather than meshing $\Omega$, the variational formulation (Larionov, Batty & Bridson, 2017)
keeps the regular staggered grid and records how much rock each control volume holds. The
`RockRatio` $\phi \in [0, 1]$ stores that fraction at every staggered location — $\phi_c$ at
centers, $\phi_v$ at vertices, $\phi_{V_x}$ and $\phi_{V_y}$ on the faces. Each term of the
momentum balance is then weighted by the fraction belonging to the location it is evaluated at,
which is what confines the equations to $\Omega$:

$\begin{align}
R_x = \delta_x(\phi_c \tau_{xx}) + \delta_y(\phi_v \tau_{xy}) - \delta_x\left(\phi_c \widetilde{P}\right) - \overline{\phi_c \rho g_x}^{\,x}
\end{align}$

$\begin{align}
R_y = \delta_y(\phi_c \tau_{yy}) + \delta_x(\phi_v \tau_{xy}) - \delta_y\left(\phi_c \widetilde{P}\right) - \overline{\phi_c \rho g_y}^{\,y} + \underbrace{\theta \Delta t\, V_y \frac{\partial (\rho g_y)}{\partial y}}_{\text{FSSA}}
\end{align}$

where $\delta_x, \delta_y$ are the staggered differences, $\overline{\;\cdot\;}^{\,x}$ the
interpolation onto the face, and $\widetilde{P} = P + \gamma_{\text{eff}} R_p + \Delta P_\psi$ the
pressure carrying the Powell-Hestenes correction. A control volume with $\phi = 0$ drops out of
every stencil, so the traction-free condition above is recovered without ever locating the surface
explicitly — the discrete equations simply stop at the last cell holding rock.

The continuity residual is imposed on the cells the mask keeps, with the *unweighted* divergence
of the velocities on their faces:

$\begin{align}
R_p = -\nabla\cdot\boldsymbol{v} - \frac{P - P^0}{K \Delta t} + \frac{Q}{\Delta t}
\end{align}$

and the pressure is advanced by the Powell-Hestenes update $P \leftarrow P + \gamma_{\text{eff}} R_p$,
with the penalty combining a numerical and a physical contribution harmonically:

$\begin{align}
\gamma_{\text{eff}} = \frac{\gamma_{\text{num}} \gamma_{\text{phy}}}{\gamma_{\text{num}} + \gamma_{\text{phy}}},
\qquad \gamma_{\text{num}} = \gamma_{\text{fact}} \cdot \eta, \qquad \gamma_{\text{phy}} = K \Delta t
\end{align}$

## Free-surface stabilization

An explicit free surface is unstable unless the time step resolves the surface-wave (drunken
sailor) time scale. The stabilization (Kaus, Mühlhaus & May, 2010)
removes that restriction by anticipating the buoyancy at the end of the step: the surface moves
by $\boldsymbol{v}\Delta t$, so

$\begin{align}
\rho g_y(\boldsymbol{x} + \boldsymbol{v}\Delta t) \approx \rho g_y(\boldsymbol{x}) + \Delta t\, \boldsymbol{v}\cdot\nabla(\rho g_y)
\end{align}$

Retaining the vertical part of the correction and treating it implicitly in $V_y$ gives the FSSA
term of $R_y$ above, with $\theta = 1$ in the implementation. Because the term sits on the diagonal of the
$V_y$ row, it competes with the viscous and penalty entries there; where a density jump makes it
dominant the surface row relaxes far more slowly than the bulk. `γ_eff` is therefore floored at
the break-even value

$\begin{align}
\gamma_{\text{eff}} \geq \frac{1}{2} \left| \Delta(\rho g_y) \right| \Delta t\, \Delta y
\end{align}$

which is applied whenever `free_surface = true`.

## Usage

Both the `DYREL` object and the solver call take $\phi$ as an extra argument, and the solver
additionally needs to know which phase is the air:

```julia
ϕ = RockRatio(backend, ni)
update_rock_ratio!(ϕ, phase_ratios, air_phase)   # or compute_rock_fraction!(ϕ, chain, xvi, di)
                                                 # where the surface is carried by a marker chain
dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, di, dt; ϵ = 1.0e-6)

solve_DYREL!(
    stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
    kwargs = (; air_phase = air_phase, free_surface = true, iterMax = 50.0e3)
);
```

Convergence is measured over the cells and faces $\phi$ marks as rock, since the ones it masks
out carry no meaningful velocity or pressure.

# Examples

Examples of a set of miniapps using this solver can be found in [this folder](https://github.com/PTsolvers/JustRelax.jl/tree/main/miniapps/DYREL2D).
