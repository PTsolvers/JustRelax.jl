# Using APT with automatically tuned damping coefficients

> [!WARNING]
> This solver is still work-in-progress/experimental. The variational free-surface path described
> below is currently 2D only. Both standard and variational Stokes problems are supported in 2D.

Instead of using the Accelerated Pseudo-Transient method with damping coefficients that remain
constant throughout the PT iterations (as in [Räss et al., 2022](https://gmd.copernicus.org/articles/15/5757/2022/)),
we can use a self-tuning APT method based on the approach described in
[Duretz et al., 2025](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-5641/).

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
        iterMax_PH           = 1.0e3,
        iterMax_DR           = 50.0e3,
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
        free_surface         = false,
    )
);
```
where `grid` is the `Geometry` object (a tuple of grid spacings `di` is also accepted), and the
solver keyword arguments are:
- `iterMax_PH` $\rightarrow$ maximum number of Powell-Hestenes sweeps. This keyword applies to
  the variational solver.
- `iterMax_DR` $\rightarrow$ maximum number of dynamic-relaxation iterations per
  Powell-Hestenes sweep. This keyword applies to the variational solver. `iterMax` remains a
  compatibility alias for `iterMax_DR`; an explicit `iterMax_DR` takes precedence.
- `total_iterMax` $\rightarrow$ maximum number of dynamic-relaxation iterations summed over all
  Powell-Hestenes sweeps of one time step. This is what actually caps the cost of a time step, so
  a step that stops short of `ϵ` has usually hit this rather than `iterMax_DR`.
- `nout` $\rightarrow$ damping coefficients are re-computed every `nout` iterations.
- `rel_drop` $\rightarrow$ relative residual reduction targeted by each inner dynamic-relaxation
  solve.
- `λ_relaxation_PH` $\rightarrow$ relaxation coefficient for the plastic multiplier during the
  Powell-Hestenes loop. `λ_relaxation_PH = 1` means no relaxation.
- `λ_relaxation_DR` $\rightarrow$ relaxation coefficient for the plastic multiplier during the
  dynamic-relaxation loop. `λ_relaxation_DR = 1` means no relaxation.
- `verbose_PH` $\rightarrow$ print solver metrics during the Powell-Hestenes loop.
- `verbose_DR` $\rightarrow$ print solver metrics during the dynamic-relaxation loop.
- `viscosity_relaxation` $\rightarrow$ relaxation coefficient for the viscosity. `viscosity_relaxation=1` means no relaxation.
- `linear_viscosity` $\rightarrow$ if the rheology is linear (viscosity will not be updated during the solver iterations).
- `free_surface` $\rightarrow$ adds the free-surface stabilization (FSSA) term $V_y \frac{\partial (\rho g)}{\partial y} \Delta t$ to the vertical momentum residual, and puts a lower bound on the penalty so that it stays commensurate with that term.
- `viscosity_cutoff` $\rightarrow$ viscosity is clamped so that $\text{viscosity_cutoff}_1 \leq \eta \leq \text{viscosity_cutoff}_2$.
- `free_surface` $\rightarrow$ include the 2D density-gradient free-surface stabilization in both the momentum residual and the self-tuned vertical pseudo-transient coefficients. The default is `false`.

When `free_surface=true`, DYREL adds the local diagonal
$-\Delta t\,\partial_y(\rho g_y)$ to `Dy` and to the corresponding Gershgorin
row bound whenever the pseudo-transient coefficients are refreshed. The same
term is used by the Powell--Hestenes and dynamic-relaxation residual kernels.

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
pressure carrying the Powell-Hestenes correction. A zero-weight contribution drops out of its
stencil, so the traction-free condition above is recovered without explicitly meshing the
surface.

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

## Compatible reduced space

Zero-weight samples create algebraic nullspaces if their unknowns remain in the linear system.
Following Larionov, Batty & Bridson (2017), the implementation eliminates an unknown whenever
its equation constrains a zero-weight degree of freedom, and extends the same rule to the
viscous terms. In 2D this means:

- A `Vx` row is retained only when its face, its two neighboring pressure/stress centers, and
  its two shear-stress vertices have positive rock fractions. `Vy` uses the transposed stencil.
- A pressure row is retained only when its center is positive and all four velocity rows read by
  its divergence constraint are retained.

The second rule must use the *full* velocity-row validity, not just the four face fractions. For
example, a void shear vertex can eliminate a velocity row even while that face's fraction is
positive. Keeping a pressure constraint that still reads that velocity produces an incompatible
reduced system and a pressure residual that cannot converge.

An advected marker chain can change this reduced space between time steps. At the beginning of
each variational DYREL solve, pressure, pressure history, plastic pressure correction, plastic
multiplier, and velocity values on eliminated degrees of freedom are projected to zero. The
dynamic-relaxation velocity and residual histories are also reset because they belong to the
previous operator. Residual norms, Gershgorin estimates, Rayleigh quotients, pressure updates,
and the volumetric compatibility correction all use the rebuilt masks.

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
    kwargs = (;
        air_phase = air_phase,
        free_surface = true,
        iterMax_PH = 1.0e3,
        iterMax_DR = 50.0e3,
        total_iterMax = 50.0e3,
    )
);
```

Convergence is measured over the compatible reduced pressure and velocity spaces described
above, rather than every location whose own fraction is merely positive.

The solver returns a named tuple containing the residual histories, final normalized residual,
iteration count, and convergence status:

```julia
out = solve_DYREL!(...)
out.converged || @warn "DYREL did not converge" out.err out.iter
```

# Examples

Examples of a set of miniapps using this solver can be found in [this folder](https://github.com/PTsolvers/JustRelax.jl/tree/main/miniapps/DYREL2D).
