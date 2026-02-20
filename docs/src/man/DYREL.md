# Using the APT method with auto tuned damping coefficients

> [!WARNING]
> This solver is still work-in-progress/experimental. In the current state, only 2D Stokes is supported. Variational Stokes and 3D version coming up soon.

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
    di,
    dt,
    igg;
    kwargs = (;
        iterMax              = 50.0e3,
        nout                 = 10,
        rel_drop             = 0.1,
        λ_relaxation_PH      = 1,
        λ_relaxation_DR      = 1,
        verbose_PH           = false,
        verbose_DR           = false,
        viscosity_relaxation = 1,
        linear_viscosity     = true,
        viscosity_cutoff     = (-Inf, Inf),
    )
);
```
where the solver keyword arguments are:
- `iterMax` $\rightarrow$ maximum number of total iterations.
- `nout` $\rightarrow$ damping coefficients are re-computed every `nout` iterations.
- `rel_drop` $\rightarrow$ the tolerance for the inner dynamic relaxation loop is $error(P^n) \text{rel_drop}$ where $n$ is the inner Powell-Hesteness iteration counter.
- `λ_relaxation_PH` $\rightarrow$ relaxation coefficient for the plastic multiplier ($\cdot\lambda$) during the inner Powell-Hesteness loop. `λ_relaxation_PH=1` means no relaxation.
- `λ_relaxation_DR` $\rightarrow$ relaxation coefficient for the plastic multiplier ($\cdot\lambda$) during the innes Dynamic Relaxation loop. `λ_relaxation_DR=1` means no relaxation.
- `verbose_PH` $\rightarrow$ # print solver metrics during  inner Powell-Hesteness loop.
- `verbose_DR` $\rightarrow$ # print solver metrics during  innes Dynamic Relaxation loop.
- `viscosity_relaxation` $\rightarrow$ relaxation coefficient for the viscosity. `viscosity_relaxation=1` means no relaxation.
- `linear_viscosity` $\rightarrow$ if the rheology is linear (viscosity will not be updated during the solver iterations).
- `viscosity_cutoff` $\rightarrow$ viscosity is clamped so that $\text{viscosity_cutoff}_1 \leq \eta \leq \text{viscosity_cutoff}_2$.

# Examples

Examples of a set of miniapps using this solver can be found in [this folder](https://github.com/PTsolvers/JustRelax.jl/tree/main/miniapps/DYREL2D).
