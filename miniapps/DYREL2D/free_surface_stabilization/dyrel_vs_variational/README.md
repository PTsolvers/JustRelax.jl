# DYREL vs. variational DYREL — iteration comparison

Measures how many dynamic-relaxation iterations each solver needs per time step on the
Rayleigh-Taylor problem of the two neighboring miniapps:

- **standard** — `solve_DYREL!` without a `RockRatio`, as in `RayleighTaylor2D_DYREL.jl`.
  Sticky air is stabilized through the free-surface velocity boundary condition, particles are
  advected with plain RK2.
- **variational** — `solve_DYREL!` with a `RockRatio` mask, as in
  `RayleighTaylor2D_VariationalStokes_DYREL.jl`. The air/rock interface is carried by a marker
  chain, particles are advected with MQS.

The metric is the `iter` field of the `solve_DYREL!` return value: the total number of
dynamic-relaxation iterations a time step consumed across all of its Powell-Hestenes sweeps.

## Running

```sh
julia --project=miniapps/DYREL2D/free_surface_stabilization/dyrel_vs_variational \
      miniapps/DYREL2D/free_surface_stabilization/dyrel_vs_variational/run_comparison.jl [nsteps]
```

`nsteps` defaults to `NSTEPS` in `config.jl`. The environment devs the JustRelax checkout this
folder lives in, so the comparison always exercises the working tree rather than a registered
release.

Results land in `results/`: `iterations.csv` (per-step raw data), `iterations.png` (per-step,
cumulative, and ratio) and `summary.txt`.

## What is shared and what is not

Both runs are handed the same problem, so that iteration counts are comparable:

| | value |
|---|---|
| grid, domain, rheology, gravity | identical |
| particle seeding | identical (`SEED` fixes `rand()`) |
| particles per cell | identical |
| initial pressure | hydrostatic in both |
| time step | fixed and identical (see below) |
| `ϵ`, `rel_drop`, `nout`, `iterMax`, `total_iterMax` | identical |
| `λ_relaxation_PH/DR`, `viscosity_relaxation`, `linear_viscosity`, `viscosity_cutoff` | identical |
| backend | identical |

Three differences remain, because they are what distinguishes the two solvers rather than
settings that could be harmonized:

1. **`γfact`** — the Powell-Hestenes penalty scaling, `20.0` for the standard solver and
   `100.0` for the variational one, matching each miniapp. This is a tuning parameter and it
   moves iteration counts substantially. Set `GAMMA_FACT_STANDARD == GAMMA_FACT_VARIATIONAL`
   in `config.jl` for a like-for-like solver comparison instead of an as-tuned one.
2. **Free-surface treatment** — boundary condition vs. marker chain plus `RockRatio`. Not
   separable from the choice of solver.
3. **Advection** — RK2 vs. MQS, which is tied to the free-surface treatment.

### Time step

`compute_dt` is velocity-dependent. The two solvers produce different velocity fields, so an
adaptive step would give each solver a *different* sequence of problems and make "iterations
per step" incomparable. The comparison therefore runs at a fixed 10 kyr step by default. Set
`ADAPTIVE_DT = true` in `config.jl` to recover each miniapp's own behavior, at the cost of
comparability.

Even at fixed `dt` the two trajectories diverge: the solvers treat the free surface
differently, so by late steps they are relaxing genuinely different interface geometries. Read
the early steps as a solver comparison and the late steps as a comparison of two models.

## Results

64×64, 50 steps, fixed 10 kyr step, CPU. Every step of every run converged; none hit
`total_iterMax`. Total dynamic-relaxation iterations over the 50 steps:

| `γfact` | standard | variational | variational / standard | output |
|---|---|---|---|---|
| as tuned (20 / 100) | 323,200 | 226,200 | 0.70 | `results/` |
| 20 / 20 | 323,200 | 430,700 | 1.33 | `results_gamma20/` |
| 100 / 100 | 227,300 | 226,200 | 1.00 | `results_gamma100/` |

At matched `γfact` the two solvers cost the same (γ=100) or the variational one costs more
(γ=20). The apparent advantage in the as-tuned row is the penalty scaling, not the
formulation: `γfact` is worth ~30% to *either* solver on this problem, and the two miniapps
happen to be tuned differently.

The variational solver has the shorter tail in every configuration — 7,300 vs 8,600 worst step
at γ=100, 13,900 at the standard solver's own tuning — so its per-step cost is the more
predictable of the two even where its mean is not lower.

### Resolution scaling at `γfact = 20`

CUDA, dt = 10 kyr, 50 steps, `total_iterMax = 3e5`. Every step converged.

| | 64² std | 64² var | 128² std | 128² var | 256² std | 256² var |
|---|---|---|---|---|---|---|
| total iterations | 322,100 | 492,000 | 471,400 | 276,100 | 580,800 | 1,425,500 |
| var / std | | 1.53 | | 0.59 | | 2.45 |
| worst step | 11,500 | 11,700 | 35,800 | 14,300 | 32,000 | 45,800 |

The standard solver scales smoothly (1.46× then 1.23× per refinement). The variational solver
does not: its cost falls from 64² to 128², then rises 5.2× from 128² to 256². The ranking
between the two therefore inverts twice along this row, purely from where `γfact = 20` sits
relative to each solver's own resolution-dependent optimum.

`γfact` is not resolution-portable, and it is the dominant control on both solvers. Any ratio
quoted from this harness is meaningless without stating both `γfact` and the resolution.

### `γfact` stability limit of the variational solver

The usable `γfact` shrinks by roughly 4× per 2× refinement:

| | 64² | 128² |
|---|---|---|
| γ = 20 | converges | converges |
| γ = 50 | — | converges |
| γ = 100 | converges | marginal, some steps fail |
| γ = 200 | converges | fails on 40 of 50 steps |

Past the limit the failure is in the outer Powell-Hestenes loop, not the velocity solve: the
velocity residuals reach 1e-4..1e-6 while the pressure residual `Rp` oscillates at O(0.1-1)
with no downward trend, and the solve exhausts the `itPH in 1:1000` cap. Raising
`total_iterMax` does not help. Two properties drive it — `P += γ_eff · RP` is an *inexact*
augmented-Lagrangian update whose inner tolerance `ϵ_vel = err · rel_drop` is floored at 1e-3,
and `γ_num = γfact · η_local` spans the full air-to-rock viscosity range before being weighted
by `ϕ`, so interface cells carry a near-zero penalty beside neighbors carrying the full one.

### Sensitivity to `γfact`

Both solvers bottom out near `γfact = 100` and cost more in either direction (dt = 10 kyr,
total iterations over 50 steps):

| `γfact` | standard | variational |
|---|---|---|
| 20 | 323,200 | 430,700 |
| 100 | 227,300 | 226,200 |
| 200 | 270,500 | 237,500 |

### Sensitivity to the time step

At `γfact = 200`, going from a 10 kyr to a 25 kyr constant step costs neither method anything
in convergence. Every step of every run converged; none approached `total_iterMax`. Mean
iterations per step, dt=25 relative to dt=10:

| window | standard | variational |
|---|---|---|
| step 1 (identical initial state) | 1.01 | 1.13 |
| steps 1-10 | 0.93 | 1.03 |
| all 50 steps | 1.00 | 0.88 |
| worst single step | 1.01 | 1.17 |

Only step 1 isolates the time step: from step 2 on, the dt=25 run has advanced 2.5× further,
so its interface geometry differs and the counts mix the step size with a differently deformed
problem.

The rheology is purely `LinearViscous`, so `dt` never enters the constitutive update — it
reaches the solver only through the free-surface stabilization term and the DYREL coefficient
setup. Insensitivity to `dt` here does not imply the same for a visco-elastic run, where `dt`
sets the Maxwell time.

Reproduce the γ-matched rows with:

```sh
DYREL_GAMMA=20.0  DYREL_OUTDIR=results_gamma20  julia --project=... run_comparison.jl
DYREL_GAMMA=100.0 DYREL_OUTDIR=results_gamma100 julia --project=... run_comparison.jl
```

## Layout

| file | contents |
|---|---|
| `config.jl` | every knob, grouped into shared and per-solver |
| `model.jl` | grid, rheology, phase initialization, Stokes arrays and boundary conditions |
| `runners.jl` | the two time loops, stripped to what affects the solver |
| `run_comparison.jl` | entry point: backend init, both runs, CSV/figure/summary |

`@init_parallel_stencil` may be called only once per session, so both runs share one process
and therefore one backend; `IS_CUDA` in `config.jl` switches both together.
