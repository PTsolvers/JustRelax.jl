# Temporary adjoint solver experiments

This folder contains one-off investigation scripts, benchmark drivers and findings.
It is separate from the production change and is not loaded by the package or discovered
by the regular test runner. Keep it out of a production-only commit. Files remain untracked
and visible in Git; nothing has been deleted or hidden with ignore rules.

## Actual production change

| Files | Purpose |
|---|---|
| [adjoint_solver_utils.jl](../../src/DYREL/adjoint/adjoint_solver_utils.jl) | Local-penalty options, calculation and preconditioner rebuild |
| [solver_adjoint.jl](../../src/DYREL/adjoint/solver_adjoint.jl), [solver_adjoint_VS.jl](../../src/DYREL/adjoint/solver_adjoint_VS.jl) | Wire the option into both adjoint solvers |
| [solver.jl](../../src/DYREL/solver.jl) | Public keyword documentation |
| [test_adjoint_dot_product.jl](../../test/test_adjoint_dot_product.jl), [test_adjoint_gradients.jl](../../test/test_adjoint_gradients.jl) | Permanent regression tests; retain with the implementation |
| [DYREL.md](../../docs/src/man/DYREL.md) | User-facing documentation |

The [next-steps plan](../../docs/adjoint_solver_next_steps.md) is a supporting roadmap and
validation record, separate from the implementation. No experiment in this folder is
required to use `adjoint_kwargs = (; penalty_scaling = :local, penalty_factor = 20.0)`.

## Experiment contents

- [convergence_findings.md](convergence_findings.md): diagnosis and comparisons.
- [adaptive_penalty_findings.md](adaptive_penalty_findings.md): experimental adaptive controller results.
- `production_local_penalty.jl`: one-off verification using saved subduction states.
- `dora_maira_local_smoke.jl`: one 32 × 64 step of the adapted companion example,
  using its local-penalty default and writing figures/VTK to a temporary directory.
- Remaining `.jl` files: operator probes and alternative-solver/tuning experiments.

Run scripts from the repository root with `--project=.`. Commands below use the new location.
Saved model states and logs under `/private/tmp` remain external temporary data; they have
not been added to the repository. Some scripts require those states or the companion
SubductionShearZone example; their existing environment overrides are documented below.

---

## Small-grid adjoint operator diagnostic

The latest [convergence investigation](convergence_findings.md) identifies slow
divergence-free modes caused by the global pressure penalty and compares local
penalty scaling and patch corrections, including a reduced reproduction of the
user's `SubductionShearZone` model.

The follow-up [adaptive-penalty experiment](adaptive_penalty_findings.md) tests
bounded adjustments between pressure iterations. It recovers poor starting factors
while leaving the effective fixed-local factor 20 unchanged in these cases.

## Production local-penalty check

The fixed-local option is available through
`adjoint_kwargs = (; penalty_scaling = :local, penalty_factor = 20.0)`.
See [the next-steps plan](../../docs/adjoint_solver_next_steps.md).
`production_local_penalty.jl state.jls` runs the production variational adjoint at tolerances
1e-3 and 1e-6 on a frozen, full-rock, incompressible state saved by
`subduction_shearzone_diagnostic.jl`. It checks forward-state preservation and material
gradient agreement. Older private solver experiments below remain historical diagnostics.

## Historical diagnostics

Run from the repository root with the JustRelax environment and a single CPU process:

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no scratch/adjoint_solver_experiments/operator_bounds.jl
```

The script assembles the standard DYREL adjoint's preconditioned inner velocity operator on a
4 × 4 grid, one velocity basis vector at a time. It calls the production adjoint residual and
velocity update kernel, including the pressure-penalty correction, with zero pressure multiplier,
no objective seed and zero pseudo-velocity update. The matrix represents the negative residual
change divided by the forward diagonal `D`. It reports its diagonal, absolute row sums relative
to the forward `λmaxV`, and asymmetry. A separate deterministic vector checks that the assembled
matrix reproduces a direct operator application. `run_case` returns the matrix, row sums, forward
bounds and the mapping from rows to component and grid index.

It also compares the ratio of the largest to smallest nonzero singular value before and after
scaling each row by the exact adjoint diagonal. Singular values below `1e-12` times the largest
are excluded; the numerical rank is reported. This is a matrix conditioning diagnostic, not a
prediction of iteration count or a guarantee of convergence for a nonsymmetric operator.

The four cases use free-slip boundaries, constant density, no free-surface stabilization and a
zero-velocity, zero-pressure viscoelastic state. The two phases have viscosities 1 and either 1
or 100; the shear modulus is either 1 or 10,000. This is a diagnostic of these particular
linearized states, not a convergence test or proof of stability. Two additional synthetic
pure-shear states use Drucker–Prager plasticity with friction 20° and dilation 0° or 20°; all
16 cells yield. These states have not been obtained from a converged forward solve. The cases
do not cover nonlinear viscosity, pressure-dependent density, variational masks, periodic
boundaries, or a user's converged simulation state. The preconditioned operator can be
nonsymmetric; an absolute row-sum bound does not alone guarantee dynamic-relaxation stability.

On CPU (Julia 1.13.2), the diagnostic produced:

| Viscosity contrast | Shear modulus | Maximum row sum / forward bound | Diagonal range |
|---:|---:|---:|---:|
| 1 | 1 | 0.8333 | 0.7407–0.7407 |
| 100 | 1 | 0.8487 | 0.6824–0.7623 |
| 1 | 10,000 | 0.6250 | 0.4879–0.4879 |
| 100 | 10,000 | 0.6517 | 0.01078–0.5167 |

All 24 rows in each case were below the forward bound. Direct matrix-vector residuals were below
`6e-16`. The wide diagonal range in the last case shows that the forward diagonal can be a poor
match to the preconditioned adjoint tangent even where its row-sum bound is conservative.

The comparison with exact adjoint-diagonal scaling gives the following singular-value ratios
(smaller is better for this measure):

| State | Numerical rank | Forward diagonal | Exact adjoint diagonal |
|---|---:|---:|---:|
| Viscoelastic, contrast 100, G = 10,000 | 15 / 24 | 269.8 | 11.66 |
| Non-associated plastic, dilation 0° | 24 / 24 | 1,495 | 1,845 |
| Associated plastic, dilation 20° | 24 / 24 | 302.0 | 272.7 |

The exact diagonal greatly improves the first matrix's singular-value ratio but worsens the
non-associated example's. The plastic cases also remain below the forward row-sum bound (maximum
ratios 0.843 and 0.700 respectively). These results do not justify replacing the solver's
preconditioner without testing actual adjoint iterations on converged forward states.

## Converged forward state: DYREL and matrix-free GMRES

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/compare_gmres.jl
```

This experiment uses the non-associated Drucker–Prager sinking-block setup on an 8 × 8 grid.
It runs the production forward and adjoint solves, restores the forward pressure and stress
history to their values at the adjoint linearization point, and applies the production adjoint
residual as a matrix-free coupled velocity/pressure operator. Restarted GMRES uses the forward
velocity diagonal and a fixed pressure scaling; its true scaled residual is checked at each
restart. The script also checks GMRES against a 2 × 2 system and compares the phase-1 cohesion
gradient against central differences of independently converged forward solves.

One CPU run (Julia 1.13.2, one thread) gave:

| Measurement | DYREL adjoint | GMRES |
|---|---:|---:|
| Iterations | 1,400 inner iterations | 805 Arnoldi iterations |
| Same scaled coupled residual | 9.60e-10 | 9.91e-10 |
| Phase-1 cohesion gradient | 3.508571337 | 3.508571314 |

The iteration counts are not timings: a GMRES step includes orthogonalization and restart
residual evaluations, and a DYREL solve has outer pressure iterations. GMRES residuals at
restarts were approximately `1, 7.49e-2, 9.94e-3, 8.25e-4, 1.04e-4, 1.80e-5, 2.80e-6,
3.16e-7, 5.09e-8, 9.73e-9, 1.10e-9, 9.91e-10`.

At cohesion perturbation `h = 1e-6`, the central difference was 3.508562057 (relative error
`2.65e-6` for both gradients). At `h = 1e-5` and `1e-4`, the positive perturbation produced
an objective around -2.577 rather than the base value -2.767; the resulting central differences
were 9,511 and 953. The yielded-cell masks were identical, so a mask count alone does not
explain the different solutions. This behavior is consistent with the nonlinear forward solve
reaching another solution or basin, but its cause has not been established. A finite difference
in that range therefore does not validate the local gradient. A second unperturbed forward solve
reproduced the base objective exactly in this run.

This is one small CPU case. Neither GMRES nor the exact-diagonal experiments above have been
introduced into the production solver. Comparisons across representative converged states,
including wall time, memory, other rheologies, the variational solver and GPUs, are needed before
choosing a solver change.

A second run with argument `12` used a 12 × 12 grid. DYREL reached a scaled residual of
`2.34e-10` in 2,200 inner iterations. GMRES needed 2,243 Arnoldi iterations to reach `9.93e-10`;
it had only reached `5.52e-6` at 1,200 iterations. The respective phase-1 cohesion gradients
were 2.760466855 and 2.760466862, versus a central difference of 2.760473286 at `h = 1e-6`
(relative errors `2.33e-6`). As on the smaller grid, larger finite-difference perturbations
produced different forward objectives despite an unchanged yielded-cell mask. The repeated
unperturbed solve again reproduced its objective exactly.

## Cohesion sweep and velocity continuation

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/cohesion_continuation.jl
```

This sweeps cohesion from 0.0199 to 0.0201 at 12 × 12. For each value it compares a fresh
initial velocity with the velocity from the previous converged point. Initial pressure and
stress history stay fixed. This is **velocity continuation only**: `solve_DYREL!` copies the
initial pressure into `P0`, the physical previous-time pressure, so changing the initial
pressure would change this problem. The diagnostic independently recomputes the production
momentum and continuity kernels and reapplies the unrelaxed constitutive update at each saved
state. It reports RMS residuals, maximum changes to stress and plastic multipliers, pressure
range, center and vertex yield counts, and the difference between prescribed normal velocities.
Use `JUSTRELAX_SWEEP_QUICK=1` to run the first four values.

A CPU run gave these pairs for selected points:

| Cohesion | Fresh objective | Continued objective | Maximum pressure difference | Center / vertex yield-mask differences |
|---:|---:|---:|---:|---:|
| 0.01995 | -2.167674 | -2.165100 | 0.181 | 0 / 1 |
| 0.01999 | -2.202470 | -1.803831 | 0.251 | 1 / 0 |
| 0.02000 | -2.215117 | -1.960378 | 0.0999 | 0 / 0 |

The prescribed normal velocities matched exactly in these comparisons. At cohesion 0.02000,
the maximum center plastic-multiplier difference was 8.74 despite identical yield masks.
Across the complete sweep, independently recomputed momentum RMS residuals were below `4.7e-10`
and continuity RMS residuals below `2.9e-12`; reapplying the production constitutive update
changed the stored stress and multipliers by zero at printed precision. An alternative short
continuation path, directly from 0.01999 to 0.02000, was rejected by the diagnostic's forward
convergence check. These observations show strong dependence on the initial velocity; they do
not by themselves prove mathematical nonuniqueness or explain the different states. In
particular, the constitutive check reuses the production update rather than an independent yield
and complementarity implementation, and a full pressure-and-velocity continuation has not been
run.

## Adjoint iteration-control benchmark

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/adjoint_tuning.jl 16
```

This solves the forward problem once and compares fresh adjoint solves on that same frozen
state. All use an adjoint tolerance of `1e-9`, the same observation and material gradient, and
no warm start. Julia compilation is warmed before timing. The table shows adjoint iterations
on one CPU thread, for the same non-associated plasticity setup at three grid sizes:

| `nout` | `rel_drop` | 8 × 8 | 12 × 12 | 16 × 16 |
|---:|---:|---:|---:|---:|
| 50 | 0.5 | 1,400 | 2,200 | 2,850 |
| 50 | 0.1 | 1,650 | 2,150 | 2,550 |
| 50 | 0.01 | 1,400 | 1,950 | 2,600 |
| 25 | 0.5 | 1,575 | 3,725 | 7,625 |
| 100 | 0.5 | 1,600 | 2,200 | 3,400 |
| 25 | 0.1 | 2,150 | 2,000 | 3,425 |
| 100 | 0.1 | 1,600 | 2,300 | 2,800 |

All gradients agreed within `1e-6` relative to the scale used in the script. At 16 × 16,
single warmed timings ranged from 1.61 seconds (`nout = 50`, `rel_drop = 0.01`) to 5.03 seconds
(`nout = 25`, `rel_drop = 0.5`); these single samples are illustrative rather than a timing
study. More frequent checks also change the adaptive damping, so smaller `nout` is not always
faster. `rel_drop = 0.01` is already the forward solver default, and these measurements do not
justify changing defaults across different rheologies or objectives.

## Adjoint-specific pseudo-time bounds

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/adjoint_exact_bounds.jl 16
```

This experiment probes every velocity degree of freedom at a converged state, assembles the
actual preconditioned adjoint velocity operator including the pressure-penalty correction, and
computes its absolute row sums. It then substitutes these bounds for the forward-derived bounds
when calculating adjoint pseudo-time steps. It keeps the production damping update and diagonal
preconditioner, and compares convergence and the cohesion gradient with the original bounds.
The dense probing and storage are for small reference grids only, not a scalable algorithm.

| Grid | Rows above forward bound | Largest actual / forward bound | Forward bounds: iterations | Adjoint bounds: iterations |
|---:|---:|---:|---:|---:|
| 8 × 8 | 2 / 112 | 1.021 | 1,400 | 1,350 |
| 12 × 12 | 11 / 264 | 1.039 | 1,950 | 2,350 |
| 16 × 16 | 25 / 480 | 1.044 | 2,600 | 3,350 |

The gradients agreed to the precision reported by the script. The 16 × 16 dense setup took
0.335 seconds after its probing path was warmed, in addition to the solve (single CPU timing).
The smaller-grid setup timings included compilation and are not useful for comparison. An
actual adjoint-derived row bound can exceed the forward estimate, but directly substituting it
for the pseudo-time calculation worsened iteration counts on two grids. The row-sum bound is
not an exact stability limit for a nonsymmetric dynamic-relaxation iteration; designing the
step and damping together and validating the result across adjoint cases remain necessary.

## Variational adjoint with dimensional viscosity cutoffs

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/cutoff_contrast.jl 8
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_cutoff_damping.jl 8
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_cutoff_bounds.jl 8
```

The full-rock variational case uses no air phase, no nondimensionalization, a dimensional
viscosity cutoff of `1e18–1e23`, and a small weak inclusion in background shear. On the 8 × 8
CPU case, all 64 center viscosities were at a cutoff (12 lower and 52 upper). The forward took
750 iterations and the adjoint 8,550 to reach its `1e-7` tolerance. Both a power-law exponent
of 1 and 1/3 produced this count when all cells were clipped. The computed nonlinear
viscosity feedback was zero at clipped cells in this particular case.

On the same frozen state, changing only the adjoint damping factor `c_fact` gave:

| `c_fact` | Adjoint iterations |
|---:|---:|
| 0.15 | 27,350 |
| 0.25 | 13,200 |
| 0.50 | 8,550 |
| 0.75 | 7,550 |
| 1.00 | 12,250 |
| 1.50 | 21,400 |

These are experimental adjoint-only factors, not a proposed new default. The returned adjoint
velocities agreed to within `1.5e-8` relative to the `c_fact = 0.5` result. The dense variational
adjoint-operator probe found actual-to-forward row-bound ratios from 0.625 to 1.0; replacing the
inherited bounds with the actual row sums increased the adjoint from 8,550 to 8,900 iterations.
The computed adjoint velocity fields agreed to `1.5e-8` relative. This identifies a useful
reproduction of slow convergence, but it does not yet establish a robust new damping or
preconditioning formula. These tests use one small CPU grid, prescribed geometry and boundary
conditions, and do not establish GPU or large-problem behavior.

Keeping the same dimensional geometry and moving only the upper cutoff gave:

| Cutoff | Effective viscosity ratio | Forward iterations | Adjoint iterations |
|---|---:|---:|---:|
| `1e18–1e20` | `1e2` | 950 | 1,100 |
| `1e18–1e21` | `1e3` | 850 | 1,750 |
| `1e18–1e23` | `1e5` | 750 | 8,550 |

In each case all 64 centers were clipped. The changing iteration counts identify contrast as a
relevant factor in this setup; they do not isolate the mechanism or provide a general scaling
law.

For the 8 × 8 cutoff case, the diagonal of the measured forward-preconditioned adjoint velocity
operator ranged from 0.9218 to 1.0. Its numerical singular-value ratio was approximately
`8.98e5` (full rank at the `1e-12` relative threshold). Replacing both the forward diagonal and
row bounds with their measured adjoint counterparts took 8,900 iterations, versus 8,550 for
the original parameters, and preserved the adjoint velocity to `1.5e-8` relative. This small
experiment suggests that a purely local diagonal correction does not resolve the difficult
high-contrast modes in this case.

## Variational cutoff: flexible GMRES with DYREL preconditioning

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_cutoff_fgmres.jl 8
JUSTRELAX_FGMRES_RESTART=120 JUSTRELAX_FGMRES_MAXITER=1200 JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_cutoff_fgmres.jl 12
```

This standalone CPU prototype applies the coupled variational adjoint operator matrix-free,
scales the dimensional velocity and pressure unknowns, and uses right-preconditioned flexible
GMRES. Each preconditioner application starts from zero and performs two pressure updates with
four fixed DYREL velocity sweeps each, without adapting the inner damping. It tests the operator
and preconditioner for linearity. It also computes the production solver's momentum/continuity
criterion independently, because that criterion and the prototype's scaled full-vector residual
are different. Krylov storage numbers below include the `V` and `Z` bases, not all solver memory.
Single warmed CPU timings are illustrative; the prototype timing also includes its first
specialized GMRES compilation.

| Grid / restart | DYREL | FGMRES–DYREL | Production-style criterion | Sensitivity |
|---|---|---|---|---|
| 8 × 8 / 40 | 8,550 iterations; 5.71 s | 55 outer, 440 preconditioner sweeps; 1.39 s; `V`/`Z` 0.109 MiB | `7.76e-8` / `2.27e-8` | viscosity field gradients agree to `1.1e-8` relative |
| 12 × 12 / 40 | 7,500 iterations; 4.76 s | 400 outer, 3,200 sweeps; 3.26 s; `V`/`Z` 0.252 MiB | `5.00e-8` / `0.72` | not evaluated: FGMRES failed to converge |
| 12 × 12 / 120 | 7,500 iterations; 4.59 s | 1,200 outer, 9,600 sweeps; 8.58 s; `V`/`Z` 0.750 MiB | `5.00e-8` / `1.29e-5` | not evaluated: FGMRES failed to converge |

All reference adjoints use tolerance `1e-7`, so the first FGMRES case also satisfies the
production criterion. At 12 × 12 a longer restart helped but did not reach tolerance. The
prototype currently uses a full momentum residual vector, whereas the production variational
convergence test trims boundary rows; a correct reduced-space treatment and pressure-null-space
analysis need investigation before attributing stagnation to the solver method itself. This is
not a production implementation or evidence that a general FGMRES–DYREL method fails. It is
CPU-only and has not been evaluated on a realistic larger grid or GPU.

## Local cutoff derivative check

```sh
julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/cutoff_derivative.jl
```

Checks the production stress-to-viscosity tangent and material-parameter pullback against
central differences for a nonlinear power law with cutoffs `1e18` and `1e23`. The fixture
contains cells strictly below, inside, and strictly above the bounds. Both derivatives
must vanish in strictly clipped cells. A separate scalar check records Enzyme's interior
branch convention at exact equality; a hard cutoff has no unique derivative there.
This local CPU check does not test an entire adjoint solve or accelerator backends.

## Adjoint-only DYREL penalty on nonlinear viscosity

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_nonlinear_penalty.jl 8
JUSTRELAX_REFRESH_ONLY=1 JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_nonlinear_penalty.jl 12
JUSTRELAX_REFRESH_ONLY=1 JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_nonlinear_penalty.jl 16
JULIA_JUSTRELAX_BACKEND=CPU julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_penalty_pilot.jl 8
```

These CPU experiments keep the converged physical state and pressure history fixed while
constructing a separate adjoint DYREL preconditioner and pressure penalty. The material bulk
viscosity is checked unchanged. The case has power-law exponent 1/3, with a weak inclusion
clipped at `1e18` and other cells mostly within `1e18–1e23`; the nonlinear viscosity feedback
is active in unclipped cells.

| Grid | Forward iterations | Adjoint inherited parameters | Adjoint parameters refreshed from final viscosity |
|---|---:|---:|---:|
| 8 × 8 | 2,100 | 6,800 | 5,600 |
| 12 × 12 | 2,750 | 2,950 | 3,050 |
| 16 × 16 | 4,600 | 10,350 | 10,600 |

The adjoint velocity fields agreed to `2.1e-7` relative or better, at adjoint tolerance `1e-7`.
The adjoint numerical penalty is inherited from the beginning of the forward solve, when the
viscosity is initially computed; it is not updated as viscosity evolves. Rebuilding it from
the final state changes the penalty (relative maximum difference 2.11, 0.394 and 0.823 for the
three grids), but this is not consistently faster, so it is not enabled in the solver.

A naive automatic selection was also tested on the 8 × 8 case: run each penalty candidate for
200 or 500 adjoint iterations and choose the lowest reported residual. That would choose a
refreshed `γfact = 1`, which actually needs 9,950 iterations to converge, versus 5,600 for
`γfact = 10`. At 1,000 iterations the ranking had changed again. The pilot costs several
adjoint sweeps per candidate and the transient residual does not reliably predict total solve
cost, so it is not a sound automatic parameter rule in this experiment. These results do not
show that the adjoint can be made faster than the forward solve with these local parameters.
