# Adaptive adjoint penalty experiment

This extends the [convergence investigation](convergence_findings.md) while retaining
the production DYREL residuals, nonlinear tangent, damping adaptation, and stopping
criterion. It changes only a private instrumented copy of the adjoint loop. No
production defaults or example files were changed.

The experiment is restricted to one CPU/MPI rank, full-rock, incompressible cases.
It adjusts the factor `g` in `γ_eff[i] = g * η_converged[i] / 2`. The initial factor
20 is the local scaling that already worked well in the subduction tests. This
controller does not switch between global and local spatial scaling.

## Rule and safeguards

At a nonconverged outer pressure iteration, after evaluating the true residual:

- Double `g` when pressure is the dominant normalized residual, remains above the
  tolerance, and its **unnormalized** RMS residual contracts by less than 20% in
  each of two consecutive completed passes. Using the unnormalized residual for
  contraction avoids changes in the adjoint-velocity normalization mimicking progress.
- Halve `g` when momentum is the dominant normalized residual, each of the previous
  two velocity solves took more than 200 iterations, and pressure contracted by
  more than 30% in each pass. In particular, do not reduce it while pressure dominates.
- Bound `g` to `[0.1, 320]`, limit changes to factors of two, and allow two completed
  outer passes between changes.
- If the total outer residual more than doubles on the first pass after a change,
  restore the previous factor and impose an additional cooldown. This restores
  parameters, not an earlier solution iterate; it is not a general divergence recovery.

Each change rebuilds the velocity diagonal and Gershgorin bounds, retains the frozen
constitutive tangent amplification, recalculates time steps, clears pseudo-velocity
and stored residual/step history, resets damping to 1, and invalidates the stored
Rayleigh secant. The adjoint pressure multiplier is not rescaled. The forward penalty,
physical viscosity, and physical bulk viscosity are checked unchanged after every run.

The thresholds are experimental heuristics, not universal parameter estimates. The
200-iteration threshold in particular needs assessment at other grid sizes. An early
version could raise and lower the penalty in succession while pressure still dominated;
the momentum-dominance guard removed that cycle in these tests.

## Frozen subduction states, 32 × 64

Same first-step hard-cutoff and smooth-bounds states as the previous investigation.
Fresh adjoints use `nout=10`, `rel_drop=0.5`, and tolerance `1e-3`. The fixed and
adaptive runs have the same starting factor. No tolerance is loosened by the controller.

| State | Initial `g` | Fixed iterations | Adaptive iterations | Final adaptive `g` |
|---|---:|---:|---:|---:|
| Hard cutoff | 0.2 | 45,470 | 7,830 | 6.4 |
| Hard cutoff | 20 | 710 | 710 | 20 |
| Hard cutoff | 200 | 1,960 | 1,960 | 200 |
| Smooth bounds | 0.2 | Failed at 60,000 | 8,470 | 6.4 |
| Smooth bounds | 20 | 1,000 | 1,000 | 20 |
| Smooth bounds | 200 | 2,690 | 2,600 | 100 |

The fixed smooth run at `g=0.2` ended with outer residual `7.31e-3`, above `1e-3`.
Both poorly initialized adaptive runs increased `g` five times. The smooth `g=200`
run decreased it once. None of these runs activated the residual-growth rollback.

Against fixed-local `g=20` reference solves at `1e-6`, the largest relative
viscosity-field gradient difference among the successful adaptive runs was `2.02e-4`;
the largest relative adjoint velocity difference was `6.74e-4`. These compare linear
solver outputs, not finite differences of the nonlinear model. Material-parameter
gradients were not rerun in this experiment.

## Pressure observation

A fully clipped 8 × 8 inclusion uses the mean pressure inside the weak body minus
the mean outside, preserving the pressure gauge. The tolerance is `1e-7`, with
`nout=10`, `rel_drop=0.5`, and a 500-outer-pass limit. This differs from the earlier
pressure benchmark's iteration settings, so its counts should not be mixed with that table.

| Initial `g` | Fixed | Adaptive |
|---|---:|---:|
| 0.2 | Failed at 500 outer passes | 1,860 iterations |
| 20 | 420 iterations | 420 iterations |
| 200 | 360 iterations | 330 iterations |

The adaptive weak-penalty case raises `g` to 3.2. The high-penalty case reaches the
upper bound 320. A tighter `1e-10` reference checks solution agreement.

## Interpretation

The controller adds robustness to a poorly chosen local penalty and leaves the
already effective factor 20 unchanged. It does **not** find the fastest factor:
the recovered runs still take substantially longer than starting at 20. The reduction
branch provided only a small gain in the tested subduction case.

Keep this experimental. Starting at 20 with a safeguarded adaptation is a reasonable
candidate for full-resolution/time-dependent validation, but there is no evidence yet
for enabling it by default. No evolving 64 × 128 model, compressible/enclosed weak-body
case, GPU, or multi-rank MPI run was tested. Single CPU timings are logged but are not
controlled performance measurements.

## Run

```sh
julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/adaptive_adjoint_penalty.jl hard
julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/adaptive_adjoint_penalty.jl smooth
julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/adaptive_adjoint_penalty.jl pressure
```

The first two commands use `/private/tmp/subduction32hard.jls` and
`/private/tmp/subduction32smooth.jls`. Override with `JUSTRELAX_PENALTY_STATE`, or
regenerate the same frozen states using `JUSTRELAX_MODEL_SAVE_STATE` in
`subduction_shearzone_diagnostic.jl` as described in the preceding investigation.
Use caches only with matching model/package revisions.

`JUSTRELAX_PENALTY_STARTS=0.2,20,200` selects starting factors. The default runs both
fixed and adaptive trials; `JUSTRELAX_PENALTY_TRIALS=adaptive` skips repeated fixed
trials. Every invocation also runs ten controller checks and a tighter fixed-local
reference. Integration checks verify convergence and invariance of physical fields
and the forward penalty. The rollback branch is covered by a controller check, not
by a deliberately divergent simulation.
