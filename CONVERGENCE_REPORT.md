# Convergence of `solve_VariationalDYREL!` on the Zwaan rift model

Report date: 2026-10-09. Branch `adm/conv`, nothing committed. Covers `CONVERGENCE_PLAN.md`, the
follow-up experiments, and the parts of `SOLVER_PLAN.md` done so far.

## Summary

- **The default convergence test accepts unconverged pressure.** It uses `min(errPt, RP_rms·dt)`,
  which passes once `RP_rms·dt` is small even when the relative continuity error is above ϵ. All
  results below use `strict_convergence = true` (`errPt < ϵ`). Under the strict test, the baseline
  solver fails at 135×76 in some runs and at 180×102 in every run.
- **Best configuration so far: `penalty_viscosity = :capped`, `penalty_ratio = 10`.** It is the only
  option that converged in every run at every size (90×51, 135×76, 180×102 on the CPU; 450×256 on
  the GPU, 2 runs of 21 steps). Against the baseline at 450×256 it uses 11% fewer DR iterations and
  5% less Stokes time. At 180×102, where the baseline never converges, it converges in 13–16k DR
  iterations for 7 steps.
- **The gain at production size is small.** It does not meet the acceptance rule in
  `SOLVER_PLAN.md` (≥ 15% less GPU time, or ≥ 25% fewer DR iterations per step ÷ `nx`). Iterations
  still grow with resolution; only a coarse-grid correction (SOLVER_PLAN item M1) addresses that.
- **The late-time stall at 450×256 is caused by plasticity in the shear zones, not by an
  ill-conditioned penalty.** Field dumps at a stalled pass put 95–99% of the squared momentum
  residual in rows of yielding cells.
- **Unconfirmed: your own settings with an `iterMax_DR` cap.** `GFACT=5 NOUT=20 VREL=1e-2 PREL=0.5`
  plus `iterMax_DR = 500` converged at 450×256 in 844 s (one run), the fastest converged GPU run.
  The same settings without the cap fail at step 2. They have not been repeated or run on the CPU
  grids, and `γfact = 5` and `nout = 20` each failed at 90×51 in the default configuration.

## Setup

- Model: `Rift_files/Rift2D_Zwaan.jl`, harness `Rift_files/convergence/bench.jl`. Drucker–Prager
  with tension cap (`pT = -C/2`), friction softening, elasticity (ν = 0.3), free surface,
  viscosity cutoff 1e19–1e24 Pa·s.
- Solver defaults in the harness: ϵ = 1e-3, `CFL = 0.99`, `c_fact = 0.5`, `γfact = 20`,
  `nout = 100`, `rel_drop = 1e-2`, `viscosity_relaxation = 1e-3`, `pressure_relaxation = 1`,
  `iterMax_DR = 1e5`, `total_iterMax = 1e5`.
- Grids: S = 90×51, M = 135×76, L = 180×102 (CPU, 8 threads, 7 steps, `TEND = 0.06` Myr) and
  450×256 (RTX 3080, 21 steps, `TEND = 0.2` Myr).
- CPU runs are not bit-reproducible (threaded reductions), so runs near the stability limit fail at
  random. Results that matter were repeated.
- Cost is reported as total DR iterations over all steps, and total Stokes wall time.
  `Rift_files/convergence/summarize.py logs/*.log` regenerates every number in this report.

## Baseline (strict test, default settings)

| grid | runs converged | DR iterations | Stokes time |
|---|---|---|---|
| S 90×51 | 3/3 | 42.5–46.4k | 55–59 s |
| M 135×76 | most, not all (NaN at step 2 in about 1 run in 3) | 15–23k | 27–40 s |
| L 180×102 | 0/3 (step 1 never converges) | — | — |
| 450×256 GPU | 2/2 | 217.2k, 228.4k | 919 s, 967 s |

At 180×102 the inner loop stops at `iterMax_DR = 1e5` with the error still above 1. On the GPU the
solve succeeds but most of the cost is in the last steps, after shear zones localize (from about
step 11).

## Best configuration: capped penalty

The penalty viscosity in each center cell is

```
η_pen = min(η_mean, R · η_loc),    η_loc = min(η, η_vep) where η_vep > 0, else η
γ_num = γfact · η_pen
```

`η_mean` is the mean viscosity over rock cells, and R is `penalty_ratio`. The penalty is computed
once at the start of each solve, after which the Gershgorin bounds and `dτ` are refreshed.

- R → ∞ gives a uniform penalty (`:mean`). R = 1 is close to using `η_vep` directly.
- In yielding cells `γ_eff` is already bounded by `K·dt`. The cap therefore binds mainly in weak
  cells that do not yield, where a uniform penalty is about 1e3 × the local viscosity.

### Results across R

| R | S | M | L | 450×256 GPU |
|---|---|---|---|---|
| baseline (`:local`) | 42.5–46.4k | 15–23k, some fail | fails | 217–228k, 919–967 s |
| 3 | 43.0k | 12.7k | 28.1k | 198.6k, 859 s |
| **10** | **40.5k** | **10.2–11.2k (3 runs)** | **13.1–15.8k (3 runs)** | **193.7k / 201.3k, 904 / 885 s** |
| 30 | — | 7.8k | 10.2k | 204.4k, 875 s |
| 100 | 45.5k | 6.6k | 8.4k | stalls at step 21 |
| ∞ (`:mean`) | 43.7k | 6.4–6.5k (3 runs) | 8.0–8.1k (3 runs) | stalls (0/3 completed) |

Every CPU run with `:capped` converged (15/15). On the CPU, a larger R is cheaper. On the GPU,
R = 3, 10 and 30 cost the same within noise, and R ≥ 100 stalls once shear zones form. R = 10 is
recommended because it has the most repeats and sits well below the R where the GPU run fails.

### How to use it

```julia
solve_VariationalDYREL!(
    stokes, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt, igg;
    kwargs = (;
        strict_convergence = true,
        penalty_viscosity = :capped,
        penalty_ratio = 10,
        # other settings as in the baseline
    ),
)
```

## What the field dumps show

Dumps were taken at pass 60 of the stalled step on the GPU, for the baseline and for `:mean`
(`analyze_dump.jl`).

- 95–99% of the squared momentum residual sits in rows of yielding cells.
- In yielding cells, `γ_eff·ϕ/η_vep` is about 0.5–0.7 under both penalty rules, because `K·dt`
  bounds `γ_eff`. The penalty there is not too stiff.
- With `:mean`, `γ/η_vep` reaches about 1e3 in weak cells that do not yield. That is what the cap
  removes, and it explains why `:capped` with a moderate R stays stable where `:mean` does not.
- The remaining stall is the plastic nonlinearity: the outer error cycles at 1–2ϵ while the return
  mapping in the shear zones keeps changing the operator.

## Other findings

- **Inner tolerance floor.** `ϵ_vel = max(errV·rel_drop, floor·ϵ)` lets the outer loop cycle just
  above ϵ after localization. Lowering the floor (0.1, or adaptively while the error is in (ϵ, 2ϵ))
  removes the stall for `:mean`, `:geomean` and `:floored`, but costs 1199–1313 s on the GPU, and
  1246–1286 s for `:capped` at R = 10 and 100. Not worth it.
- **`iterMax_DR` cap.** `iterMax_DR ≈ 2·nx` converged in all 34 runs at all sizes, at about
  baseline cost on the GPU (cap 1000: 228.9k / 227.9k iterations, 963 / 957 s). Caps of 200 and 500
  are 8–25% slower on the GPU. It is a robustness fix, not a speed-up. Not yet combined with
  `:capped`.
- **Tension cap.** Without it (`CAP=0`) step 1 fails at 90×51. With the cap beyond the
  Drucker–Prager apex (`pT_factor = 5`) the run is invalid. `λ_relaxation_PH/DR` have no effect:
  the cap return mapping ignores `λ_relaxation`.
- **`viscosity_relaxation = 1e-2`.** Converges at S and M. 1 of 2 runs at L, 1 of 2 on the GPU (the
  successful GPU run was the fastest seen, 798 s). Too fragile.

## Rejected experiments

| experiment | result |
|---|---|
| E1: penalty from `η_vep`, rebuilt every pass | diverges at S (all three variants) |
| E2: adaptive pressure step | stalls at S |
| E3: Anderson acceleration on pressure | fails at S |
| E4: extrapolated initial guess | no gain at S |
| E5: `CFL > 1` | NaN at S (1.2 and 1.5) |
| E6: viscosity cutoff continuation | marginal; fails at L without an iteration cap |
| E7: coarse first solve | marginal |
| E9: yield at augmented pressure | fails at S |
| Rp-growth exit from the inner loop | fails at S |
| frozen viscosity within a pass | NaN at S and M |
| `nout = 20` | fails at S |
| `rel_drop` 1e-1 / 1e-3 | fails at M / at S |
| `γfact` 5 / 40 / 80 | fails at S / fails on the GPU at step 11 / slower |
| `pressure_relaxation` 0.5–0.75 | converges, 1.3–4× slower |
| momentum restart when `Σ R·dVdτ < 0` (every 10 or 50) | fails at L |
| per-iteration residual check, stagnation exit (K3) | fails at S |
| `:floored` (local penalty with floor 0.1·η_mean) | fast on the CPU; GPU stalls at step 13 without the adaptive floor, 1236 s with it |
| `:geomean` | like `:mean`: fast on the CPU, 1313 s on the GPU |

## Krylov inner solve (GCR)

`inner_solver = :gcr` freezes the plastic state for each pass (secant stiffness
`k = τII/(2·εII_eff)`) and solves the linearized momentum equation with GCR(m), diagonal
preconditioner, restarts. The operator is about 1.6% non-symmetric (penalty and free-surface
terms), so CG stalls and GCR is needed.

| configuration | S | M | L |
|---|---|---|---|
| GCR(60), local penalty | fails | fails | fails |
| GCR(60), `iterMax_DR = 300` | 87.1k, 108 s | fails | fails |
| GCR(60), `:mean` | **14.7k, 22 s** | fails at step 6 | 19.6k, 53 s |
| GCR(60), `:mean`, adaptive floor | 19.1k, 28 s | 9.9k, 28 s | 14.8k, 53 s |
| DR, `:mean` (for comparison) | 43.7k, 59 s | 6.4k, 17 s | 8.1k, 29 s |

GCR is 2.7× faster than DR at S, but slower from M onward, because freezing the plastic state
turns the outer loop into a Picard iteration that needs more passes. Not run on the GPU.

## Code state

All changes are uncommitted, in `src/DYREL/solver_VS.jl` and `src/DYREL/constructors.jl`. New
keywords of `solve_VariationalDYREL!`, all defaulting to the current behavior:

| keyword | recommendation |
|---|---|
| `strict_convergence` | keep; consider making it the default |
| `penalty_viscosity = :capped`, `penalty_ratio` | keep |
| `penalty_viscosity = :mean / :geomean / :floored`, `penalty_floor_fraction` | remove (`:capped` with large R covers `:mean`) |
| `inner_tolerance_floor`, `adaptive_inner_tolerance` | remove |
| `momentum_restart_every`, `residual_check_every`, `stagnation_window` | remove |
| `inner_solver = :pcg / :gcr`, `gcr_depth` | keep only if M1 or K2 need it; otherwise remove |
| `viscosity_relaxation_PH` | remove |

`test/test_variational_dyrel.jl` has a "solver options" testset covering each keyword (31/31
pass). The pre-existing testsets "partial-volume rows" and "cut-cell penalty" fail with a
ParallelStencil `##META` error on clean `HEAD` as well. The full experimental diff, including
removed experiments, is in `Rift_files/convergence/experiments_full.patch`.

## Next steps

1. Repeat `:capped` R = 10 combined with `iterMax_DR ≈ 2·nx` on all grids. Both converged
   everywhere on their own; together they may be both robust and cheaper.
2. Repeat your settings with `iterMax_DR = 500` on the GPU, and run them at S, M and L.
3. Measure the cost of a global dot product relative to one DR iteration on the GPU, to choose
   between a Krylov inner solve (K1) and residual-minimizing acceleration of DR (K2).
4. Design note and prototype for the two-level coarse correction (M1), the only option that can
   stop iterations growing with resolution.
5. Optional: refresh the capped penalty once per Powell–Hestenes pass with relaxation. A refresh
   every DR iteration is not advisable: it changes the operator under the damping estimates.
