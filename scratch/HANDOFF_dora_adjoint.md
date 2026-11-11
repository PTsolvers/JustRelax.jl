# Handoff: slow convergence of the variational DYREL adjoint (DoraMairaVerticalWeakZone)

## Goal and constraints
- Investigate why the variational DYREL adjoint in `SubductionShearZone/DoraMairaVerticalWeakZone.jl` converges slowly.
- Stay with pseudo-transient/DYREL. Penalty sweeps already failed; do not repeat them as the main next step.
- Keep experiments separate from production code. No production solver or example changes have been made. Preserve the user's modified example file.
- Check the run status before launching anything new.
- Order of work: (1) finish and report the two-control comparison (iterMax_DR 50000 vs 1000): runtimes, iterations, final residuals; (2) if shorter blocks don't help, look into damping estimates, pseudo-time steps, and whether the preconditioner matches the nonlinear adjoint operator.
- The user stopped the last session ("stop"). Don't relaunch long GPU jobs without their go-ahead.
- Don't run Runic on JustRelax.jl unless asked. Commit/push only when asked.

## Where things are
- Login: `ssh octopus`. It has no GPU, nvidia-smi or squeue. Compute: `ssh octopus 'ssh node04 ...'` (4× GTX TITAN X, CUDA.jl devices 0–3, 12 cores).
- Workspace: `/home/cschuler/work2/JustRelaxAdjointExamples` (same as `/work2/cschuler/...`). Julia: `/home/cschuler/.julia/juliaup/julia-1.13.2+0.x64.linux.gnu/bin/julia`, `--project=.`
- Active package source (read-only, installed): `/home/cschuler/.julia/packages/JustRelax/CBHPP/src/DYREL/adjoint/{solver_adjoint_VS.jl, adjoint_solver_utils.jl}`. This is NOT the `/work2/cschuler/JustRelax.jl` checkout.
- Experiments are in `experiments/dora_adjoint/`:
  - `compare.jl`: two-control harness. It includes the example with `NO_AUTORUN`, then builds a renamed, instrumented in-process copy (`experiment_adjoint!`) with `Base.include_string(JR, ...)` and `replace_once`. Env vars: `DORA_CHECKPOINT`, `DORA_RESULTS`, `DORA_DEVICE`, `DORA_NX` (128 → grid 128×256).
  - Checkpoint: `results_retry/step10.jls` (sha256 dac2a087…1767d6). It holds `(pos[1:11], kwargs)` = (stokes, stokes_ad, ρg, dyrel, flow_bcs, phase_ratios, ϕ, rheology, args, grid, dt). After deserializing, restore `args.P === stokes.P`.
  - `README.md`: the protocol and a source audit.
  - **Mine, never run to completion:** `damping_probe.jl`, `damping.log`, `results_damping/` (partial). Local copy: `/private/tmp/claude-501/-Users-cschule1-Documents-work-JustRelax-jl/06b25542-a898-48a5-a434-5578aedd7f75/scratchpad/damping_probe.jl`

## Solver mechanics (from the source)
- DYREL update: `dVdτ ← α·dVdτ + R/D`, `ΔV = dVdτ·β·dτ`, `dτ = 2/√(λmax·amp)·CFL`, `β = 2dτ/(2+c dτ)`, `α = (2−c dτ)/(2+c dτ)`. Damping is `c = 2√λmin·c_fact` (c_fact 0.5, CFL 0.99; the example sets only `γfact=20, ϵ=1e-3`).
- λmin: every `nout` iterations, the Euclidean Rayleigh quotient of the diagonally preconditioned operator, `Σ dV·(r/D − r0) / Σ dV²` (`adjoint_λminV`). The forward solver uses the same form.
- Powell–Hestenes outer loop: `λP += relax·γ_eff·P` after each inner block. The inner target is `err·rel_drop`. `rel_drop` is multiplied by 0.1 (floor 1e-3) when `err > 1.05·err_min`. The inner loop test is `itPT ≤ iterMax_DR`, so each block does 50001 or 1001 updates.
- Penalty: `penalty_scaling=:local, penalty_factor=80`. `configure_adjoint_penalty!` rebuilds D and the Gershgorin λmax from the frozen secant viscosity. The nonlinear tangent enters only through the amplification factor in dτ.
- A fresh adjoint inherits the forward damping c (≈0.0024).
- Known issue: `damping.stored` is not invalidated after λP updates.

## Measurements: 50000 baseline (compare.jl, pid 3366670 on node04, results in `results_retry/`)
| iterations | seconds | outer error |
|---|---|---|
| 50001 | 118.8 | 1.199 |
| 100002 | 237.1 | 0.749 |
| 150003 | 355.7 | 0.462 |
| 200004 | 476.9 | 0.284 |
| 250005 | 596.6 | 0.174 |
| 300006 | 715.5 | 0.107 |

- Every block runs to the iteration limit. The error falls by ≈0.61 per 50k-iteration block, at ≈2.39 s per 1000 iterations. `Ry` dominates.
- c rises from 0.0024 to ≈0.040–0.042, with dτ ≈ 1.0 and D spanning 4.6e12 to 4.6e18.
- Last checked about 11:34 CET on 2026-11-11, still in the baseline. My projection (not measured): ≈8e-4 at the 800k budget, finishing about 11:54; the 1000 trial would then run until about 12:27.
- **First action:** read `results_retry/summary.txt` and `trial*.records`, and report both trials. The "warmup" line in summary.txt is an expected 20-iteration non-convergence, not a failure.
- Other agent processes of the user (codex, claude, a `sleep 180` poller) also run on node04, so another session may be watching the run.

## Hypothesis (untested): the adjoint is overdamped
- The observed decay is ≈8e-6 to 1e-5 per iteration at c≈0.041, dτ≈1. When overdamped, rate ≈ λ/c, so λ_slow ≈ 3–4e-7 and critical damping is c ≈ 2√λ ≈ 1.2e-3. That is close to the inherited forward c (0.0024) and about 30× below what the adaptive estimate sets.
- Suspected cause: the Euclidean quotient of the non-symmetric D⁻¹A, with D spanning 6 orders of magnitude, overestimates λmin. Alternative estimate: the D-weighted quotient `dVᵀ(A dV) / dVᵀ(D dV)`.
- Predicted speed-ups: frozen c=0.0024 gives ≈1.8e-4 per iteration (~19×); c≈1.2e-3 gives ≈6e-4 per iteration (~65×), if the system stays stable.
- Test: `damping_probe.jl` replays step10.jls. `DORA_TRIALS` takes `kind[:c][@budget]` entries separated by `;`. Kinds are `adaptive | frozen | fixed:c | dweighted`, with a default budget of 300000. It starts with a 20-iteration warm-up, records both quotients in every mode, and writes `$DORA_RESULTS/{progress.records, summary.txt, sources.txt}`.
  Launch (only with the user's OK, on a free GPU):
  ```
  cd /home/cschuler/work2/JustRelaxAdjointExamples && DORA_DEVICE=2 DORA_RESULTS=experiments/dora_adjoint/results_damping DORA_CHECKPOINT=experiments/dora_adjoint/results_retry/step10.jls DORA_TRIALS="adaptive@20000;frozen;fixed:0.0012;dweighted" nohup julia --startup-file=no --project=. experiments/dora_adjoint/damping_probe.jl > experiments/dora_adjoint/damping.log 2>&1 &
  ```
  Clear the partial `results_damping/` first (the progress file is opened in append mode).
- Remaining questions after that: dτ/amplification versus the true tangent λmax, and whether D (built from secant viscosity) matches the nonlinear adjoint operator.
