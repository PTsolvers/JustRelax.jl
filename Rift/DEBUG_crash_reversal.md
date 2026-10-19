# Rift2D_Zwaan crash: investigation notes

Date: 2026-10-18. Investigation run locally on an RTX 5070 Ti (CUDA), at the script's resolution
`(nx, ny) = (112, 64)`, with `t_reverse_Myr = 1.0` and `t_rift = 15`.

## Summary

The crash is not caused by the particles, and it is not specific to the velocity reversal. It is a
mass-conservation hole in the variational (cut-cell) Stokes solver along the free surface. It
becomes fatal at the corners where the surface meets the lateral walls.

- A cut cell whose top face carries no rock loses its pressure row (`isvalid_c` needs all four
  faces active). Its side faces still carry a little rock, so they stay active velocity unknowns.
- A velocity face between two such cells has no continuity equation on either side. The solver
  uses it as a free mass source or sink.
- The converged "solution" then has the whole mantle circulating at ~100x the plate velocity. Crust
  is crushed into the wall corner, at ~6 % volume per step through the elastic bulk compliance.
  This ends in either `NaN detected in inner loop` or a garbage solution that collapses `dt`.

A second, independent bug in the script made things worse and made the crash timing look random:
`zero_air_velocity!` zeroed active free-surface velocities.

Both are fixed in `Rift2D_Zwaan.jl`. A matching patch to `src/variational_stokes/mask.jl` exists
but is **not applied**, because it changes solver semantics and breaks existing tests (see
"Package-level fix").

## Symptom

The run diverges inside `solve_VariationalDYREL!` at a random step. Observed failures:

| Run | Velocity zeroing | Failure |
|---|---|---|
| 1 | `zero_air_velocity!` (original) | step 28 (extension), NaN |
| 2 | original | step 159–164 (compression), dt collapse to 0.04 kyr, left wall corner |
| 3 | original | step 178 (compression), NaN, right wall corner |
| fix runs (3 of them) | ϕ-based zeroing | step 31–32 (extension), NaN, right wall |

The failure always sits at a lateral wall, in the crust or upper mantle, usually up to the corner
where the free surface meets the wall. λ reaches 1e3–1e4, P reaches 2e3–4e4 nondimensional (the
deepest lithostatic value is ~600), and velocities reach 50–1000x the plate rate.

## How it was diagnosed

- **Diagnostics before each solve**: marker-chain extrema and slope, cut cells per column, NaN
  and partition-of-unity checks on all phase ratios, rock-cell vs particle-phase mismatches,
  particles per cell, η / τ / P / T extrema.
- **Field dumps**: before each solve, and of the failed state on NaN or dt collapse.
- **Checkpoints every step** (`checkpointing_jld2` + `checkpointing_particles`). Restarting from
  checkpoint `it_0031` while keeping the stored velocity reproduces the failure **exactly**. Every
  hypothesis below was tested on that deterministic state.

## Findings

### 1. Particles are healthy before every failure

Just before each failing solve:

- particle phases agree with the marker chain (0–5 mismatching cells, all at the surface);
- every cell has 50–100 particles;
- particle-derived τ_o is within ~10 % of the grid τ;
- no NaNs anywhere.

Particles only spread the damage afterwards: with the garbage velocity, the FLIP stress update
drives τxx_o at the corner to −5562.

### 2. Bug in the script: `zero_air_velocity!`

It picks "air" faces from the **particle phase ratios**. That also zeroes free-surface Vy faces
of cut cells whose control volume is up to ~58 % rock, and those faces are real unknowns of the
variational solver. The count of such faces grows from 1 at step 20 to 16 at step 24 and ~55 by
step 140. Consequences:

- the solver's starting residual jumps from ~0.5 to 20–60 and stays there. The jumps coincide
  with the steps where these faces appear;
- particles and the marker chain get advected with zero vertical velocity at those faces;
- the evolution is chaotic enough that crash timing varies from run to run.

Fix (now in the script):

```julia
function zero_inactive_velocity!(stokes, ϕ)
    @views stokes.V.Vx[:, 2:(end - 1)] .*= ϕ.Vx .> 0
    @views stokes.V.Vy[2:(end - 1), :] .*= ϕ.Vy .> 0
    return nothing
end
```

With it, the starting residual is O(1) again (1.4 instead of 56 from the same state). On its own
it does **not** stop the crash: the runs then fail reproducibly at step 31.

### 3. Hypotheses ruled out on the deterministic step-31 state

| Change | Result |
|---|---|
| `λ_relaxation_PH/DR = 0.5` | NaN |
| `viscosity_relaxation = 1e-3` | converges to garbage, dt collapse |
| `γfact = 5` or `50` | NaN |
| `global_penalty = false` | NaN (outer loop) |
| dilation angle Ψ = 0 (instead of 10°) | garbage, dt collapse |
| no tensile cap (`DruckerPrager_regularised`), Ψ = 10 or 0 | garbage, dt collapse |
| no friction softening | NaN |
| `η_vp = 1e21` | garbage, dt collapse |
| `dt = 5 kyr` | never converges (150k iterations), dt collapse |
| ν = 0.49 (nearly incompressible) | NaN |
| ν = 0.5 | rejected by `DruckerPragerCap` (needs finite Kb); without the cap, NaN at the first residual, probably 0·Inf in `fn_ratio` (not verified) |
| moving grid: x-coordinates stretched with the plates every step, so walls move with the material and nothing crosses them | same crash at step 31 |
| no plasticity within 20 km of each wall (copies of phases 1–4 without plasticity) | runaway moves to the buffer edge |
| free-surface stabilisation off (`free_surface = false`) | NaN |
| give every face of a cut cell ϕ ≥ 1e-3, so slivers keep their pressure row | NaN |
| mirror the full state left↔right | runaway moves to the **other** wall: the code is symmetric and the state drives it |

The particle-related changes (moving grid, so no in/outflow and no wall injection) and the
rheology changes (cap, dilation, softening, regularisation, compressibility) all fail the same way.

### 4. Root cause: orphan velocity faces in surface cut cells

Look at the converged garbage solution (cap off, Ψ = 0):

- the whole asthenosphere flows sideways at Vx ≈ −40, against a plate velocity of ±0.35;
- the domain rises at Vy ≈ +15 almost everywhere;
- everything drains into the top-right corner, where Vy ≈ −230.

Per-cell divergence near that corner:

- the solver's continuity residual RP is ~1e-7, so it "converged";
- yet the actual ∇V is −70 to −140 in the rock cells below the corner and +3 to +15 across the
  mantle;
- this is balanced by elastic compressibility, ∇V = −ΔP/(Kb·dt), with Kb·dt ≈ 3.6 nondimensional
  (78 GPa × 14.7 kyr = 3.6e22 Pa s).

The source or sink sits in the surface sliver row (row 60, ϕc ≈ 0.005–0.05). Those cells have a
dry top face (`ϕ.Vy = 0`) and so no pressure row, but their side faces (`ϕ.Vx ≈ ϕc`) are active. A
face between two such cells is constrained by no continuity equation.

Tests that confirm it:

| Change | Result |
|---|---|
| zero every ϕ < 0.05 | passes step 31, runs to step 152, then fails on slivers with ϕc = 0.053, the same corner mechanism |
| prune orphan faces (below) | passes step 31; restarted from step 31 it ran to **step 385 (5.66 Myr, ~4.6 Myr of full compression)** with no failure, stopped only by timeout. The result looks physical: an inverted rift with a central pop-up and conjugate thrusts |

## Fix

Rule: a velocity face is an unknown only if it carries rock **and** borders at least one cell that
keeps its pressure row. A cell that keeps its pressure row never loses a face to this rule,
because each of its faces has that cell as a neighbour. So `isvalid_c` is unchanged and **one pass
is exact**: no iteration needed, and an in-place GPU update is race-free.

### In `Rift2D_Zwaan.jl` (applied)

1. `zero_air_velocity!` is replaced by `zero_inactive_velocity!(stokes, ϕ)` at every call site;
   the old function and its kernels are removed.
2. A new `prune_orphan_faces!(ϕ)` (two `@parallel_indices` kernels) is called right after every
   `compute_rock_fraction!(ϕ, chain, xvi, dxi)`.

A backup of the original script is at
`C:\Users\Albert\AppData\Local\Temp\claude\c--Users-Albert-Documents-JustRelax-jl\d78d7a01-9eb8-4a42-bfad-4429103e4725\scratchpad\run\Rift2D_Zwaan.backup.jl`.
The `Rift/` folder is untracked, so git cannot restore it.

### Package-level fix (patch saved, NOT applied)

`C:\Users\Albert\AppData\Local\Temp\claude\c--Users-Albert-Documents-JustRelax-jl\d78d7a01-9eb8-4a42-bfad-4429103e4725\scratchpad\run\mask_orphan_faces.patch`
puts the same rule in `src/variational_stokes/mask.jl`:

- `isvalid_vx/vy/vz` (2D and 3D) require a neighbouring retained cell;
- `isvalid_v` (2D), `isvalid_velocity`, and `isvalid_xz/xy/yz` use those predicates instead of raw
  face fractions. Without this second part, shear stress stays active at vertices next to dropped
  faces, and the starting residual jumps to ~84.

Every momentum-row consumer (DYREL, APT variational kernels, Gershgorin, masks, projection) goes
through these predicates, so this covers both variational solvers. On the step-31 state it
reproduces the script-level prune almost exactly and ran 148 steps without failure.

It was reverted because 5 CPU test files fail with it:

- `test_rockratio`, the mask assertions in `test_variational_stokes3D`, and
  `test_variational_operators_2D:349` encode the old rule ("a face is active iff ϕ > 0"). Expected
  for a deliberate semantic change.
- `test_variational_operators_2D:233–234`: residuals become zero because the hand-built ϕ has
  faces with no retained neighbour.
- `test_variational_free_surface`: rigid-body invariance ratio 0.94 instead of 1. Dropping orphan
  faces costs exact rigid-body invariance at the free surface. **This is a real trade-off.** The
  script-level fix has the same property.
- `test_PlumeFreeSurface_DYREL`: the reference regression values shift (maxVx 1.70e-9 vs 2.49e-9).

Whether to adopt this in the package, and how to update those tests, is a solver design decision.
The other tests in that batch passed: `test_mask`, `test_variational_dyrel` and
`test_variational_stokes_3D`.

## Current status

- **Full validation run**, fresh from step 0, with orphan-face pruning and `zero_inactive_velocity!`
  (same logic as the script, host-side implementation): at **step 765 (~11.2 Myr)** when these
  notes were written. That is ~10 Myr past the reversal, with no failure. Log:
  `C:\Users\Albert\AppData\Local\Temp\claude\c--Users-Albert-Documents-JustRelax-jl\d78d7a01-9eb8-4a42-bfad-4429103e4725\scratchpad\run\log_full.txt`,
  figures in the `out_full\fig\` folder next to it.
- **Smoke test of the edited `Rift2D_Zwaan.jl`** (GPU kernel version; only figdir, `Pkg.add` and
  `do_vtk` changed in a local copy): reached step 109, through the reversal into compression, with
  no failure before its 20-min timeout.
- `src/variational_stokes/mask.jl` is unchanged (reverted). No other repo files touched.

## Open issues

- **Spikes in the starting residual remain.** Whenever the surface crosses a cell row and faces
  switch on or off, the starting residual jumps to ~3–30 for a step. The solver recovers each
  time, but each spike costs iterations. Newly activated faces start from a zero velocity guess;
  seeding them from neighbours would likely remove the spikes.
- `semilagrangian_advection_markerchain!` runs with the default `conserve_mean = true`, which
  pins the mean surface height. That is probably not what you want with net inflow/outflow or
  during compression.
- The run is non-deterministic (random particle seeding and injection).
- Blocky viscosity patches near x ≈ 200 km, y ≈ −120 km (base of the lithosphere) in the
  compression phase. Unexplained, unrelated to the crash.
- **Not tested:** higher resolution, MPI, the cluster/GH200, and AMDGPU.
- At startup the CUDA runtime warns "You are using CUDA 13.4.0, but CUDA.jl was precompiled for
  CUDA 13.2.0". It was harmless here.

## Reproducing

All scripts live in
`C:\Users\Albert\AppData\Local\Temp\claude\c--Users-Albert-Documents-JustRelax-jl\d78d7a01-9eb8-4a42-bfad-4429103e4725\scratchpad\run\`:

- `rift_t.jl`: restartable copy with diagnostics and env switches;
- `diag.jl`: helpers (diagnostics, prune, mirror, regrid);
- `rheology_psi.jl`: rheology with env switches;
- `ckpt_keep2\it_0031\`: checkpoint that fails deterministically without the fix;
- `plotfail.jl`, `div.jl`: analysis of failure dumps.

```sh
# from the repo root; add RIFT_PRUNE=1 to apply the fix
RIFT_KEEPV=1 RIFT_CKPT=".../scratchpad/run/ckpt_keep2/it_0031" \
  julia --project=. --startup-file=no .../scratchpad/run/rift_t.jl
```

Environment switches: `RIFT_PRUNE`, `RIFT_CLIP` (+ `RIFT_PHI_TOL`), `RIFT_ACTIVATE`, `RIFT_FS`,
`RIFT_PSI`, `RIFT_CAP`, `RIFT_SOFT`, `RIFT_ETAVP`, `RIFT_NU`, `RIFT_DT`, `RIFT_GF`, `RIFT_GP`,
`RIFT_LRPH`, `RIFT_LRDR`, `RIFT_VR`. On failure the script writes `fail_*.jld2`.
