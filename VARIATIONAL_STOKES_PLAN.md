# Variational Stokes — audit against Larionov, Batty & Bridson (2017)

Auditing `src/variational_stokes/` against *Variational Stokes: A Unified
Pressure-Viscosity Solver for Accurate Viscous Liquids* (ACM TOG 36(4), 101).

## Scope

The paper solves **unsteady** Stokes (backward Euler; the `ρ/Δt` term is what makes
the system SPD) as one direct solve in `(τ, p)` after eliminating `u` — Eqs. 20-21.
JustRelax solves **steady-state** Stokes by accelerated pseudo-transient iteration.
A one-to-one correspondence is impossible by construction.

What is transferable, and therefore what "implemented correctly" means here, is the
**embedded-boundary volume-fraction discretization of §5** (Eqs. 15-25, §5.1-5.3):
the `W` weights, the null-space elimination, and the free-surface conditions that
follow from them.

## Status

| Item | State |
|---|---|
| Rigid-body invariance (translation, rotation) | verified exact |
| Buoyancy double-counts the rock fraction | **fixed** |
| `Inf` viscosity in all-air cells poisons the solve | **fixed** |
| `air_phase` never forwarded by any caller | open — fix is inert until done |
| Hydrostatic equilibrium not exact | partly fixed; candidate fix derived, not landed |
| Continuity uses unmasked velocity | open, untested |
| Missing `W_L^{u,-1}` in the velocity update | open, untested |
| 3D variational path has zero callers | open |

## Verification harness

`variational_stokes_verification.jl` (repo root). Run with:

```
julia --project=test -t 6 variational_stokes_verification.jl
```

Two exactness checks, both taken from the paper's §6.2 (which claims *exact*
solutions, to numerical precision, for problems with linear solutions on irregular
domains — not merely convergent ones):

- **A. Rigid body.** Free-floating blob (non-grid-aligned circle) in vacuum, zero
  gravity, rigid velocity seeded wherever `ϕ > 0`. Rigid motion has zero strain
  rate, so stress, pressure and the momentum residual must all vanish and the
  velocity field must be left untouched.
- **B. Hydrostatic.** Fluid layer under a flat free surface at height `y_s`, swept
  across one cell in sub-grid steps. Exact solution is `V = 0`,
  `P = ρg(y_s - y)`. Reported as *surface misplacement in cells*,
  `(P_num - P_exact)/(ρg·Δy)` — an exact scheme gives `0` at every `δ` and every `n`.

This should eventually become `test/test_variational_free_surface.jl`.

## What was learned

### Rigid-body invariance passes, and is not enough

Translation and rotation are retained to 1e-16, and the result is **unchanged across
`η_air` = 1e-6 … 1e-1**. That air-viscosity independence is the meaningful part: the
air exerts no traction on the liquid, so the failure the paper is about — decoupled
solvers destroying rotation — is genuinely absent.

But this test passes just as well for a *voxelized* boundary: a staircased blob is
still a blob. It cannot distinguish sub-grid weighting from staircasing, which is why
check B exists.

### Hydrostatic equilibrium is not exact

Baseline, before any fix (resolution-independent, so `O(Δx)` and not converging away):

| δ | n=32 | n=64 | n=128 | spurious max\|V\| |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 2.1e-13 |
| 0.25 | −0.224 | −0.224 | −0.224 | 1.1e-06 |
| 0.50 | −0.376 | −0.376 | −0.375 | 1.2e-06 |
| 0.75 | −0.155 | −0.158 | −0.155 | 2.3e-04 |

The velocity column matters as much as the pressure: a static equilibrium was
drifting at up to 2e-4 — the classic sticky-air artifact.

### Root cause: the rock fraction was counted twice in the buoyancy

`compute_ρg!` averages density over **all** phases including air
(`fn_ratio(compute_density, ...)`, no air exclusion anywhere in
`src/rheology/BuoyancyForces.jl`), giving `ρg = f·ρ_rock + (1-f)·ρ_air`. The
variational momentum residual then multiplies that by the rock fraction *again*
(`av_ya(ρgy, ϕ.center)`, `src/variational_stokes/VelocityKernels.jl:210` and five
sibling sites; 3D uses the same `av_x(fx, ϕ.center)` pattern). `update_rock_ratio!`
sets `ϕ.center` to that same fraction `f`.

Body force in partial cells was therefore **`f²ρ_rock` where it should be `f·ρ_rock`**.
Every other field weighted by `ϕ.center` — `P`, `τ` — is a pure-rock quantity, so
density was the odd one out.

Confirmed by direct probe, not just by reading: the discrete balance
`P = Δy·(ρg₁ϕ₁ + ρg₂ϕ₂)/2` predicted 0.009826 against a measured 0.0098160.

### Two traps found along the way

1. **`correct_phase_ratio` returns an all-zero ratio for a 100%-air cell.** That is
   correct for density (no rock, no weight) and is pinned by `test/test_rheology.jl:459`.
   But it sends the *harmonic* viscosity mean to `Inf`, and `ητ = maxloc(η)` then
   spreads `Inf` into cells that do carry rock, NaN-ing the solve. Passing
   `air_phase` to `solve_VariationalStokes!` used to poison it outright.
2. **`Geometry(ni, li)` takes its spacing from the global ImplicitGlobalGrid grid**
   once `init_global_grid` has run — not from the `ni` passed to it. A refinement
   study must re-init the global grid per resolution
   (`finalize_global_grid(; finalize_MPI = false)`), or every resolution silently
   inherits the first one's spacing. An earlier version of the harness got this
   wrong and its n=64/128 columns were meaningless.

## What was changed

Nothing is committed yet.

- `src/rheology/BuoyancyForces.jl` — `compute_ρg!` for `PhaseRatios` gains an
  `air_phase` kwarg (default `0`, so existing callers are unaffected) that drops the
  air phase and renormalizes over the rest. Threaded into both `compute_ρg_kernel!`
  methods.
- `src/rheology/Viscosity.jl` — new `viscosity_phase_ratio` (line 640) wraps
  `correct_phase_ratio` and keeps an all-air cell's own ratio, so the harmonic mean
  stays finite. Used at both viscosity call sites. `correct_phase_ratio` itself is
  unchanged.
- `src/variational_stokes/Stokes2D.jl:101`, `Stokes3D.jl:73` — forward `air_phase`
  to `compute_ρg!`.
- `src/ext/{CUDA,AMDGPU}/{2D,3D}.jl` — forward the kwarg through the GPU methods.

Result, with `air_phase` supplied:

| δ | before | after |
|---|---|---|
| 0.00 | 0.000 | 0.000 |
| 0.25 | −0.224 | −0.125 |
| 0.50 | −0.376 | −0.250 |
| 0.75 | −0.155 | **0.000** |
| spurious max\|V\| | 1e-6 … 2.3e-4 | **2.1e-13** |

Rigid-body invariance unchanged (1e-16). `test_rheology`, `test_rockratio`,
`test_mask`, `test_mini_kernels`, `test_Volcano2D`, `test_sinking_block` pass.
Runic clean.

## What is left

### 1. Forward `air_phase` from the callers — blocks everything else

The fix is **inert in the repo as it stands**. Ten miniapps plus `test_Volcano2D`
each define an `air_phase` variable and use it for `update_rock_ratio!`, then omit it
from the `solve_VariationalStokes!` call, so the solver always sees `0`:

```
miniapps/subduction/2D/Subduction2D_VS.jl
miniapps/subduction/2D/Subduction2D_VS_ND.jl
miniapps/benchmarks/stokes2D/Volcano2D/Caldera2D.jl
miniapps/benchmarks/stokes2D/shear_band/ShearBand2D_variational.jl
miniapps/benchmarks/stokes2D/shear_band/ShearBand2D_variational_MPI.jl
miniapps/benchmarks/stokes2D/free_surface_stabilization/PlumeFreeSurface_VariationalStokes.jl
miniapps/benchmarks/stokes2D/free_surface_stabilization/RayleighTaylor2D_VariationalStokes.jl
miniapps/benchmarks/stokes2D/free_surface_stabilization/Crameri2D.jl
miniapps/benchmarks/stokes2D/StickyAirSubduction/VariationalSubduction2D.jl
miniapps/benchmarks/thermal_stress/Thermal_Stress_Magma_Chamber_nondim_VS.jl
test/test_Volcano2D.jl
```

Wiring these up will shift benchmark values, including `test_Volcano2D`'s reference.

**Open design question:** the variational solver cannot be correct without knowing
the air phase — it is already required to build `ϕ`. A default of `air_phase = 0`
therefore silently produces wrong buoyancy. Making it mandatory is breaking but
fail-fast; leaving it optional keeps a footgun.

### 2. Face-weighted body force — closes the gap to the paper's exactness claim

The residual error after the fix is exactly `−δ/2` cells when the surface sits in the
**lower half** of a cell. There `isvalid_c` fails (the upper `Vy` control volume is
dry), `P` is pinned to zero in a cell that still holds real rock and real density,
and the head of the surface layer is lost.

Hand algebra on the discrete balance says the exact treatment is the paper's `W^u`
weighting of the momentum row: body force at a face =

    (ϕ-weighted average of ρg over the two adjacent cells) × (face fraction ϕ.Vy)

rather than an average of cell-centre weights. Checked by hand at δ = 0, 0.25, 0.5,
0.75 — exact at all four, including reproducing the accidental exactness the current
scheme already has at δ = 0.75. The ϕ-weighted density average is needed because a
plain average against a zero-density air cell halves the result.

Not implemented: it rewrites the body-force term in six 2D kernels plus the 3D ones,
changes the discretization rather than fixing a bug, needs a guard for
`ϕ₁ + ϕ₂ = 0`, and shifts every variational benchmark. It is also four hand-checked
configurations, not a tested result.

### 3. Open leads from the code audit — identified, not yet tested

- **Continuity uses unmasked velocity.** `compute_∇V!`
  (`src/variational_stokes/VelocityKernels.jl:6`) calls the plain `div` from
  `src/MiniKernels.jl:57` with no `ϕ`, gated only by a binary `isvalid_c`. The
  paper's continuity row is `W_L^p Gᵀ W_F^u u` — velocities weighted by the face
  fraction *before* differencing. A binary gate is not a fractional weight.
- **Pressure is unweighted, deliberately.** `src/variational_stokes/PressureKernels.jl`
  is 100% commented out, under the claim that pressure needs no masking because it is
  "already masked when calculating the velocity fields and the stress fields". The
  paper's `A22` carries `W_L^p` on both sides. That claim is checkable and is exactly
  the sort of thing that holds in the interior and fails at the surface.
- **`W_L^{u,-1}` appears to be missing.** Every block of Eq. 21 carries
  `P⁻¹ W_L^{u,-1} W_F^u`. The velocity update is `V += R·ηdτ / av_xa(ητ)` with no
  division by `ϕ.Vx`. Under PT the inverse mass weight may be legitimately absorbed,
  but that inverse is what makes the traction-free condition correct rather than a
  `p = 0` / `τn = 0` split. Highest-value item to settle analytically.
- **`W_F` (fluid-vs-solid) does not exist.** `RockRatio` encodes only rock-vs-air, so
  the combined Eq. 19 is really only the free-surface half; solid boundaries are left
  to conventional `flow_bcs!`. Defensible for geodynamics with grid-aligned walls,
  but it should be stated rather than implied.
- **§5.2 stress reduction and §5.1 non-zero traction BCs are absent.** The former is
  a reasonable deviation (JustRelax is compressible-capable, so `τ` is not traceless);
  the latter simply means no surface tension.
- **§5.3 null-space elimination looks faithful.** `isvalid_c` requiring all adjacent
  faces to be wet matches the paper's tagging rule closely.

### 4. 3D

There are **no 3D callers of `solve_VariationalStokes!` anywhere** — every miniapp and
test above is 2D. The 3D path carries the same body-force pattern
(`av_x(fx, ϕ.center)`) and so had the same double-count, now fixed, but it is
otherwise unexercised. `docs/src/man/DYREL.md` also notes 3D variational support is
still pending. A 3D smoke test would be worth adding before trusting it.

Minor API inconsistency: 2D calls `compute_viscosity!(...; air_phase = air_phase)`
while 3D passes it positionally, `compute_viscosity!(stokes, phase_ratios, args,
rheology, air_phase, viscosity_cutoff)`.

## Suggested order

1. Forward `air_phase` from the callers, re-baseline the affected benchmarks, and
   decide whether it should be mandatory. (Do this before 2, or benchmarks get
   re-baselined twice.)
2. Promote the harness to `test/test_variational_free_surface.jl`.
3. Settle the `W_L^{u,-1}` question analytically — it determines whether items 2 and 3
   above are separate patches or one coherent rederivation of the momentum row.
4. Implement and test the face-weighted body force.
5. Weight continuity by the face fractions; revisit the commented-out pressure masking.
6. Add a 3D smoke test.
