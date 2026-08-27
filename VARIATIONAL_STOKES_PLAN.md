# 2D Variational Stokes — audit against Larionov, Batty & Bridson (2017)

Auditing the 2D implementation against the local paper
[`Larionov2017.pdf`](Larionov2017.pdf), *Variational Stokes: A Unified
Pressure-Viscosity Solver for Accurate Viscous Liquids* (ACM TOG 36(4), 101).

The implementation work is tracked separately in
[`VARIATIONAL_STOKES_COUPLED_2D_PLAN.md`](VARIATIONAL_STOKES_COUPLED_2D_PLAN.md).

## Scope

The paper solves **unsteady** Stokes (backward Euler; the `ρ/Δt` term is what makes
the system SPD) as one direct solve in `(τ, p)` after eliminating `u` — Eqs. 20-21.
JustRelax solves **steady-state** Stokes by accelerated pseudo-transient iteration.
A one-to-one correspondence is impossible by construction.

What is transferable, and therefore what "implemented correctly" means here, is the
**embedded-boundary volume-fraction discretization of Section 5** (Eqs. 15–25,
Sections 5.1–5.3): the `W` weights, the null-space elimination, and the free-surface
conditions that follow from them.

The mathematical reference is the checked-in [`Larionov2017.pdf`](Larionov2017.pdf):
Eq. 15 is the weighted variational functional, Eq. 16 its saddle-point system, and
Eqs. 19–21 the combined free-surface/solid system after eliminating velocity. In
particular, Eq. 21 uses distinct liquid weights for velocity faces (`W_L^u`), cell
pressures (`W_L^p`), and stress locations (`W_L^τ`); the
`P⁻¹(W_L^u)⁻¹W_F^u` factor belongs inside the eliminated-velocity blocks.

Important scope boundary: the paper’s Eqs. 5–21 contain no gravity or other body
force. Acceleration enters through the initial velocity `u*` and the `P W_L^u` mass
term. The hydrostatic test below is therefore an additional steady geodynamics test,
not a direct exactness test specified by the paper; its body-force discretization
needs a separate derivation.

## Status

| Item | State |
|---|---|
| Rigid-body invariance (translation, rotation) | verified exact |
| Buoyancy double-counts the rock fraction | **fixed** |
| `Inf` viscosity in all-air cells poisons the solve | **fixed** |
| `air_phase` never forwarded by any caller | wired everywhere; Volcano regression passes |
| Hydrostatic equilibrium not exact | steady body-force discretization open |
| Continuity uses unmasked velocity | open; isolated face mask rejected |
| Missing `W_L^{u,-1}` in the velocity update | open; requires coupled derivation |
| Invalid pressure DOFs remain in the variational pressure row | **fixed; weighted metric verified** |

## Verification harness

`variational_stokes_verification.jl` (repo root). Run with:

```
julia --project=test -t 6 variational_stokes_verification.jl
```

The rigid-body check follows the paper's Section 6.2 exactness claim for linear
solutions on irregular domains. The hydrostatic check is an additional steady-force
regression; it is included because this package uses the variational kernel for
geodynamic gravity as well:

- **A. Rigid body.** Free-floating blob (non-grid-aligned circle) in vacuum, zero
  gravity, rigid velocity seeded wherever `ϕ > 0`. Rigid motion has zero strain
  rate, so stress, pressure and the momentum residual must all vanish and the
  velocity field must be left untouched.
- **B. Hydrostatic.** Fluid layer under a flat free surface at height `y_s`, swept
  across one cell in sub-grid steps. Exact solution is `V = 0`,
  `P = ρg(y_s - y)`. Reported as *surface misplacement in cells*,
  `(P_num - P_exact)/(ρg·Δy)` — an exact scheme gives `0` at every `δ` and every `n`.

The regression wrapper is `test/test_variational_free_surface.jl`.

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

### 2D weight mapping from the paper to the implementation

For the 2D staggered layout, the current `RockRatio` fields have the natural
interpretation `ϕ.center → W_L^p`, `ϕ.vertex → W_L^τ` for corner shear stresses, and
`ϕ.Vx`/`ϕ.Vy → W_L^u` for face velocities. The masked finite differences already
apply the center and vertex weights to pressure and stress terms. The body-force
term, however, averages `ρg` with center weights and does not apply the corresponding
velocity-face weight. Equation 16 requires that row to be assembled consistently with
the face control volume; Equations 20–21 then require the same face operator when
velocity is eliminated. This is why changing only the body-force term, or inserting
an isolated `inv(ϕ.Vx)`/`inv(ϕ.Vy)` in the pseudo-transient update, is insufficient.

### Root cause: the rock fraction was counted twice in the buoyancy

`compute_ρg!` averages density over **all** phases including air
(`fn_ratio(compute_density, ...)`, no air exclusion anywhere in
`src/rheology/BuoyancyForces.jl`), giving `ρg = f·ρ_rock + (1-f)·ρ_air`. The
variational momentum residual then multiplies that by the rock fraction *again*
(`av_ya(ρgy, ϕ.center)`, `src/variational_stokes/VelocityKernels.jl:210` and five
sibling sites). `update_rock_ratio!`
sets `ϕ.center` to that same fraction `f`.

Body force in partial cells was therefore **`f²ρ_rock` where it should be `f·ρ_rock`**.
Every other field weighted by `ϕ.center` — `P`, `τ` — is a pure-rock quantity, so
density was the odd one out.

Confirmed by direct probe, not just by reading: the discrete balance
`P = Δy·(ρg₁ϕ₁ + ρg₂ϕ₂)/2` predicted 0.009826 against a measured 0.0098160.

### Two traps found along the way

1. **`correct_phase_ratio` returns an all-zero ratio for a 100%-air cell.** That is
   correct for density (no rock, no weight) and is pinned by `test/test_rheology.jl:459`.
   Nearly/all-air cells could nevertheless divide a zero non-air sum by zero; the
   normalization now guards that case, while viscosity retains the original
   all-air ratio so the harmonic mean remains finite.
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
  stays finite. Used at both viscosity call sites. `correct_phase_ratio` now guards
  zero non-air sums.
- `src/variational_stokes/Stokes2D.jl:101` — forward `air_phase` to `compute_ρg!`.
- `src/ext/{CUDA,AMDGPU}/2D.jl` — forward the kwarg through the 2D GPU methods.
- All variational-Stokes callers now pass `air_phase`; the iterative `update_ρg!`
  paths also preserve it for non-constant-density materials.
- The verification harness now passes its solver options directly, so `iterMax`,
  `viscosity_cutoff`, and convergence settings are actually exercised.
- The public 2D dispatch wrapper now correctly forwards variadic keywords,
  which is required for `air_phase` to reach the solver implementation.

Result, with `air_phase` supplied:

| δ | before | after |
|---|---|---|
| 0.00 | 0.000 | 0.000 |
| 0.25 | −0.224 | −0.125 |
| 0.50 | −0.376 | −0.250 |
| 0.75 | −0.155 | **0.000** |
| spurious max\|V\| | 1e-6 … 2.3e-4 | **2.1e-13** |

Rigid-body invariance unchanged (1e-16). `test_rheology`, `test_rockratio`,
`test_mask`, `test_mini_kernels`, and `test_sinking_block` pass. Runic clean.

The rheology suite now also checks the parallel `compute_ρg!` and non-constant-density
`update_ρg!` phase-ratio paths with an air phase, verifying that both return the
non-air material value.

The Volcano failure was traced to its nested `kwargs = (; ...)` call: the solver
options were silently ignored by `_solve_VS!`. Passing those options directly now
restores the intended free-surface configuration, and the regression passes with
the corrected `air_phase` path.

## What is left

### 1. Forward `air_phase` from the callers — completed

Ten miniapps plus `test_Volcano2D` define an `air_phase` variable and now pass it
to `solve_VariationalStokes!`. The solver’s iterative buoyancy refresh also carries
the value through `update_ρg!`. `test_Volcano2D` passes solver options directly so
they are not discarded as an unused nested keyword tuple.

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

Wiring these up shifts benchmark values, including `test_Volcano2D`'s reference.

**Open design question:** the variational solver cannot be correct without knowing
the air phase — it is already required to build `ϕ`. A default of `air_phase = 0`
therefore silently produces wrong buoyancy. Making it mandatory is breaking but
fail-fast; leaving it optional keeps a footgun.

### 2. Verification harness — completed

The harness is now included by `test/test_variational_free_surface.jl` and checks
rigid-body invariance, finite hydrostatic results, negligible velocity drift, and
the current documented pressure-residual pattern across three resolutions.

`compute_variational_P!` now restores the complete 2D material-dependent pressure
dispatch with the §5.3 invalid-pressure elimination inside each branch. It covers
compressible, thermal, melt-dependent, and plastic-pressure inputs while preserving
the existing `_compute_P!` formulas. Volcano passes 1/1 and the variational harness
passes 16/16 with this dispatch. Its pressure residual now receives the center weight
`W_L^p` before relaxation in every branch; both regressions remain unchanged.

### 3. Steady body-force discretization — separate from the paper audit

The residual error after the fix is exactly `−δ/2` cells when the surface sits in the
**lower half** of a cell. There `isvalid_c` fails (the upper `Vy` control volume is
dry), `P` is pinned to zero in a cell that still holds real rock and real density,
and the head of the surface layer is lost.

Earlier hand algebra suggested the paper's `W^u` weighting of the momentum row:
body force at a face =

    (ϕ-weighted average of ρg over the two adjacent cells) × (face fraction ϕ.Vy)

rather than an average of cell-centre weights. Checked by hand at δ = 0, 0.25, 0.5,
0.75 — exact at all four, including reproducing the accidental exactness the current
scheme already has at δ = 0.75. The ϕ-weighted density average is needed because a
plain average against a zero-density air cell halves the result.

That interpretation is not directly mandated by Larionov2017 because gravity is
absent from its variational system. The normalized upper-face candidate gives exact
hydrostatic results in the focused flat-interface harness, but is not yet landed: it
currently destabilizes the Volcano multiphase regression. The correct steady force
treatment still needs an independent derivation compatible with the pseudo-transient
solver.
The simpler direct candidate `ϕ.Vx * av_xa(ρgx)` / `ϕ.Vy * av_ya(ρgy)` was also tested
against the harness: it merely phase-shifted the hydrostatic residual and failed 4/16
checks, confirming that the force row cannot be corrected independently of the other
weighted operators.

The face data itself is not merely a point sample: JustPIC constructs `Vx`/`Vy` from
the half-cell control volume surrounding each velocity face, which is the natural
discrete analogue of `W_L^u`. The failed direct candidate therefore indicates a
remaining indexing or operator-consistency issue between that face row and the
center/vertex pressure-stress rows, rather than an invalid face fraction.

The staggered index audit rules out an off-by-one explanation: the update at
`Vx[i + 1, j + 1]` corresponds to `ϕ.Vx[i + 1, j]`, and the update at
`Vy[i + 1, j + 1]` corresponds to `ϕ.Vy[i, j + 1]`. Those are the locations used by
the direct candidate. What remains is algebraic coupling between the face row and
the center-weighted pressure/stress rows.

Masking the diagonal pseudo-transient viscosity average
(`av_xa(ητ, ϕ.center)` / `av_ya(ητ, ϕ.center)`) passed the 16-case harness but produced
`NaN` in the Volcano regression. It is therefore also not a safe standalone change;
the preconditioner must be re-derived together with the weighted momentum operator.

A first coupled trial replaced each force row by `W_L^u f` and divided the velocity
update by the corresponding face weight, implementing the visible
`(W_L^u)⁻¹` factor from Eq. 21. It was reverted after the rigid-body case produced
`NaN`: the current binary validity rule admits arbitrarily small positive face
fractions, so the inverse is unbounded before the reduced null-space system is
assembled. This is evidence that the inverse belongs in a coupled reduced operator,
not a pointwise update; no production code from this trial remains.

A second prototype applied the existing `10⁻⁵` sliver cutoff consistently to masked
stencils, validity checks, force rows, and face-mass inverses. It still produced
`NaN` in the multiphase Volcano regression. Thus the failure is not fixed by tuning
the geometric active threshold: the pseudo-transient diagonal and the weighted
couplings must be assembled as one reduced operator. This prototype was also
reverted; the passing solver does not use a pointwise face inverse.

The paper's §5.3 rule removes a DOF when its volume weight is zero. In this code,
`isvalid_vx` and `isvalid_vy` currently implement only `ϕ > 0`, while the geometric
fraction fields can contain very small positive slivers. A future reduced operator
must therefore define one active-face policy and use it consistently in four places:
the face weight in the momentum row, the face-weighted divergence, pressure/stress
coupling, and the diagonal pseudo-time preconditioner. A cutoff in only the inverse
would create a different operator; a cutoff only in `isvalid_*` would still leave
the fractional stencil contributions. This is the next implementation boundary,
not a safe scalar-kernel patch.

The remaining implementation is consequently larger than a force-kernel replacement:
the current solver stores one pressure value at each cell center but applies it as
`G W_L^p p`. In a partially filled cell, exact hydrostatic balance requires deciding
whether that value represents the center pressure or the wet-volume average. The
paper does not resolve this for gravity because its variational system has no body
force term. A correct steady implementation must therefore introduce a consistent
pressure/force reconstruction (and then use the same face mass in the pseudo-time
operator) before the hydrostatic exactness test can be made a production requirement.

### 3a. Concrete 2D reduced-operator target

For a cell `(i,j)`, the continuity row must be assembled from the same face DOFs
used by the momentum rows:

```
W_L^p [ (W_L^{u,x} u_x)ᵢ₊₁/₂ − (W_L^{u,x} u_x)ᵢ₋₁/₂ ]/Δx
    + [ (W_L^{u,y} u_y)ᵢ,ⱼ₊₁/₂ − (W_L^{u,y} u_y)ᵢ,ⱼ₋₁/₂ ]/Δy = 0.
```

The corresponding momentum rows must contain the same active face set, with the
force represented as a liquid-face average times `W_L^u`. The pseudo-transient
update then needs the diagonal of the assembled weighted face operator, including
the density/mass factor, rather than `av_xa(ητ)` or `av_ya(ητ)` alone. Pressure
cells and stress locations coupled only to inactive faces must be eliminated before
these rows are applied. This is the matrix-free equivalent of the paper's §5.3
reduced system and is the implementation target for the next patch.

Acceptance criteria for that patch are: exact rigid translation and rotation;
finite Volcano iterations; no update from inactive faces; weighted continuity and
momentum rows using identical face indices; and no hydrostatic baseline changes
until the independent body-force reconstruction is validated.

A diagnostic implementation of only the weighted continuity stencil failed the first
criterion immediately (`rel. change ≈ 1.91` for translation and `0.11` for rotation)
and was reverted. This demonstrates that the current stress/momentum discretization
is not the adjoint partner of a face-weighted divergence. The continuity change must
therefore be landed together with the corresponding weighted stress divergence,
pressure coupling, and active-DOF elimination.

Repeating that experiment after restoring the variational pressure dispatch and
center-weighted pressure relaxation still failed (`rel. change ≈ 1.54` for
translation and `0.077` for rotation). The failure is therefore not repaired by the
pressure metric alone: the current masked pressure/stress gradients use a different
discrete inner product from a face-weighted divergence. The complete adjoint pair,
including its boundary/null-space rows, remains to be derived before any weighted
continuity kernel can be retained.

### 4. Open leads from the code audit — tested where possible

- **Continuity uses unmasked velocity.** `compute_∇V!`
  (`src/variational_stokes/VelocityKernels.jl:6`) calls the plain `div` from
  `src/MiniKernels.jl:57` with no `ϕ`, gated only by a binary `isvalid_c`. The
  paper's continuity row is `W_L^p Gᵀ W_F^u u` — velocities weighted by the face
  fraction *before* differencing. A binary gate is not a fractional weight. A first
  2D implementation using the correct `ϕ.Vx`/`ϕ.Vy` indices destroyed rigid-body
  invariance, so the face-weighted divergence cannot be inserted independently of
  the staggered stress and pressure operators.
- **Pressure is unweighted, deliberately.** `src/variational_stokes/PressureKernels.jl`
  is 100% commented out, under the claim that pressure needs no masking because it is
  "already masked when calculating the velocity fields and the stress fields". The
  paper's `A22` carries `W_L^p` on both sides. That claim is checkable and is exactly
  the sort of thing that holds in the interior and fails at the surface.
  The solver now dispatches through `compute_variational_P!`, which restores the
  complete 2D family and gates invalid pressure cells in every branch. The formulas
  still come from the generic `_compute_P!`; only the dispatch and §5.3 elimination
  are variational-specific. Coupling its pressure metric to the weighted continuity
  row remains open; the pressure metric itself is now center-weighted.
- **`W_L^{u,-1}` appears to be missing.** Every eliminated-velocity block of Eq. 21
  carries `P⁻¹(W_L^u)⁻¹W_F^u`. The velocity update is
  `V += R·ηdτ / av_xa(ητ)` with no face-weight inverse. Under PT this factor may be
  absorbed into a consistently defined diagonal pseudo-time operator, but inserting
  `inv(ϕ.Vx)` or `inv(ϕ.Vy)` alone is not the paper's discretization and changes the
  free-surface scaling. Highest-value item to settle analytically.
- **`W_F` (fluid-vs-solid) does not exist.** `RockRatio` encodes only rock-vs-air, so
  the combined Eq. 19 is really only the free-surface half; solid boundaries are left
  to conventional `flow_bcs!`. Defensible for geodynamics with grid-aligned walls,
  but it should be stated rather than implied.
- **§5.2 stress reduction and §5.1 non-zero traction BCs are absent.** The former is
  a reasonable deviation (JustRelax is compressible-capable, so `τ` is not traceless);
  the latter simply means no surface tension.
- **§5.3 null-space elimination looks faithful.** `isvalid_c` requiring all adjacent
  faces to be wet matches the paper's tagging rule closely.

## Suggested order for the current 2D scope

1. Decide whether `air_phase` should be mandatory. (The current default remains
   backward-compatible but can silently produce wrong buoyancy.)
2. Settle the `W_L^{u,-1}` question analytically — it determines whether items 3 and 4
   above are separate patches or one coherent rederivation of the momentum row.
3. Independently derive the steady gravity discretization, then update affected
   benchmark baselines only if the coupled 2D operator remains stable.
4. Weight continuity by the face fractions; revisit the commented-out pressure masking.
