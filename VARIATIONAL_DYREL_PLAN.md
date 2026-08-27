# Plan: 2D Variational DYREL

Larionov et al. (2017) should be the mathematical reference, while PR #520 provides
the repository-specific variational-Stokes implementation baseline.

The key constraint is that variational DYREL must preserve the weighted coupled
operator, not merely add `ϕ` factors to standard DYREL kernels.

Larionov's formulation uses distinct weights:

- `W_L^u`: liquid volume at velocity faces;
- `W_L^p`: liquid volume at pressure cells;
- `W_L^τ`: liquid volume at stress locations;
- `P`: velocity mass/density operator;
- `M`: viscosity operator;
- `G`: pressure-gradient operator;
- `D`: deformation-rate operator.

The free-surface system is assembled from weighted terms such as:

```text
Dᵀ W_L^τ
G  W_L^p
W_L^u
```

Velocity elimination produces the coupled pressure/stress operators described in
Larionov et al., Eqs. 15–21. The paper explicitly identifies the staggered 2D
pressure, velocity, and stress placement and derives the volume-weighted formulation
for free surfaces.

Reference: [Larionov, Batty, and Bridson (2017)](https://cs.uwaterloo.ca/~c2batty/papers/Larionov2017/Larionov2017.pdf).

## Implementation status

- [x] Dedicated `solve_VariationalDYREL!` API and 2D backend dispatch scaffolding.
- [x] Port the historical solver and masked velocity-kernel source files.
- [x] Add the variational constructor, viscosity, and Gershgorin wiring.
- [ ] Reconcile their dependencies with PR #520 and validate the weighted operator.
- [ ] Weighted momentum diagonal estimation.
- [ ] Focused regressions and documentation completion.

Implementation note: the repository contains a historical `pa-dyrel_VS_2D` branch
with `src/DYREL/solver_VS.jl` and `src/DYREL/velocity_kernels_VS.jl`. Those files
are the starting point for the implementation, but they predate PR #520 and must be
adapted rather than copied unchanged.

Checkpoint: the API, constructor, solver-loop source, masked kernels, viscosity
helpers, and Gershgorin wiring are now staged for review. The next implementation
step is to reconcile the remaining mask/utility dependencies and verify the
weighted momentum diagonal against the PR #520 operators before enabling tests.

## 1. Define the mathematical DYREL form first

Write down the 2D weighted residuals before modifying kernels.

The variational DYREL path should approximate the weighted Euler–Lagrange equations:

```text
Rᵤ = W_L^u [
    -G p
    + Dᵀ W_L^τ τ
    + ρg
]

Rₚ = W_L^p Gᵀ W_L^u u
```

with stress determined from the weighted deformation operator:

```text
τ ≈ 2 M D u
```

The DYREL diagonal and pseudo-time update then act as a preconditioner for `Rᵤ`;
they must not redefine the underlying operator.

This separates the responsibilities:

- Larionov weights define the physical/discrete operator.
- DYREL defines iterative relaxation and diagonal preconditioning.
- `ϕ` supplies geometric weights and active-row elimination.

## 2. Add a dedicated 2D API

Add:

```julia
solve_VariationalDYREL!
```

and preserve standard `solve_DYREL!` unchanged.

The new function should accept a `RockRatio` explicitly:

```julia
solve_VariationalDYREL!(
    stokes,
    ρg,
    dyrel,
    flow_bcs,
    phase_ratios,
    ϕ,
    rheology,
    args,
    grid,
    dt,
    igg;
    kwargs...
)
```

Only define the implementation for `Geometry{2}`. Do not generalize the feature to
3D in this change.

## 3. Reuse the audited variational operators

Reuse the existing PR #520 implementations for:

- `compute_∇V!`;
- `compute_strain_rate!`;
- `compute_strain_rate_from_increment!`;
- `compute_variational_P!`;
- `update_stresses_center_vertex!`;
- `variational_face_mass`;
- active-cell and active-face predicates;
- free-surface stress boundary handling.

Retain the existing Powell–Hestenes and dynamic-relaxation structure, but replace
its standard-grid physics with these weighted operators.

## 4. Add variational DYREL momentum residuals

Implement variational 2D residual/update kernels using:

```text
pressure force      → G W_L^p p
stress divergence   → Dᵀ W_L^τ τ
velocity row        → W_L^u
body force          → weighted face force
```

The update should use:

```julia
variational_face_mass(ϕ.Vx)
variational_face_mass(ϕ.Vy)
```

with the same bounded small-fraction treatment already established by PR #520.

Inactive faces must receive:

```julia
velocity = 0
residual = 0
```

rather than an artificial air-material equation.

## 5. Make the stress/strain pair consistent

The stress kernel must consume the same weighted strain quantities used by the
variational solver.

In particular:

- center normal strain uses `ϕ.center`;
- vertex shear strain uses `ϕ.vertex`;
- stress divergence uses the corresponding `W_L^τ`;
- pressure residual uses `W_L^p`;
- plastic and viscoelastic corrections remain unchanged mathematically.

If the existing variational stress kernel cannot directly support DYREL's incremental
pressure correction, add a small 2D DYREL wrapper around its local constitutive
calculation instead of duplicating the rheology implementation.

## 6. Derive the variational DYREL diagonal

The current standard Gershgorin estimate cannot simply be reused.

Add a variational 2D estimator based on the diagonal of the weighted velocity
operator:

```text
Aᵤ ≈ W_L^u [
    Dᵀ W_L^τ M D
    + G W_L^p Gᵀ
]
```

The estimator must use:

- the same `ϕ` masks;
- the same center, vertex, and face weights;
- the same pressure-gradient and stress-divergence stencils;
- the same bounded face mass;
- the free-surface diagonal when enabled.

This is the main DYREL-specific numerical component.

## 7. Keep pressure and continuity weighted

The pressure update must use the variational pressure residual:

```julia
compute_variational_P!(...)
```

or a DYREL-specific fused equivalent.

Do not use the existing DYREL fused pressure residual unchanged, because it assumes
standard unweighted divergence and penalty fields.

Preserve the existing DYREL penalty mechanism:

```julia
P += γ_eff .* RP
```

with:

```text
γ_eff = variational penalty field
RP    = weighted variational pressure residual
```

## 8. Update constructors and initialization

Provide a variational initialization path for `DYREL` that computes:

- weighted bulk viscosity;
- weighted `γ_eff`;
- variational Gershgorin diagonals;
- variational `λmaxVx`, `λmaxVy`;
- initial `dτV`, `αV`, and `βV`.

The existing `DYREL!` overload taking `ϕ` should either be reused by the new solver or
corrected so that it also calls the variational Gershgorin estimator.

Relevant files:

- `src/DYREL/constructors.jl`;
- `src/DYREL/Gershgorin.jl`;
- `src/DYREL/solver.jl`.

## 9. Add operator-level tests

Before solver convergence tests, add small explicit matrix checks for the 2D weighted
operators.

For random fields and irregular masks, verify:

```text
<u, Dᵀ W_L^τ τ> = <D u, W_L^τ τ>
<p, Gᵀ W_L^u u> = <G p, W_L^u u>
```

Also test:

- zero-volume rows;
- positive sliver fractions;
- disconnected velocity faces;
- pressure-cell elimination;
- bounded face mass;
- free-surface rows.

Extend `test/test_variational_operators_2D.jl` where practical.

## 10. Add solver regressions

Create `test/test_variational_dyrel.jl` covering:

- full-volume `ϕ == 1`;
- partially filled/free-surface cells;
- rigid translation;
- rigid rotation;
- hydrostatic equilibrium;
- linear viscosity;
- plasticity;
- thermal/melt argument dispatch;
- finite residuals and no NaNs;
- convergence of both PH and DR loops.

The full-volume case should reproduce standard DYREL's converged solution, even if
iteration counts differ.

Add a small 2D comparison miniapp under:

```text
miniapps/DYREL2D/free_surface_stabilization/
```

using the existing standard-versus-variational comparison setup.

## 11. Backend and documentation work

Add 2D dispatch in:

- `src/JustRelax_CPU.jl`;
- `src/ext/CUDA/2D.jl`;
- `src/ext/AMDGPU/2D.jl`.

Update:

- `docs/src/man/DYREL.md`;
- `docs/src/man/variational_stokes.md`;
- `docs/src/man/listfunctions.md`.

Document that the implementation follows the Larionov weighted formulation and
currently supports only 2D.

## Acceptance criteria

The feature is ready when:

1. Weighted `D/G` operator adjoint tests pass.
2. Standard `solve_DYREL!` behavior remains unchanged.
3. `ϕ == 1` gives the same converged physical solution as standard DYREL.
4. Partial-volume cells remain finite and are correctly eliminated.
5. Rigid-body and hydrostatic invariants pass.
6. CPU tests pass, with CUDA/AMDGPU smoke tests added where hardware permits.
7. Runic and `git diff --check` are clean.

The central design principle is to reuse PR #520's variational discretization for the
operator, and reuse DYREL for the nonlinear constitutive iteration, Powell–Hestenes
update, damping, and diagonal relaxation.
