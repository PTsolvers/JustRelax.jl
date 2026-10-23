# Adjoint DYREL solver: review findings and to-dos

Review of `src/DYREL/adjoint/` and the forward kernels it differentiates (2026-10-23, branch
`cs-adjoint2026`). Ordered by priority. Status: ✅ done · 🟡 partial · ⬜ open.

## P1 — Wrong results

### 1. ✅ Pressure objective (`field = :P`) gave wrong gradients
The inner loop dropped the pressure seed (`observation.field !== :P && seed_observation!`), and the
`λP` update reads that unseeded residual. Standard solver: `λV`, `λP` stayed zero → zero gradient,
warning only. Variational: seed reached `λP` once, then a different fixed point; `errV0 = eps()`
made the relative error explode ("Kaboom!").

**Fix:** seed in every iteration; `errV0` from the first nonzero momentum residual
(`momentum_residual_reference`, `relative_momentum_residual` in `adjoint_solver_utils.jl`); the
standard solver gained the variational guards (no continuity check while `λV = 0`, first pass of a
pressure objective never accepted). Verified against central FD (rel. err ≤ 6e-10, both solvers,
`:Vy` and `:P`) in `test/test_adjoint_gradients.jl`.

Note: a `:P` objective needs a unique pressure (finite `Kb` or an open boundary). On a closed,
incompressible domain the pressure is defined up to a constant and the adjoint has no solution.

### 2. ✅ Plastic multiplier relaxation `λ_relaxation < 1` scaled the plastic tangent
Drucker–Prager: `λ = (1 - r)·λold + r·F/(η + ηvp + …)` (`plastic_correction`,
`src/rheology/CapReturnMapping.jl`). The pullback holds `stokes.λ` constant, so the derivative was
taken of the relaxed update, in the adjoint operator and in the sensitivity pass (r = 0.5: adjoint
0.499 vs FD 2.407).

**Fix:** both adjoint solvers differentiate with `λ_relaxation = 1`, whatever the forward used; at
convergence λ equals its trial value, so the state is unchanged. Tested with r = 1 and 0.5 in
`test/test_adjoint_gradients.jl`.

### 3. ✅ Sensitivity pass could overwrite the forward viscosity (defensive)
`compute_viscosity_parameter_gradients!` passed `stokes.viscosity.η` / `ηv` as the output of the
differentiated viscosity function. Not reproduced on CPU (an `η_multiplier`-scaled viscosity
survives: Enzyme skips the `DuplicatedNoNeed` primal write), but Enzyme does not guarantee that.
Now passes copies; `test/test_adjoint_gradients.jl` checks the forward viscosity is kept.

### 4. ✅ Race in the colored reverse on periodic grids with an odd cell count
The vertex-extent stress reverse (`ni + 1` points) wraps periodic vertices onto cells through
`clamped_indices`: vertices 1 and `nx + 1` both read cells `nx` and `1`. For odd `nx` the extent is
even, so the parity colors put vertices 1 and `nx`, and 2 and `nx + 1`, in one launch; on GPU their
shadow updates raced.

**Fix:** `color_groups(n, periodic)` stops the alternating colors short of the last index in a
periodic direction and gives the remaining indices colors of their own. `enzyme_reverse_pointwise!`
and `differentiate_pointwise_material_kernel!` take a `periodic` keyword; the stress pullbacks of
the adjoint loop and of the material-gradient pass (both solvers) pass `periodic_dims(stokes)`.
Non-periodic coloring is unchanged. `test_dyrel_kernels.jl` checks that no color reuses a cell
(the old coloring fails for 3, 5, 7, 9 cells) and runs a periodic vertex-stencil pullback on the
selected backend. Not yet run on GPU.

### 5. ✅ End-to-end gradient tests
Two layers, all on the selected backend (CPU-verified only so far):

- **Operator layer**, `test/test_adjoint_dot_product.jl` (~40 s): ⟨J v, w⟩ = ⟨v, Jᵀ w⟩ with
  pseudo-random v, w for every pullback the adjoint uses — momentum, continuity/strain rate,
  stress (viscoelastic, Drucker–Prager, cap), pressure-dependent buoyancy, power-law viscosity,
  velocity BCs — and for the full `adjoint_residual!` / `variational_adjoint_residual!` sweep
  (sticky air, free-surface term), each with free-slip, no-slip and periodic-x BCs on odd and
  even grids. 106 checks.
- **Physics layer**, central FD of whole solves:
  - `test/test_adjoint_gradients.jl` (~2.5 min): density for `:Vy`/`:P`, both solvers; linear
    viscosity; forward viscosity kept; Drucker–Prager cohesion at λ_relaxation 0.5 (and 1 with
    `JULIA_JUSTRELAX_LONG_TESTS=1`).
  - `test/test_adjoint_gradients_physics.jl` (~5 min, mostly its five adjoint solves; see 11):
    one central difference per case along a weighted direction through all its parameters
    (each parameter on its own with `JULIA_JUSTRELAX_LONG_TESTS=1`). Power law + Drucker–Prager (η0, n, G,
    Kb, C, ϕ, Ψ, thermal α, weighted objective, nonlinear viscosity feedback); cap (C, pT);
    variational solver with sticky air, free-surface stabilization and periodic x (ρ, η of
    block and matrix, `:Vy` and `:P`); variational solver with power law or Drucker–Prager,
    no-slip bottom, background shear (η0, n, C, ϕ, ρ0, β, g).

Not covered: nonlinear viscosity next to sticky air (the forward diverges, see 17), strongly
non-associated plasticity (the adjoint diverges, see 18), MPI, GPU runs of any of the above.

## P2 — Consistency and robustness

6. ✅ **Non-convergence handling unified.** Both adjoint solvers now reset the shared DYREL
   state and throw `"… adjoint did not converge (err=…, ϵ=…, …)"` instead of the standard solver
   returning unconverged sensitivities with a warning. Tested in `test_adjoint_gradients.jl`.
7. ✅ **MPI guarded.** `check_adjoint_support` (`src/DYREL/solver.jl`) throws an `ArgumentError`
   for `adjoint = true` with `nprocs > 1`: the reverse pass adds into neighbouring cells, which
   would need a summing halo exchange. The NaN/"Kaboom" errors now fire on every rank (their
   `err` is a global reduction), not only on rank 0. Real MPI support remains open.
8. ✅ **3D guarded.** The same check throws an `ArgumentError` for `adjoint = true` in 3D before
   reaching the 2D-only adjoint code. Both guards are unit-tested in `test_dyrel_kernels.jl`.
9. 🟡 **GPU.** CUDA error 717 fix (`detach(rheology)` for the inactive cap Newton solve, commit
   `c1c91c6b`) not yet verified on CUDA: rerun diagnostic cases B/E and the dike example, and
   run `test_dyrel_kernels`, `test_adjoint_dot_product`, `test_adjoint_gradients` and
   `test_adjoint_gradients_physics` with `JULIA_JUSTRELAX_BACKEND=CUDA`. AMDGPU never run.
10. ✅ **Multi-threaded pullbacks deadlocked.** Cause: `autodiff_deferred` compiles its gradient
    on the first call, which happened inside the threaded kernel loop; threads compiling the
    same gradient at once deadlocked. `compile_pullback_serially!` (`Enzyme_pointwise.jl`) now
    compiles each pullback signature once on the main thread, on copies, before its first
    launch (CPU Threads/Polyester backends only). Verified with 2 and 4 threads (dot-product
    test, miniapp); `test_adjoint_dot_product.jl` runs a two-thread pullback in a subprocess.

17. ⬜ **Variational forward diverges with a power-law viscosity in two setups** (found while
    writing the tests). Purely buoyancy-driven flow from rest: the residual reaches ~1e41 within
    the first 50 iterations, before the first step-size update, whether the initial η is 1, 10
    or 100. Sticky air plus a power law: NaN or "Kaboom" with or without the free-surface term,
    also with background shear. A forward-solver issue; the tests avoid both setups.
18. ⬜ **Adjoint iteration diverges for strongly non-associated plasticity.** Drucker–Prager with
    ϕ = 20°, Ψ = 0°, η_vp = 1e-2 and ~85 % of cells yielded: the forward converges, the adjoint
    residual grows ×100 per 50 iterations in both solvers. Associated flow (Ψ = ϕ) or
    η_vp = 0.1 converges. The operator itself is the exact transpose (dot-product test), so this
    is the damped iteration on a non-symmetric, possibly indefinite consistent tangent, not a
    derivative bug. Options: a step-size estimate from the adjoint operator, or an error message
    that names the cause instead of NaN. Reproducer:
    `miniapps/Adjoint2D/NonAssociatedPlasticity2D_adjoint.jl` (diverges at Ψ = 0°, converges at
    Ψ = ϕ and at η_vp = 0.1).

## P3 — Performance

11. ⬜ **Adjoint needs 50–100× the forward iterations** (128²: forward ~300–500 it / 5 s, adjoint
    23,700–47,600 it / 265–520 s), each ~3.5× a forward iteration. Candidates: step sizes fixed at
    the start (forward re-estimates every `nout` via Gershgorin); relative-to-first-residual
    criterion stricter than the forward's scale-free one; `constitutive_scratch` (~12 arrays)
    reallocated on every call.

## P4 — Cleanup

12. ⬜ Remove the `ReadOnlyShadow` workaround (`Duplicated` instead of `DuplicatedNoNeed`) once
    the Enzyme bug is fixed upstream (issue drafted; reproducer
    `test/mwe_enzyme_reverse_duplicatednoneed.jl`).
13. ⬜ Vertex→center gradient fold (`combine_center_vertex_gradient_kernel!`) maps boundary vertex 1
    to cells 1 and 2, while the forward `clamped_indices` reads cell 1 only. Sum preserved,
    attribution at the boundary differs. Align or fix the docstring.
14. ⬜ With `viscosity_relaxation = 1e-2`, the forward stops on residuals, so η may not satisfy
    η = viscosity(τ) exactly at the linearization point. Consider one unrelaxed viscosity update
    before the adjoint.
15. ⬜ `test/test_dyrel.jl` uses an undefined `JR3` (pre-existing).
16. ⬜ Typos "caclulate feedbck" in `Enzyme_kernels.jl` and `Enzyme_kernels_VS.jl`; the `typos`
    CI check will likely flag them.

## Checked and correct

- Implicit viscosity correction (division by `1 - feedback`) is exact: each location's viscosity
  feeds back only through its own stress, in both solvers (variational vertex stresses read center
  η but do not feed back into η).
- Seed sign conventions (`-λV`, `-λP` in the sensitivity pass) are consistent.
- Per-area scaling uses the reciprocal cell widths `_di.vertex`.
- Boundary pullbacks run in the reverse order of `_flow_bcs!`.
