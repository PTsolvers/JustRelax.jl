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

### 2. ⬜ Plastic multiplier relaxation `λ_relaxation < 1` scales the plastic tangent
Drucker–Prager: `λ = (1 - r)·λold + r·F/(η + ηvp + …)` (`plastic_correction`,
`src/rheology/CapReturnMapping.jl`). The pullback holds `stokes.λ` constant, so the derivative is
`r ×` the converged tangent. Affects the adjoint operator (`λ_relaxation_PH/DR`) and the sensitivity
pass. **Fix:** pass `1` in the adjoint, as the viscosity path already does (`1.0 => nothing`); at
convergence the value is unchanged. Latent: defaults and all miniapps use 1. The cap return map
already returns the unrelaxed multiplier.

### 3. ⬜ Sensitivity pass can overwrite the forward viscosity
`compute_viscosity_parameter_gradients!` (`sensitivity_utils.jl`) passes `stokes.viscosity.η` / `ηv`
as the output of the differentiated viscosity function, which writes η recomputed from the rheology
(relaxation 1). Breaks `η_multiplier` / `update_material = false` runs and changes η for the next
step and for output. **Fix:** use a copy, like `η_target` in the adjoint loop.

### 4. ⬜ Race in the colored reverse on periodic grids with an odd cell count
The vertex-extent reverse (`ni + 1` points) wraps periodic vertices onto cells `{nx, 1}`. For odd
`nx`, `nx + 1` is even → two colors only; vertices 1 and `nx` both write cell `nx`, vertices 2 and
`nx + 1` both write cell 1, in the same launch (`color_groups`, `Enzyme_pointwise.jl`).
Non-deterministic on GPU / threads. **Fix:** a separate color for the periodic seam.

### 5. 🟡 End-to-end gradient test
`test/test_adjoint_gradients.jl` now checks density gradients against FD for both solvers and both
`:Vy` / `:P`. Still missing: viscosity, plasticity (DP and cap), `Kb`, nonlinear viscosity,
periodic BCs, masked (air) variational case. The FD/Taylor benchmarks in
`miniapps/Adjoint2D/benchmark` are not run by CI.

## P2 — Consistency and robustness

6. ⬜ **Non-convergence handling differs.** Standard solver warns and still computes sensitivities;
   variational throws. Pick one.
7. ⬜ **MPI not supported, but looks supported.** No `update_halo!` anywhere in the adjoint; the
   reverse scatters into neighbour cells across rank boundaries, which needs a reverse (summing)
   halo exchange that ImplicitGlobalGrid does not provide. Uses `norm_mpi` etc. Throw for
   `nprocs > 1` until supported. `igg.me == 0 && error(...)` hangs the other ranks.
8. ⬜ **3D not guarded.** The adjoint files are included only in the 2D modules; `adjoint = true`
   in 3D fails with `UndefVarError`. Throw an `ArgumentError`.
9. 🟡 **GPU.** CUDA error 717 fix (`detach(rheology)` for the inactive cap Newton solve, commit
   `c1c91c6b`) not yet verified on CUDA: rerun diagnostic cases B/E, the cap test in
   `test_dyrel_kernels.jl`, and the dike example. AMDGPU never run.
10. ⬜ **More than one Julia thread deadlocks the pointwise pullbacks.** Run with `-t 1` for now.

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
