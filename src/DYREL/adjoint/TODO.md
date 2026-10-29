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

### 20. ✅ Pressure dependence of the viscosity was not differentiated
The viscosity pullback marked `args` as constant, but `args.P` is `stokes.P`: for a creep law
with an activation volume (V ≠ 0) the term (∂η/∂P)ᵀ η̄ never reached `adjoint.P`. The implicit
viscosity step of both solvers now passes `args.P` as an argument of its own and seeds
`adjoint.P` (`compute_viscosity_pressure_point!`, only when `args.P === stokes.P`). The
feedback F is unchanged: P is an independent input of η = V(τ(ε, P, η), P). Tests: a
dot-product check with `DislocationCreep` (V ≠ 0), and FD gradients of a sinking block whose
viscosity depends on P only, in both solvers; without the fix the gradient was off by 0.3 %.

## P2 — Consistency and robustness

6. ✅ **Non-convergence handling unified.** Both adjoint solvers now
   throw `"… adjoint did not converge (err=…, ϵ=…, …)"` instead of the standard solver
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
    Also the purely viscous shear-thinning shear band (n = 1/3, `viscosity_cutoff = (-Inf, Inf)`):
    `ShearBand2D_PowerLaw_DYREL.jl` stays at err = 5.09 for 50,000 iterations, while the standard
    DYREL solver converges on the same problem in 4,400.
18. ⬜ **Adjoint iteration diverges for strongly non-associated plasticity.** Drucker–Prager with
    ϕ = 20°, Ψ = 0°, η_vp = 1e-2 and ~85 % of cells yielded: the forward converges, the adjoint
    residual grows ×100 per 50 iterations in both solvers. Associated flow (Ψ = ϕ) or
    η_vp = 0.1 converges. The operator itself is the exact transpose (dot-product test), so this
    is the damped iteration on a non-symmetric, possibly indefinite consistent tangent, not a
    derivative bug. Options: a step-size estimate from the adjoint operator, or an error message
    that names the cause instead of NaN. Reproducer:
    `miniapps/Adjoint2D/NonAssociatedPlasticity2D_adjoint.jl` (diverges at Ψ = 0°, converges at
    Ψ = ϕ and at η_vp = 0.1).

19. ✅ **GeoParams' `PowerlawViscous` with n > 1 is shear-thickening.** GeoParams computes
    τII = ε0 η0 (εII/ε0)^n (its docstring says τⁿ = 2 η0 ε̇/ε̇0, the shear-thinning form), so a
    shear-thinning stress exponent 3 needs `n = 1/3`. The five power-law miniapps used `n = 3`
    while meaning shear thinning; they now use `n = 1 / 3` (`ShearBand2D_PowerLaw*.jl`, the two
    adjoint shear-band miniapps, `SinkingBlock2D_VE_PowerLaw_ViscositySensitivity.jl`). The
    tests keep n = 2 or 3: they check derivatives, which hold either way, and shear thinning is
    covered by `test_dyrel_viscosity_residual.jl` (n = 1/3) and the implicit-viscosity check
    in `test_adjoint_dot_product.jl` (`DislocationCreep`, n = 3). The docstring mismatch is
    for GeoParams upstream. Checked with the APT `ShearBand2D_PowerLaw.jl` at 64²: n = 3 gives
    an almost uniform strain rate, n = 1/3 X-shaped bands with ~25× strain-rate contrast. The
    APT, standard-DYREL and sinking-block miniapps converge with n = 1/3; the two variational
    shear-band miniapps (`ShearBand2D_PowerLaw_DYREL.jl`, `ShearBand2D_PowerLaw_ViscositySensitivity.jl`)
    do not (err stays at 5.09), see 17.

## P3 — Performance

11. 🟡 **Adjoint efficiency.** Profiled 2026-10-23 (CPU, 1 thread). One adjoint iteration at 128²
    costs 4.8 ms (viscoelastic), 6.7 ms (Drucker–Prager), 12.3 ms (cap); the stress pullback is
    54–83 % of it, since it recomputes the return mapping (cap Newton solve included) every
    iteration although the forward state is fixed.
    - ✅ **Adaptive damping.** The adjoint kept the forward's final damping, which does not fit the
      softer consistent tangent. Both solvers now re-estimate λmin every `nout` iterations
      (`AdjointDamping`, `update_adjoint_damping!`) on the adjoint's own `DYREL` state
      (`adjoint_dyrel!`, kept in `AdjointStokesArrays`), which carries the damping over to the
      next adjoint solve and leaves the forward state untouched.
      Drucker–Prager shear 64²: 64,200 → 6,700 adjoint iterations; linear sinking block
      1,800 → 1,400; variational Drucker–Prager 6,700 → 4,950. Gradient tests: 136 → 59 s and
      313 → 146 s. The dike example (50–100× forward) still to be re-measured.
    - ✅ **Implicit-viscosity derivatives computed once per solve.** Nonlinear viscosity cost one
      extra stress reverse sweep, for (∂τ/∂η)ᵀ τ̄, and two viscosity reverse sweeps per
      iteration. Now ∂τ/∂η, ∂η/∂τ, ∂η/∂P and F come from forward-mode sweeps once per solve
      and stress relaxation (`implicit_viscosity_scratch`, `enzyme_forward_pointwise!`); an
      iteration applies them pointwise and reverses only the stress kernel. The variational
      solver seeds η̇ = η, so the harmonic vertex mean H gives Ḣ = H and ∂τ/∂H follows by
      division; its transpose is a cheap reverse of `harm_clamped`. Matches the previous sweeps
      to 1e-10 (`test_adjoint_dot_product.jl`). One nonlinear sweep at 128²: 12.0 → 5.6 ms
      (power law), 17.7 → 8.3 ms (+ Drucker–Prager), variational 13.5 → 6.3 and
      16.8 → 9.3 ms, 1 thread; 1.5–2× with 4 threads.
    - ⬜ **Precompute the local stress tangent** once per adjoint solve and apply it with a
      gather-form stencil kernel, without Enzyme. Stress evaluation only (centers 5×5, vertices
      3×5; the nonlinear viscosity feedback folds into it); momentum, continuity and buoyancy
      pullbacks stay on Enzyme. Stages: standard solver with linear viscosity (measure here
      first), nonlinear viscosity, variational solver, tests. About 3–4 days. Bounded by the
      stress pullback's share of an iteration (54–83 %).
    - ✅ Skip the buoyancy pullback for a constant density (`ConstantDensityTrait`, as the
      forward `update_ρg!`): 0.57 ms → 0 per iteration at 128², 1 thread.
    - ⬜ Thread/GPU scaling: 4 threads give only 1.5× per iteration (colored launches); mostly
      solved by the stress-tangent item. The momentum and continuity pullbacks stay colored
      Enzyme launches.
    - ✅ Warm start from the previous step's λ; a separate adjoint tolerance (`adjoint_ϵ`,
      `adjoint_rel_drop`); merged per-iteration zeroing (`PERFORMANCE.md` 6–8).

## P4 — Cleanup

12. ⬜ Remove the `ReadOnlyShadow` workaround (`Duplicated` instead of `DuplicatedNoNeed`) once
    the Enzyme bug is fixed upstream (issue drafted; reproducer
    `test/mwe_enzyme_reverse_duplicatednoneed.jl`).
13. ⬜ Vertex→center gradient fold (`combine_center_vertex_gradient_kernel!`) maps boundary vertex 1
    to cells 1 and 2, while the forward `clamped_indices` reads cell 1 only. Sum preserved,
    attribution at the boundary differs. Align or fix the docstring.
14. ✅ **Forward stopped with an inconsistent viscosity.** The forward convergence test only
    checked the momentum and continuity residuals, which use the stress of the previous viscosity,
    so a solve could stop with η far from h(τ(η)) (a factor 2.6 seen from a homogeneous start,
    where the residuals vanish at once). Both forward solvers now also require the viscosity
    residual RMS(|h(τ) − η| / η) < ϵ for nonlinear viscosity (rock cells only in the variational
    solver); `test/test_dyrel_viscosity_residual.jl`. The adjoint linearizes about a consistent
    state.
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
