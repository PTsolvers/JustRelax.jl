# DYREL adjoint solver: status and open points

Written on 2026-11-11, after the investigation of the slow adjoint convergence in the
Dora Maira example (`JustRelaxAdjointExamples/SubductionShearZone/DoraMairaVerticalWeakZone.jl`).
Branch `cs-adjoint2026`, commit `269daeb8`.

## Done

1. **BiCGStab inner solver.** Opt-in through `adjoint_kwargs = (; inner_solver = :bicgstab)`,
   in both 2D adjoint solvers (standard and variational). Committed and pushed as `269daeb8`.
   The default is still dynamic relaxation (`inner_solver = :dr`).
2. **Tests.** `test/test_adjoint_gradients.jl` checks that BiCGStab gives the same gradient as
   the default solver for velocity and pressure objectives, and checks the plasticity gradient
   against finite differences. 249 of 249 pass on CPU.
3. **Docs.** The option is described in the docstring in `src/DYREL/solver.jl` and in
   `docs/src/man/DYREL.md`.
4. **Dora Maira example.** It defaults to BiCGStab locally (penalty factor 20) and on the
   cluster (penalty factor 80). Both edits are uncommitted.

## Bugs and weak spots to fix

5. **The adjoint runs after an unconverged forward solve.** Both solvers call it anyway
   (`src/DYREL/solver_VS.jl:365-371`, `src/DYREL/solver.jl:479-483`). The gradients are then
   wrong, with at most a warning. This is the most important item. Fix: throw an error before
   the adjoint call if the forward solve did not converge.
6. **BiCGStab breakdowns are silent.** `adjoint_bicgstab!` stops the inner solve early and
   restarts in the next outer pass, without a message. Fix: one `@warn ... maxlog = 5` at the
   two `break` lines.
7. **The damping estimate can mix in a pressure update.** This happens when the relaxation
   loop ends on its iteration limit right after storing a step
   (`update_adjoint_damping!` in `adjoint_solver_utils.jl`). Fix: set `damping.stored[] = 0`
   after the pressure update, one line in each solver.
8. **Inner loops run one iteration over the limit.** `itPT ≤ iterMax_DR` gives 50,001
   iterations for a limit of 50,000. Fix: `itPT < iterMax_DR`.
9. **`test/test_dyrel.jl:31` calls an undefined module `JR3`** in the 3D adjoint allocator
   test, so that test set errors.

## Clean-up

10. **The two adjoint solvers are near-copies.** The outer loop of about 250 lines exists in
    both `solver_adjoint.jl` and `solver_adjoint_VS.jl`, and every change must be made twice.
11. **The Schur correction exists in four kernels.** Two are in `update_adjoint_V_damping_DR!`
    and two in `adjoint_schur_correction!`, which was added for BiCGStab.

## Not yet verified

12. **The production BiCGStab code on a GPU.** Only the prototype ran on CUDA. The pushed code
    has run on CPU only.
13. **A full Dora Maira time-stepping run with BiCGStab.** All results are from one frozen
    state at step 10.
14. **A non-symmetric case larger than 16×16.** The 32×32 non-associated plasticity forward
    solve did not converge in 200,000 iterations, so there was no valid state to test.
15. **`test_adjoint_gradients_physics.jl`, the full test suite and the docs build** after the
    change.
16. **Runic formatting** of the changed files. It was not run.

## What we learned

17. **Why dynamic relaxation is slow here.** On the Dora Maira state (64×128, step 10) the
    relaxation diverges for a damping below about 0.12 and crawls above it, so no fixed
    damping makes it fast. The likely cause is that the slightly non-symmetric tangent has
    complex eigenvalues. That is inferred from the growth rates, not measured directly.
18. **Conjugate gradients do not work**, even on the nearly symmetric power-law case.
19. **Measured gain.** On the 128×256 checkpoint (step 10, penalty factor 80, GPU), the
    BiCGStab prototype needed 39,798 sweeps and 136 s. The best relaxation run needed
    617,617 iterations and 1,441 s.
20. **The penalty factor barely matters with BiCGStab.** On the 64×128 state:

    | Penalty factor | Outer passes | Total sweeps |
    |---|---|---|
    | 20 | 14 | 13,010 |
    | 80 | 6 | 13,830 |
    | 320 | 3 | 14,384 |
    | 1280 | 3 | 25,230 |

21. **Non-associated plasticity (16×16, tolerance 1e-8).** BiCGStab converged in 1,566 sweeps
    without breakdown; dynamic relaxation needed 2,450. Gradients agree to 2e-8.
22. **What limits it now.** The remaining cost is the velocity solve with only a diagonal
    preconditioner. A further large gain would need a better preconditioner, such as
    multigrid.

## Open decision

23. **The cluster's `experiments/` folder**
    (`/home/cschuler/work2/JustRelaxAdjointExamples/experiments/`). It still holds the
    `step10.jls` checkpoint and the baseline records of the relaxation runs. Only
    `experiment.jl` and `bicgstab.log` were removed.

## Experiment files

The scripts, logs and saved states behind these numbers (`scratch/dora_adjoint/` and
`scratch/adjoint_solver_experiments/`) were deleted on 2026-11-11. Only the write-ups remain.
