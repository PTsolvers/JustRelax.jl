# Adjoint DYREL solver: speed-up options

Profiled 2026-10-23 on branch `cs-adjoint2026` (CPU, macOS arm64). Ordered by priority. Status:
✅ done · ⬜ open. See also [`TODO.md`](TODO.md), item 11.

## Where the time goes

The adjoint's cost is the number of iterations times the cost per iteration.

**Iterations.** Measured at 64² with one thread:

| Case | Forward iterations | Adjoint iterations, fixed damping | Adjoint iterations, adaptive damping |
|---|---|---|---|
| linear sinking block | 2,900 | 1,800 | 1,400 |
| Drucker–Prager shear | 41,300 | 64,200 | 6,700 |
| variational, associated Drucker–Prager | — | 6,700 | 4,950 |

With fixed damping, the Drucker–Prager adjoint needed about 14,000 inner iterations per outer
pass, against 100–300 for the forward.

**Cost per iteration.** One adjoint iteration at 128², one thread:

| Part | Viscoelastic | Drucker–Prager | Cap |
|---|---|---|---|
| initialize (zeroing) | 0.32 ms | 0.23 ms | 0.23 ms |
| momentum pullback | 0.73 ms | 0.58 ms | 0.61 ms |
| buoyancy pullback ρg(P) | 0.44 ms | 0.56 ms | 0.53 ms |
| **stress pullback** | **2.59 ms** | **4.60 ms** | **10.16 ms** |
| continuity pullback | 0.55 ms | 0.64 ms | 0.59 ms |
| velocity BC pullback | 0.01 ms | 0.01 ms | 0.01 ms |
| damped update | 0.13 ms | 0.13 ms | 0.13 ms |
| **total** | **4.77 ms** | **6.74 ms** | **12.25 ms** |
| *forward stress kernel, for reference* | *0.75 ms* | *1.64 ms* | *11.41 ms* |
| *forward momentum kernel, for reference* | *0.14 ms* | *0.14 ms* | *0.14 ms* |

The stress pullback is 54–83 % of an iteration. It recomputes the local return mapping, the cap
Newton solve included, on every iteration, although the forward state is fixed during the
adjoint solve. Nonlinear viscosity is not in this table: it currently needs three stress
pullbacks and two viscosity pullbacks per iteration.

With 4 threads, an iteration gets only about 1.5× faster (viscoelastic 4.8 → 2.9 ms,
Drucker–Prager 6.7 → 4.4 ms), while the forward stress kernel gets 2.9× faster. Each pullback
is split into 4–9 colored launches, and every launch pays the threading overhead.

## Options

| # | Change | Gain | Effort | Status |
|---|---|---|---|---|
| 1 | **Re-estimate the adjoint damping every `nout` iterations**, as the forward does. The adjoint kept the forward's final damping, which does not fit the softer consistent tangent it iterates on. | **Measured:** Drucker–Prager 64,200 → 6,700 iterations (9.6×); linear block 1,800 → 1,400; variational Drucker–Prager 6,700 → 4,950. Gradient tests 136 → 59 s and 313 → 146 s. | Low | ✅ `b481571c` |
| 2 | **Compute the local stress tangent once per adjoint solve.** Since the forward state is fixed, store the small Jacobian of each point (outputs τ, τII, ΔPψ against ε, P, η) and apply it in a plain stencil kernel each iteration, without Enzyme. Stress evaluation only; the nonlinear viscosity feedback folds into it, the momentum, continuity and buoyancy pullbacks stay on Enzyme. | At most the stress pullback's share of an iteration (54–83 %); measure after the linear-viscosity stage | High (3–4 days): new code path, guarded by the dot-product tests | ⬜ |
| 4 | **Skip the buoyancy pullback when density does not depend on pressure.** It ran every iteration even for `ConstantDensity`. It now returns at once for `ConstantDensityTrait`, the check the forward `update_ρg!` uses, so it is exact. Densities with the trait `NonConstantDensityTrait` (e.g. `PT_Density`, also with β = 0) still run it. | **Measured:** 0.57 ms → 0 per iteration at 128², 1 thread (about 10 % of a viscoelastic iteration) | Trivial | ✅ |
| 5 | **Better scaling with threads and on GPU.** Stencil kernels gather instead of scatter, so they need no coloring. | Mostly a side effect of 2 (and 4, which removes the buoyancy launches for a constant density); the momentum and continuity pullbacks stay colored Enzyme launches | — | ⬜ |
| 6 | **Start from the previous step's λ.** λ was already kept between solves, but the momentum residual was measured against its first value, so a good start only tightened the criterion. The reference is now the residual at λ = 0 (the objective's velocity seed); a non-finite λ from a failed solve is reset to zero; the variational solver zeroes λ in rows and cells eliminated since the previous solve. | **Measured, steps 2–3 of a time loop at 64²:** linear sinking block 1,900 / 1,600 → 0 / 0 iterations (the previous λ already meets ϵ); Drucker–Prager shear 7,000 / 6,800 → 4,900 / 5,500 | Low | ✅ |
| 7 | **A separate, looser adjoint tolerance.** Iterations grow with log(1/ϵ), so 1e-8 → 1e-6 saves roughly 25 %. | about 25 %, at the cost of gradient accuracy | Trivial | ✅ `adjoint_ϵ`, `adjoint_rel_drop` (the adjoint has its own `DYREL` state). Since the forward residual is measured against the driving force, the same ϵ means the same accuracy for both solves. |
| 8 | **Small items.** One reverse sweep already zeroes every stress, strain-rate and residual seed it consumes (Enzyme zeroes propagated shadows; tested for both solvers), so each iteration now only resets P and V and copies λ into the residual seeds (`reset_adjoint_residuals!`); the full clear runs once per solve. Left as is: the viscosity seeds of the linear loop (skipping them measured no gain), `constitutive_scratch` (allocated once per solve), the norm temporaries (every `nout` iterations), and the ~85 KB per sweep allocated by ParallelStencil's threaded launches (fewer launches: items 2 and 4). | **Measured:** reset 0.25 → 0.05 ms at 128², about 4 % of a viscoelastic iteration | Low | ✅ |
| 9 | **Compute the implicit-viscosity derivatives once per solve.** Nonlinear viscosity cost one extra stress reverse sweep and two viscosity reverse sweeps per iteration, only for (∂τ/∂η)ᵀ τ̄ and (∂η/∂τ)ᵀ η̄. These derivatives, ∂η/∂P and F are fixed during an adjoint solve: forward-mode sweeps compute them once (`implicit_viscosity_scratch`), and an iteration applies η̄ = (s·τ̄)/(1 − F), τ̄ += v η̄, P̄ += v_P η̄ pointwise. | **Measured,** one nonlinear sweep at 128², 1 thread: power law 12.0 → 5.6 ms, + Drucker–Prager 17.7 → 8.3 ms; variational 13.5 → 6.3 and 16.8 → 9.3 ms. 4 threads: 1.5–2× | Low–medium | ✅ |

## Still to measure

- The dike example (`JustRelaxAdjointExamples/VolcanoStation/DikeInjection2D_VariationalDYREL.jl`),
  where the adjoint needed 50–100× the forward iterations before option 1.
- Any of this on GPU.
