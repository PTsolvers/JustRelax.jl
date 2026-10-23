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
| 2 | **Compute the local stress tangent once per adjoint solve.** Since the forward state is fixed, store the small Jacobian of each point (outputs τ, τII, ΔPψ against ε, P, η) and apply it in a plain stencil kernel each iteration, without Enzyme. The nonlinear viscosity feedback and ∂ρg/∂P fold into it. | Estimated 2.5–6× per iteration; more for nonlinear viscosity | High: new code path, guarded by the dot-product tests | ⬜ |
| 3 | **Hand-written transposes for the linear momentum and continuity residuals**, instead of Enzyme with coloring. The momentum pullback costs 5× its forward kernel. | about 20–25 % of a viscoelastic iteration | Medium: guarded by the dot-product tests | ⬜ |
| 4 | **Skip the buoyancy pullback when density does not depend on pressure.** It runs every iteration even for `ConstantDensity`; a check for constant density already exists. | 0.4–0.6 ms per iteration, about 10 % (viscoelastic) | Trivial | ⬜ |
| 5 | **Better scaling with threads and on GPU.** Stencil kernels gather instead of scatter, so they need no coloring. | Mostly a side effect of 2 and 3 | — | ⬜ |
| 6 | **Start from the previous step's λ** when the adjoint runs every time step with the same objective. | Not measured; possibly large in time loops | Low–medium | ⬜ |
| 7 | **A separate, looser adjoint tolerance.** Iterations grow with log(1/ϵ), so 1e-8 → 1e-6 saves roughly 25 %. | about 25 %, at the cost of gradient accuracy | Trivial | ⬜ |
| 8 | **Small items:** merge the per-iteration zeroing into one kernel; avoid reallocating `constitutive_scratch` and temporary arrays in the norm checks. | about 5 % | Low | ⬜ |

## Still to measure

- The dike example (`JustRelaxAdjointExamples/VolcanoStation/DikeInjection2D_VariationalDYREL.jl`),
  where the adjoint needed 50–100× the forward iterations before option 1.
- Any of this on GPU.
