# Popov tensile-cap stress integration: diagnosis and fix plan

## Decision

Add a safeguarded **local constitutive Newton solve** for `DruckerPragerCap`, shared by the PT and DYREL stress updates. Retain the existing global PT/DYREL solvers initially. Newton is not the only possible nonlinear algorithm, but a converged coupled return map is missing, and damped Newton follows the paper directly. A global Newton solver is a separate performance project, not an established prerequisite for this fix.

Reference: `popov2025.pdf`, Popov, Berlie and Kaus (2025), *A dilatant visco-elasto-viscoplasticity model with globally continuous tensile cap: stable two-field mixed formulation*, [publisher](https://gmd.copernicus.org/articles/18/7035/2025/). Relevant material: Sections 3.3–3.6, Eqs. (31)–(34), (37)–(51), and Appendix A. The paper separates the local three-variable Newton iteration from its global finite-element Newton iteration. Its local algorithm also applies to velocity-based finite differences.

This is a plan only. No solver or dependency implementation was changed.

## Evidence in the current implementation

1. **DYREL uses one trial-state correction.** `src/DYREL/stress_kernels.jl:739` forms the Maxwell trial stress, evaluates yield and potential gradients there, then computes `lambda_new = F_trial / (eta_ve + eta_reg + K*dt*F_P*Q_P)`. Stress and pressure are corrected once, without reevaluating the coupled constitutive residual. Repeated calls at fixed inputs with multiplier relaxation equal to one reproduce the same correction. On the curved cap, the gradients depend on the corrected pressure and stress; the DP denominator is not the general cap Jacobian.
2. **PT and variational PT repeat the same approximation.** See `src/stokes/StressKernels.jl:1070` and `:1116`, and `src/variational_stokes/StressKernels.jl:62`. The PT shear coefficient is `eta*d_tau_r`. These paths also lack a local residual convergence check. The legacy `src/rheology/StressUpdate.jl:2` path hard-codes the DP yield envelope; cap materials must not silently enter it.
3. **Pure tensile yielding is suppressed.** DYREL returns immediately for zero effective deviatoric strain invariant (`src/DYREL/stress_kernels.jl:754`), before checking pressure-dependent yielding. PT conditions plasticity on nonzero trial deviatoric invariant (`src/stokes/StressKernels.jl:1081`, `:1127`). A cap can yield at zero deviatoric stress and produce purely volumetric plastic flow.
4. **The installed GeoParams gradient wrapper discards state.** JustRelax forwards pressure to `GeoParams.∂Q∂τ` in `src/rheology/StressUpdate.jl:474`, but the installed tuple/SVector wrappers in `GeoParams/src/Plasticity/Plasticity.jl:83` accept and then drop those keywords. Deviatoric and volumetric flow gradients can consequently describe different pressure states. JustRelax already halves engineering shear gradients; preserve that conversion until the dependency contract is explicitly changed.
5. **Pressure handling is already structured around trial/physical pressure.** The multiphase PT solver passes `theta` as trial pressure and `stokes.P` as corrected pressure. DYREL carries `DeltaPpsi` separately in momentum and absorbs it into `P` at completion (`src/DYREL/solver.jl:348`). Preserve these distinctions when inserting the local solve.
6. **Existing cap tests do not establish the return-map equations.** `test/test_shearband2D_DPCap.jl` checks equilibrium norms, finite stresses, plastic activity and dilation. Those are useful integration checks but do not directly check the active Perzyna residual or pure hydrostatic tensile opening.

### Direct Julia checks

Ran Julia 1.13.0 with `--project=. --startup-file=no`, loading the actual JustRelax code and GeoParams 0.7.20 from `C:/Users/Albert/.julia/packages/GeoParams/aIEP1`. Called `JustRelax.JustRelax2D._compute_local_stress` with:

- `C=1`, `phi=30 degrees`, `Psi=0`, `eta_vp=0.1`, `pT=-0.5`;
- `eta=1`, `G=1`, `K=4`, `dt=1`, zero old stress and accumulated strain;
- initial multiplier zero and multiplier relaxation one.

| Effective strain tensor `(xx, yy, xy)` | Trial pressure | Returned stress invariant | Multiplier rate | Pressure correction | `F(tauII, P_trial + DeltaP) - eta_vp*lambda` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `(0, 0, 0)` | -1.0 | 0 | 0 | 0 | 0.5590169944 |
| `(0, 0, 1)` | -0.8 | 0.8764304579 | 0.2474601524 | 0.6415018059 | 0.0843120790 |

The first case proves that an overstressed hydrostatic tensile state is skipped. The second proves that the returned active state does not satisfy even the regularized yield equation. It does not isolate the numerical contribution of the gradient-wrapper defect from that of the approximate corrector.

For the same material, tensor `(0,0,1)` and pressure `-0.8`, the installed wrapper returns shear gradient `0.9987025460`; the direct component function, supplied the same pressure, returns `0.7615671888`. This independently reproduces the state-forwarding defect.

These were focused constitutive calls, not a complete simulation or CPU/GPU/MPI suite. They establish local correctness defects, not the cause of every possible global convergence failure.

## Implementation sequence

### 1. Establish material and derivative contracts

- Turn the two direct cases above into regressions and add an independent reference solve for mixed cap loading.
- Fix state forwarding upstream in GeoParams and require a verified version, or use a narrowly scoped JustRelax cap adapter until that release is available. Do not edit a user's installed package cache.
- Define a cap helper returning geometric `F`, `A_tau = (dQ/dtauII)/2`, `A_p = -dQ/dP`, their first derivatives, and yield derivatives at the same state. Use separate branch predicates for the yield surface and flow potential.
- The current GeoParams function named `∂Q∂τII` returns `A_tau`, not the full scalar derivative. Do not introduce a factor-of-two error by interpreting its name literally. Verify derivatives by scalar finite differences, including both branches and their transitions.
- Keep geometric yielding and `F - eta_vp*lambda` distinct. JustRelax already uses cap regularization in its denominator; this is not a claim that JustRelax ignores `eta_vp`. Ensure an upstream regularization change cannot cause double subtraction, including for ordinary DP.
- Preserve physical tensor shear conventions and verify the 2D invariant/out-of-plane convention before reconstructing tensors.

### 2. Implement the local coupled solve

Create an internal allocation-free routine, e.g. `src/rheology/CapReturnMapping.jl`, included through shared code. Inputs include trial stress, trial pressure, shear compliance, `K*dt`, frozen material/history state and local tolerances. Return corrected stress/pressure, multiplier rate, plastic rates, residual norm, iteration count and success status.

For linear viscosity at fixed outer-iteration state, solve for `(s, p, lambda)` with `s = tauII >= 0` and `lambda >= 0`:

```text
R_s = (s_trial - s)/(2*eta_star) - lambda*A_tau(s,p)
R_p = (p - p_trial)/(K*dt)      - lambda*A_p(s,p)
R_f = F(s,p)                   - eta_vp*lambda
```

Here `lambda` is a rate, not the integrated plastic increment. For DYREL, `eta_star = eta/(1 + eta/(G*dt))`. For PT, derive the equivalent residual using its pseudo-transient trial and `eta_star = eta*d_tau_r`; verify that the converged outer fixed point satisfies the physical constitutive equations.

- Accept an admissible elastic trial with zero multiplier, zero plastic rates and zero pressure correction. Clear stale multiplier state on unloading.
- Otherwise start from the trial state and solve the coupled 3-by-3 system with the full Jacobian, including `lambda*dA_tau/ds`, `lambda*dA_tau/dp`, `lambda*dA_p/ds` and `lambda*dA_p/dp`.
- Scale stress and strain-rate residuals before convergence and line-search comparisons. Use absolute plus relative tolerances, a bounded iteration count, finite-value checks, safeguarded Newton steps and backtracking. A small step alone does not establish convergence.
- Reevaluate the yield and potential branches at each candidate state. Use a fixed-size linear solve rather than forming an explicit inverse.
- Treat `s_trial=0` explicitly: retain zero deviatoric stress but solve pressure and multiplier when the cap yields. Never divide by the zero invariant.
- Compute all reported rates and pressure corrections from the converged state. Any outer relaxation must not be confused with convergence of the inner return map.
- Reject or explicitly handle unsupported active volumetric flow with infinite bulk modulus; suppressing `K*dt` in one equation while allowing it to become infinite in another is not a valid limit.

The first implementation may keep nonlinear viscosity lagged in the existing outer loop, provided outer convergence checks include rheological consistency. Full reproduction of the paper's nonlinear-creep algorithm additionally requires the scalar nonlinear predictor, creep term and derivative in the coupled corrector. State this boundary explicitly rather than claiming frozen viscosity solves all of Eq. (42).

### 3. Integrate with existing solver families

- Start with DYREL's shared `_compute_local_stress`, then connect PT and variational PT, including 2D/3D and strain-increment variants. Route or explicitly reject cap use through legacy DP-only stress paths.
- Retain the analytical DP fast path and existing public solver signatures. Put local cap tolerances/iteration options through existing keyword handling only as needed.
- Use corrected physical pressure in momentum. Use trial pressure in the trial-pressure continuity equation; do not also add the plastic volume source there. Commit corrected physical pressure as the next physical timestep's pressure history.
- Preserve current phase treatment initially: DYREL solves phases then blends outputs, while PT blends material/yield quantities. Specify which residual is checked under each convention; a phase average is not automatically the response of one homogeneous cap. Test pure-phase limits and zero-weight phases. Cross-solver equality should first be demonstrated for one phase, not assumed for mixtures.
- Keep accumulated deviatoric and volumetric plastic strain fixed during local/global iterations. Accumulate converged rates once per accepted physical timestep, consistent with the paper's explicit softening treatment. Audit early-exit/failure paths before committing history.
- Surface local failure to the host and reduce failure flags/residual maxima across MPI ranks. Do not report global success with unconverged constitutive states. Preserve backend-resident allocations and required halos for any new stencil-read fields.

### 4. Validate in increasing scope

1. Material-point cases: elastic loading, analytical DP limit, cap shear/tension, hydrostatic tension with `Psi=0`, nearly zero invariant, yield/flow transition crossings, unloading after yield, varying `eta_vp` and timestep, finite large bulk modulus, nonzero history, and rotated equivalent stresses in 2D/3D. Check all three residuals, rates and pressure signs, not only finiteness.
2. Verify the complete Jacobian against an independent numerical derivative away from branch boundaries; use branch-aware checks at transitions. Force poor initial guesses and failure limits to exercise the line search and failure reporting.
3. Homogeneous grid tests for hydrostatic extension and mixed loading in PT, DYREL and their variational forms. Check equilibrium, local constitutive residuals and exactly-once history accumulation. Require insensitivity to numerical relaxation once converged.
4. Retain existing DP/shear-band tests; extend the DPCap miniapp to a demonstrably active tensile regime. Use a small paper-inspired tensile benchmark and timestep/grid studies before claiming robust localization behavior.
5. Run focused tests, full CPU suite when feasible, Runic on changed Julia files, documentation checks, then available CUDA/AMDGPU and two-rank MPI checks. Report unavailable hardware checks explicitly.

## Implementation status (2026-09-21)

Steps 1-3 are implemented and Step 4 is partially done.

- `src/rheology/CapReturnMapping.jl` holds the local coupled solve: scaled residuals, a
  ForwardDiff Jacobian on `SVector`s, backtracking with `s >= 0` and `lambda >= 0`
  safeguards, and a failure flag that is turned into a nonfinite multiplier so the host
  residual check aborts instead of accepting the state.
- `plastic_correction` is the single entry point for PT, variational PT and DYREL. Non-cap
  materials keep the analytical relaxed Drucker-Prager correction. The legacy DP-only
  `_compute_τ_nonlinear!` path is routed, not entered, by cap materials.
- The DYREL zero-strain early return no longer suppresses pure tensile yielding.
- Fixed while integrating: the four strain-increment call sites passed `eta*dtau_r` where the
  kernels use `eta*dtau_r*dt`, which changed the Drucker-Prager denominator in the
  displacement formulation. Runic was re-applied to the three kernel files.

Validation that has been run:

- `test/test_cap_return_mapping.jl`: the two diagnostic cases, phase-dispatch variants, the
  AD Jacobian against central differences, failure handling, admissible elastic trial and
  stale-multiplier clearing, Drucker-Prager-limit equivalence while yielding, sweeps over
  `eta_vp`, `dt`, `K` and history with complementarity-aware residual checks, and invariance
  under rotation of the strain tensor in 2D and 3D.
- `test/test_cap_tensile_grid2D.jl`: homogeneous volumetric extension in PT. The cap stops the
  pressure just above `pT`, opens the box volumetrically and accumulates `EVol_pl`, while the
  cap-free control run decompresses far past `pT`.
- Full CPU suite including the two-rank MPI tests: pass. CUDA: `test_shearband2D_DPCap`,
  `test_shearband2D_DPCap_DYREL` and `test_cap_tensile_grid2D` pass, so the Newton solve runs
  inside GPU kernels. AMDGPU has not been checked.
- Documentation: the cap description in `docs/src/man/material_physics.md` is its own section
  rather than text wedged under "Multiple phases per cell", and its `@ref` target exists. The
  full Documenter build was not run; the docs environment is not instantiated here and the
  change is prose only.

Open items, in the order they should be picked up:

1. Settled by measurement, not changed: the cap branch returns the converged multiplier without
   the outer relaxation `relλ`. Relaxing it the way the analytical branch does was tried on the
   DPCap shear band (32², 10 steps, `nout = 25`). The converged fields are the same to six
   digits, but the run costs 9650 PT iterations at `relλ = 0.2` and 14725 at `relλ = 0.05`
   against 8425 unrelaxed. Leave it unrelaxed; the comment in `plastic_correction` records this.
2. Done: each PT solver copies its local `λ` (and the vertex multipliers) onto `stokes.λ`,
   `stokes.λv`, `stokes.λv_yz/xz/xy` before the end-of-step bookkeeping, so plastic activity is
   inspectable after a solve. Local iteration counts are still not reported.
3. Done: `reject_incompressible_cap` runs at every Stokes, variational Stokes and DYREL entry
   point and names the offending phase instead of letting the run stop on a bare `NaN(s)`.
   `miniapps/DYREL2D/thermal_stress` selects `ν = 0.5` with `DruckerPragerCap` when
   `is_compressible` is false, and now fails there with an explanation.
4. Done: the single-phase `compute_τ_nonlinear!` now takes `ε_vol_pl` and writes it in both
   branches (`-λ dQ/dP` for the cap, `λ sinψ` for the cone), so a single-phase cap run
   accumulates `EVol_pl` instead of integrating zeros. `test_cap_tensile_grid2D` exercises the
   single-phase and multiphase stress paths of the same model. The multiphase
   `compute_τ_nonlinear!` is reached only from `update_stress!`, which nothing calls and whose
   argument list no longer matches the kernel; it was left alone.
5. Fixed: the variational strain-increment center kernel scaled its stress correction without
   the `dt` its multiplier denominator carries, unlike its own vertex kernel and the
   non-variational counterpart. This predates the cap work and changes results for
   displacement-formulation variational runs with plasticity.
6. Step 4.4 is half done. `miniapps/benchmarks/stokes2D/tensile_localization/TensileLocalization2D.jl`
   puts the cap in a demonstrably active tensile regime under restrained uniaxial extension and
   runs the regularization, resolution and timestep study: every run converges, the pressure is
   held at `pT` with an overstress that grows with `eta_vp` (just short of `pT` at `1e-3`,
   `+0.08` at `1e-1`), and `Pmin`, `tauII` and `EVol` move by a few percent between `32^2` and
   `64^2` and between `dt` and `dt/2` at equal end time. What it does **not** produce is
   localized mode-I zones: the boundary conditions impose the same volumetric strain rate in
   every cell, so the opening is domain-wide whatever the mesh, the regularization or the random
   plastic-strain seeding. Localized tensile zones need a free surface, gravity and a
   depth-dependent strength, which is the configuration of the benchmark below; treat that as
   the test of localization, and this miniapp as the tensile smoke test.
7. **Running a miniapp on this machine does not test this checkout.** `miniapps/Project.toml`
   pins `JustRelax` through `[sources]` to `/home/albert/Documents/DevPkg/JustRelax.jl`, a path
   that does not exist here, so `--project=miniapps` silently resolves the registered v0.8.0
   from the depot instead. Every miniapp result has to be produced against a environment that
   develops this checkout (`Pkg.develop(path=".")` in a scratch environment, keeping
   `miniapps/Project.toml` unchanged), or it says nothing about the cap work. The symptom that
   exposed it was `compute_ρg!` rejecting the `air_phase` keyword that this source supports.
8. **First real result, and a problem it exposes.** `BrittleDuctile2D_VariationalDYREL.jl`
   (marker-chain free surface, cut cells, `solve_VariationalDYREL!`) run against this checkout
   at `64 x 24` for 16 kyr, cap and cap-free control:
   - The ductile branch of the solved far-field column lands on the analytical strength
     envelope below about 15 km in both runs. The brittle branch is still far below it, because
     16 kyr of loading only reaches about 35 MPa against a 190 MPa peak.
   - Both runs open volumetrically near the free surface. The cap run reaches `2.3e-4` there
     against `9.7e-5` for the control, and its opening is strongest in the parts of the surface
     the plastic-strain seed does not touch. So the cap does produce more near-surface tensile
     opening, but `EVol` alone is not the clean discriminator the plan assumed: the control's
     dilatant cone (`Ψ = 3°`) opens volumetrically too. A sharper diagnostic is needed, for
     instance restricting the comparison to cells whose deviatoric stress is far below the
     shear envelope.
   - **The cap run stops converging once plasticity switches on.** Steps 1-7 converge; from
     step 8, when `EVol` first grows, every step stalls at `err` around `1e-5` to `4e-5`
     against a `1e-6` tolerance. The control converges at every step. The local return map is
     not the suspect — a failed local solve would surface as a nonfinite residual, not a
     stalled one — so this is the outer iteration. The obvious experiment is the relaxation
     that item 1 measured as harmful on the easy shear band: it may be what a hard problem
     needs. Until this is resolved the benchmark cannot produce a publishable comparison.
9. The Sect. 4.5 benchmark below is drafted as
   `miniapps/benchmarks/stokes2D/brittle_ductile/BrittleDuctile2D.jl` and runs, but it has not
   produced a result yet. Two things were established before writing it: the Table 1 column
   reproduces the intended physics (analytical strength envelope peaks near 190 MPa with the
   brittle-ductile transition at 11-12 km, inside the 25 km domain), and GeoParams applies a
   uniaxial-to-invariant correction to `DislocationCreep` — a factor 5.3 at `n = 3.3` — unless
   `Apparatus = Invariant` is passed, which the script does. A first run at 128x48 did not
   converge: against a 1e23 crust, 1e17 sticky air stalls the pseudo-transient solve at 100k
   iterations. With 1e20 air and a `(1e20, 1e24)` cutoff the step converges in 20k iterations,
   at the cost of a stiffer free surface. Loading is slow by construction — one 1 kyr step adds
   about 3 MPa against a 190 MPa envelope — so plasticity only switches on after roughly ten
   steps, and nothing about the cap-versus-control comparison can be read off a short run.

## Completion condition and deferred work

Complete when tensile-only and mixed cap states satisfy the scaled coupled residual tolerances, the integration tests converge with consistent pressure/history handling, ordinary DP remains compatible, and local failures cannot be silently accepted. An increase in outer iterations alone is not a correctness fix.

Defer global Newton/Krylov tangents, changes to phase homogenization, and a general adaptive-timestep framework. Reconsider them only after the corrected constitutive solve is measured in representative simulations. If timestep retry is later added, it must restore all physical history before retrying.

## Final benchmark: brittle-ductile transition (paper Sect. 4.5)

The last validation step reproduces the paper's Sect. 4.5 test (Fig. 9). It is the only test in the paper that exercises the cap together with temperature- and stress-dependent creep, strain softening and a free surface, so it is the one that checks the corrected return map in the regime JustRelax is actually used in. Run it only after Steps 1-4 pass; it is an end-to-end validation, not a debugging tool.

### Purpose and discriminating result

The setup is the crustal extension model of Sect. 4.3, enlarged so that the brittle-ductile transition falls inside the domain. Two features must be reproduced:

1. Mode-II shear bands in the brittle upper part, at orientations consistent with the Sect. 4.3 result.
2. Near-surface **vertical mode-I failure zones** (Fig. 9b), where the confining stress is smallest. The paper states these are not reproducible without a tensile yield surface. This is the discriminating observable: run the identical setup with a cap-free Drucker-Prager rheology and the vertical tensile zones must disappear, while the shear bands remain. A run that shows them in both cases indicates a setup or diagnostic error, not a success.

Plot accumulated deviatoric viscoplastic strain over the whole domain, accumulated volumetric viscoplastic strain in a near-surface window, and the far-field depth profile of the effective deviatoric stress (the strength envelope, Fig. 9c). The envelope must show a brittle branch, a peak, and a ductile branch inside the domain; if the transition is not inside the domain the run says nothing about the coupling.

### Setup

- Domain `100 x 25 km`, top boundary stress-free.
- Background horizontal extension at a constant rate; plane strain.
- Background geothermal gradient `20 degrees C/km`, with a temperature- and stress-dependent dry upper crust quartzite rheology (Schmalholz, Kaus and Burg, 2009) combined with the linear creep term. Creep prefactors follow Eq. (8), `A_D = B_D exp(-E_D/(R*T))` and `A_N = B_N exp(-E_N/(R*T))`; Table 1 lists no `E_D`, so the linear term is temperature independent and equivalent to a constant viscosity `1/(2*B_D) = 1e23 Pa s`.
- Softening is applied to cohesion only, explicitly between timesteps, per Eqs. (24) and (50): `c = max(c_init + H_c*eps_acc, c_min)` evaluated from the accumulated deviatoric viscoplastic strain at the beginning of the step. The friction angle is not softened here.
- Localization is triggered by random perturbations of the accumulated deviatoric viscoplastic strain, clustered in the central upper part of the domain, as in Sect. 4.3.
- Reported result time is 33.5 kyr.

Table 1, `Brittle-ductile` column:

| Parameter | Value |
| --- | --- |
| `rho` | `3e3 kg/m^3` |
| `B_D` | `5e-24 Pa^-1 s^-1` |
| `B_N` | `8.8971e-25 Pa^-n s^-1` |
| `E_N` | `1.9e5 J/mol` |
| `n` | `3.3` |
| `G` | `5e10 Pa` |
| `K` | `1.1e11 Pa` |
| `phi` | `30 degrees` |
| `Psi` | `3 degrees` |
| `c_init` | `2e7 Pa` |
| `c_min` | `5e6 Pa` |
| `H_c` | `-5e8 Pa` |
| `pT` | `-1e6 Pa` |
| `eta_vp` | `1e19 Pa s` |
| `L x H` | `1e5 x 2.5e4 m` |
| element area `A` | `3.6e3 - 2.5e4 m^2` |
| `dt` | `1e3 yr` |
| `exx_bg` | `1e-15 s^-1` |
| `ezz_bg` | `0` |

Note the nonzero dilation angle `Psi = 3 degrees`: unlike the Sect. 4.4 tensile test, this benchmark does not isolate pure tensile opening from dilatant shear, and volumetric plastic strain has two sources.

### Differences from the paper's discretization

State these in the miniapp header rather than presenting the run as a like-for-like reproduction.

- The paper uses unstructured Crouzeix-Raviart triangles with variable resolution: a `40 x 7 km` refined block at `3600 m^2` (about 60 m), a `25000 m^2` background (about 500 m), and a 20 km linear transition zone. JustRelax uses a uniform staggered grid, so choose either a uniform grid at the refined spacing (about `1667 x 417` cells, a GPU run) or a coarser uniform grid, and report which. Localization width is regularization-controlled through `eta_vp`, so a coarser grid is acceptable only if the `eta_vp = 1e19 Pa s` band is still resolved by several cells; check this before interpreting band widths.
- The free surface must be an actual free surface. The near-surface mode-I zones are the target observable and they depend on low confining stress at the top, so the free-surface treatment used (variational Stokes or sticky air) has to be stated, and a sticky-air layer's viscosity and thickness reported.
- The paper uses global Newton with adaptive timestep halving on non-convergence. This plan defers both. Run at a fixed `dt = 1e3 yr` and report any step where the outer iteration or the local return map fails to converge, rather than silently continuing. If many steps fail, that is a result about the PT/DYREL outer loop, not evidence against the local solve.
- Temperature is prescribed by the geothermal gradient. If it is held fixed in time, say so; the paper's setup does not depend on thermal evolution over 33.5 kyr.

### Placement and acceptance

Add the script next to the existing cap benchmarks, e.g. `miniapps/benchmarks/stokes2D/brittle_ductile/BrittleDuctile2D.jl`, with a DYREL counterpart under `miniapps/DYREL2D/` if the DYREL path is also wired. Accept the benchmark when:

1. The far-field strength envelope shows the brittle-ductile transition inside the domain.
2. Mode-II bands develop in the brittle part with the expected orientations.
3. Near-surface vertical mode-I zones appear with the cap and vanish in the cap-free control run.
4. The local return map converges at every cell of every accepted step, with reported residual maxima, and the accumulated plastic strains are advanced exactly once per accepted physical timestep.
5. Results are reported for at least two grid resolutions at fixed `eta_vp`, so that any claim of regularized, resolution-insensitive localization is measured rather than assumed.

A visually plausible figure is not sufficient on its own; items 4 and 5 are what connect the picture to the corrected constitutive solve.
