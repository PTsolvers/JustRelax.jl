# Matrix-free geometric multigrid with DR smoothing and Anderson acceleration

Standalone design and implementation plan for JustRelax.jl (JR). Prepared 2026-10-12.

**Implementation order: (1) hierarchy and transfer design; (2) GMG with a dynamic-relaxation smoother; (3) safeguarded Anderson acceleration. GPU and MPI requirements belong to every stage.**

**Mandatory implementation rules:** use structs for meaningful solver state and configuration; keep all new GMG code type stable and free of runtime dynamic dispatch; follow the existing `src/` style and backend traits; place the new GMG solver and associated implementation in a dedicated `src/GMG/` folder; support GeoParams material models through JR's existing rheology interfaces. These are acceptance requirements from Stage 1 onward, not later cleanup tasks. Sections 10.2, 10.4, and 10.5 make them concrete.

This document proposes work; it does not report a production implementation or new validation results. Numerical thresholds called *proposed gates* are acceptance targets to freeze before experiments, not established performance guarantees. The prior prototypes provide evidence for choosing an architecture, but their assumptions must not become undocumented restrictions in JR.

## 1. Intended result and design decisions

Build an opt-in, coupled velocity-pressure geometric V-cycle that uses stencil operations on every level, including the coarsest. Fine-grid equations, constitutive laws, physical time stepping, and boundary data retain their JR meanings. DR smooths oscillatory error; coarse correction removes error that DR damps slowly. Anderson later accelerates an already validated cycle map.

The first supported path is standard DYREL on nested Cartesian grids, initially 2D and then 3D. Its production target includes CPU, CUDA, AMDGPU, and decomposed MPI, including GPU+MPI. Variational DYREL requires a separate retained-row/mask extension described below; it must not be advertised as supported by passing full-volume tests alone.

| Decision | Initial choice | Reason / condition for changing it |
|---|---|---|
| Solver organization | Coupled GMG outer iteration, DR smoother | Stronger local evidence than DR outside a velocity-only MG preconditioner |
| Cycle | V-cycle | W-cycles increased work without a material benefit in the constant-coefficient screen |
| State representation | Native staggered arrays and field tuples | Avoid packing whole fields each cycle; preserve backend allocation and MPI layout |
| Coarse operator | Rediscretized, homogeneous Picard/secant operator | Each level executes local kernels; explicitly test coefficient and transfer consistency |
| Transfer implementation | Destination-owned gather kernels | Avoid floating-point atomics and shared scatter writes on GPUs |
| Restriction | Mass-weighted adjoint for residual densities | Boundary, nonuniform, and semicoarsening weights follow geometry |
| Velocity prolongation | Freeze after Stage 1 comparison of the tested interpolant and a conservative compatible alternative | Smooth interpolation alone does not prove Stokes compatibility |
| Pressure prolongation | Parent-cell injection initially | Preserves constants; pressure residual restriction is volume averaging |
| DR smoother | Short, restarted Chebyshev-form DR on velocity, followed by PH pressure relaxation | Reproducible high-frequency target; separate from a fully converged standalone DYREL solve |
| Coarsest solve | Bounded DR/PH iteration with a global residual-drop test | No sparse assembly, LU, pseudo-inverse, or host field solve |
| Communication | Explicit context for each level | Fine-grid IGG metadata cannot implicitly describe the hierarchy |
| Anderson | Off until Stage 2 gates pass; depth 3 first candidate | Existing evidence supports small full-state history, but gains were fixture dependent |
| Plastic softening of the preconditioner | Off by default; separate ablation | Prototype shrink 0.4 is empirical, not a material law |

Matrix-free means that the solver stores fields, geometry, coefficients, diagonal scaling, transfer stencil metadata, and work vectors. It does not store a global/system matrix or a coarse factorization. Tiny explicitly assembled matrices are allowed in test oracles only. The final Anderson least-squares factor is small and dense; it is not a discretization matrix.

The implementation should not initially add FAS, Newton-Krylov, an AMG dependency, global coarse-space enrichment, automatic parameter optimization, or a second public solver family. Those are separate responses to measured limitations, not prerequisites to porting the demonstrated method.

## 2. Lessons that determine the plan

These findings are carried forward as constraints, so this plan is usable without reading the original experiment log.

| Learned fact | Consequence for implementation |
|---|---|
| Frozen constant-viscosity DR-smoothed GMG needed about 7-8 cycles through 256² in the tuned idealized hierarchy | Pursue GMG outside DR, but reproduce contraction with matrix-free coarse solves and charge all setup/communication |
| The matrix-free VEP prototype applied JR kernels on every level and converged over a 20-step shear-band sequence | Reuse residual/stress algebra and the step lifecycle; replace CPU loops and implicit global state |
| Constructing coarse `Geometry(ni, li)` with fine IGG active silently used fine spacing | Derive coarse coordinates explicitly; test coordinates and derivative scaling on every level |
| Stored `η_vep = τII/(2 εII)` used plain strain rate; using it for the elastic secant produced the wrong operator | Use the invariant of the effective strain tensor, including old stress, or use a direct effective-viscosity stress kernel |
| Plastic tangent probing produced negative velocity diagonal entries; their cause was not resolved | Do not treat that tangent as SPD; begin with a positive Picard/secant velocity block and investigate invalid scaling explicitly |
| Full coarsening degraded with stretched cells; semicoarsening worked in frozen Vanka tests | Carry per-direction coarsening from the first hierarchy design; revalidate it with DR rather than transplanting the result |
| Contrast 10,000 stalled DR-smoothed GMG; rediscretized two-level correction could become unstable | High contrast is an explicit hardening gate, not an assumed property of harmonic coefficient restriction |
| Weak-region Vanka treatment helped a different frozen hierarchy | Retain its diagnosis as a lead; its dense region solves are not a matrix-free GPU/MPI solution |
| Tight Picard inner solves lost to production DR; loose solves reduced work | Use bounded smoothing and inexact coarse solves; avoid oversolving coefficients that will change |
| Anderson improved plastic GMG cycles, but PH-map Anderson was mixed on linear and power-law cases | Add it last, with an unaccelerated parent and equal physical accuracy |
| Resetting DR history sometimes cost substantial work | Count restart cost and define all cycle-map state before accelerating it |
| Momentum/continuity alone could falsely accept a nonlinear solve | Final acceptance must include constitutive closure and use a fresh consistent state |
| Weak air could produce large field differences at an apparently acceptable residual | Compare against tighter reference fields; retain whole-domain and rock-only diagnostics separately |
| Shared-machine wall times varied materially; older result tables describe earlier algorithms | Compare adjacent repeated cohorts, record work, and identify exact method/version for each claim |

Negative projection, line, Schwarz, and single-interface-patch results also remain relevant: a correction that reduces a frozen velocity energy or works in a tiny coupled linear system can still lose total time or fail nonlinear acceptance. Reuse their operator/state oracles where appropriate, but do not accumulate those correction families inside GMG without a diagnosed mode and a new cost-inclusive experiment. In particular, the successful frozen weak-region Vanka study and the rejected nonlinear single-patch study are different methods and evidence levels.

The VEP matrix-free prototype's reported 20-step cycle totals at 128² were about 678 with full-vector Anderson depth 5, 694 with depth 3, and 2,912 without Anderson, with two unaccelerated steps exhausting their budget. This supports an eventual acceleration experiment; it **does not** establish that unaccelerated nonlinear GMG is already robust. Stage 2 therefore separates linear GMG correctness from nonlinear robustness, and may record a plastic convergence limitation before Stage 3. Anderson must not be used to excuse an incorrect transfer or an unstable frozen cycle.

The current source includes standard DYREL in the CPU, CUDA, and AMDGPU 3D modules. Older design notes describe it as 2D-only. Source inspection establishes integration seams, not validated 3D GMG performance. Variational DYREL still needs its own dimension/backend audit.

## 3. Mathematical contract: physical residual versus solver residual

### 3.1 Unknowns, fixed histories, and retained degrees of freedom

Let `x = (v, p)` contain independent staggered velocity and cell-pressure unknowns. Boundary values, physical ghosts, and duplicated MPI overlap entries are not additional unknowns. A periodic seam has one owner. Mask-eliminated rows are excluded in the variational extension.

During one physical step, old stress `τ_o`, old pressure `P0`, physical material history, prescribed boundary data, and `dt` are fixed inputs to the nonlinear solve. Stress, strain, viscosity, plastic multipliers, and pressure corrections are evaluated consistently at trial states. Physical histories advance exactly once after accepted convergence.

Use the existing JR pressure law, including supported source and thermal terms. In a simple finite-bulk case,

```text
F_p(x) = -div(v) - (p - P0)/η_b + Q/dt + thermal_source,
η_b = K_b * dt.
```

The physical momentum residual `F_v` contains the production stress divergence, pressure gradient, body force, plastic pressure correction, and enabled free-surface stabilization. Its precise signs and retained rows come from JR kernels, not a newly coded continuum formula.

### 3.2 Keep augmentation and physical acceptance separate

DR uses the PH numerical pressure `γ_eff * F_p`. Define an augmented residual, with `Grad` meaning JR's discrete pressure-gradient operator,

```text
r_aug(x) = T_γ F(x)
         = (F_v(x) - Grad(γ_eff * F_p(x)), F_p(x)).
```

For variable `γ_eff`, differentiate the product; replacing it by `γ_eff * Grad(F_p)` changes the operator. Keep the large hydrostatic `P` separate from small numerical/plastic corrections, as current fused JR kernels do.

The coarse correction operator `K_l` approximates `-d(r_aug)/dx` using frozen positive secant coefficients. The correction equation is `K_l e_l = rhs_l`, and the residual of that equation is `rhs_l - K_l e_l`. Thus the fine correction right-hand side must be the matching augmented residual. Restricting the physical residual while applying an augmented coarse operator is a different algorithm and needs a derivation; do not silently mix them.

Final stopping, trial acceptance, and reference comparisons use fresh **physical** momentum and continuity residuals plus constitutive closure. A small augmented or diagonally scaled momentum residual does not establish physical force balance.

### 3.3 Nonlinear state must be explicit

`(v, p)` is a sufficient map state only when all constitutive trial quantities can be recomputed deterministically from `(v, p)` and fixed histories. If viscosity is an independently relaxed iterate, include `log(η)` at centers/vertices in the nonlinear map state or provide a defined constitutive refresh that makes it derived state. Apply the same rule to iterative plastic multipliers. Do not leave hidden mutable quantities outside the map and then accelerate only `(v, p)` as if the map were stationary.

For the initial VEP path, prefer an isolated constitutive evaluator with explicit local closure and fixed old stress. For power-law compatibility, test both the derived-state evaluator and the existing relaxed-viscosity lifecycle before choosing one. A solver change must still satisfy the original rheology's closure, not just its lagged coefficient equations.

### 3.4 Nullspaces and compatibility

Detect supported nullspaces from the actual problem: constant pressure is a candidate for an incompressible closed/periodic problem, but finite compliance or prescribed traction can determine pressure absolutely. Periodic velocity may also have translation modes. Disconnected masked regions may have more than one pressure component.

Represent each verified null vector `Z_l`, require `K_l Z_l ≈ 0`, and preserve it through prolongation. Remove a pressure mean only when that gauge actually exists. Project residuals against the correct **left** nullspace when needed; the augmented coupled operator need not be symmetric. For homogeneous periodic Stokes, audit compatibility of total forcing rather than silently modifying a noncompatible physical problem. Use communicator-aware owned-DOF reductions for every projection.

## 4. Stage 1A: hierarchy, geometry, ownership, and communication

Stage 1 is a design-and-proof milestone for the GMG foundation. It delivers transfer kernels and numerical oracles, not a nonlinear solver. Its design must be frozen before the Stage 2 production cycle is built.

### 4.1 One explicit descriptor per level

Each level describes:

- Dimension, global cell counts, physical domain, local owned global index ranges, and array axes.
- Explicit center/vertex/velocity coordinates, spacing accessors, and cell/face dual volumes.
- Which directions were coarsened and the fine-to-coarse global parent mapping.
- Retained pressure/face unknowns, physical-boundary tags, periodic seam ownership, and ghosts/overlaps.
- Communicator, Cartesian neighbors, exchange widths, persistent pack buffers, and reduction masks.
- Backend and scalar type; coefficients/work arrays live on that backend.

The existing 2D storage contract is pressure `(nx, ny)`, `Vx` `(nx+1, ny+2)`, and `Vy` `(nx+2, ny+1)`. In 3D, `Vx` is `(nx+1, ny+2, nz+2)` and the other components follow cyclic permutations. These are **storage** sizes; independent unknown counts depend on BCs, periodicity, and MPI ownership. Never derive global owned counts from these shapes alone.

Construct coarse vertex coordinates by selecting nested fine vertices. Set coarse `ni` from those coordinates. Use the explicit-coordinate `Geometry` path or an equivalent level-aware construction; retain the global domain length separately from local subdomain extents. Never invoke a coarse uniform-grid constructor that queries the active fine IGG dimensions.

For uniform full coarsening, check `h_c = 2 h_f` in each coarsened direction. For nonuniform nesting, each coarse width equals the sum of its child widths. Test linear derivatives and quadratic viscous scaling against analytic/manufactured fields; do not require different discretizations to be exactly identical on arbitrary smooth fields.

### 4.2 Coarsening policy

Use a direction tuple such as `(2,2)`, `(1,2)`, or `(2,1)` in 2D and its 3D analogue. The prototype's isotropic first case uses full coarsening. Semicoarsening is a first-class shape/transfer capability, even before its performance policy is enabled.

Start with globally nested even cell counts and aligned rank ownership. If a direction cannot coarsen legally, leave it uncoarsened; if no direction can coarsen, stop. Do not halve local array lengths including overlaps, silently drop a last cell, or change the physical extent. An odd-grid extension uses an explicit final unpaired child and geometry-dependent volumes; it needs separate tests before support is declared.

For stretched grids, test coarsening the strongly coupled direction using actual spacing and coefficient information. The y-only success in older fixtures is evidence for those fixtures, not a rule to hard-code y. Freeze a tested directional policy; record coarse-grid growth and total cost.

### 4.3 MPI adapter: explicit decision and early proof

JR currently uses IGG for fine-grid halos, while `norm_mpi` and related helpers reduce on `MPI.COMM_WORLD`. That does not by itself provide independent multigrid level contexts. The documented IGG lifecycle is initialization, halo updates, and finalization; GPU-aware transport is supported where the runtime provides it. Verify the **installed version's** handling of multiple array shapes, overlap, and contexts before relying on it. [ImplicitGlobalGrid documentation](https://github.com/eth-cscs/ImplicitGlobalGrid.jl).

Initial architectural choice: reuse the caller's fine Cartesian topology and MPI runtime, but give MG its own level-aware exchange/reduction interface. A proven IGG path may implement a level exchange behind that interface. Otherwise implement narrowly scoped MPI pack/send/receive/unpack for coarse level arrays with preallocated buffers. Do not initialize/finalize a global IGG grid inside each recursive cycle, mutate IGG private globals, or call ordinary fine `update_halo!` on arbitrary coarse arrays.

Required interface, names illustrative:

```julia
exchange_level!(level, fields, location)
level_sum(level, local_owned_sum)
level_max(level, local_owned_max)
restrict_across_levels!(transition, coarse_rhs, fine_rhs)
prolong_across_levels!(transition, fine_correction, coarse_correction)
```

For the first distributed hierarchy, keep all ranks active and coarsen only while each rank has enough aligned owned cells for the stencil. Global parent numbering determines weights and ownership across rank boundaries. Start with blocking communication for a correctness oracle, then add overlap once dependencies are proven.

Transfers can need a wider neighbor footprint than a momentum stencil. Compute that footprint from the chosen interpolation and transpose restriction. Corners/edges matter for tensor-product interpolation; either exchange them explicitly or use staged directional exchanges that correctly forward newly received data. Make every read's dependency visible in the level schedule.

Periodic wrapping in current kernels requires auditing: a local array endpoint on a decomposed rank is not automatically the global periodic seam. Physical BCs and seam wrapping use global boundary tags, with interior-rank reads supplied by the level exchange.

### 4.4 Coarse rank agglomeration is a scalability milestone

Keeping all ranks active proves correctness but eventually stops coarsening too early. Before claiming scalable MPI GMG, add deterministic Cartesian rank grouping where coarse work per rank becomes too small:

1. Form child groups and an active coarse communicator at setup, with a fixed root and index mapping.
2. Restrict each owned fine contribution once and redistribute to the designated coarse owners.
3. Execute coarse halos and reductions on the active communicator only.
4. Send corrections back and prolong on fine owners.
5. Keep inactive parent ranks in the prescribed transition communication, never in child collectives.

Do not gather a large coarse problem onto a CPU or replicate it on all ranks. An eventual one-rank coarsest level can remain on that rank's GPU and use matrix-free DR. Bound the aggregate memory and communicate failure back to all parent ranks. Test the entire schedule for deadlocks, buffer lifetimes, and communicator cleanup.

## 5. Stage 1B: prolongation and restriction design

The two operators are **prolongation** (coarse correction to fine) and **restriction** (fine residual to coarse). They are different from coefficient averaging and physical-state restriction. Give these operations separate names and contracts.

### 5.1 Algebra and weights

Write `P_l` for prolongation from coarse level `l+1` to fine level `l`. It is a block operator on velocity components and pressure. For residuals stored as equation densities, define

```text
R_l = M_c^(-1) P_l^T M_f,
<P_l c, f>_Mf = <c, R_l f>_Mc.
```

`M_f` and `M_c` are positive diagonal geometric measures on independent retained DOFs: cell volume for pressure and matching staggered dual volume for momentum densities. Physical constitutive coefficients and DR diagonals do not belong in these masses. Zero-volume eliminated rows are removed before forming this identity; they are not inverted.

On uniform, fully coarsened interior grids, equal density measures give the familiar `R = Pᵀ/4` in 2D and `Pᵀ/8` in 3D. Semicoarsening gives the corresponding factor 1/2 when one direction is coarsened. These constants are shortcuts **after** the actual DOF/BC geometry is checked. Boundary rows, partial aggregates, masks, and nonuniform dual volumes use their true weights.

If a future operator stores integrated forces rather than densities, derive its restriction as a dual/load transfer instead of applying this density formula by habit. Test any row scaling explicitly. Restrict unpreconditioned residuals and apply the coarse `D_c` afterwards; `R(D_f⁻¹ r)` generally differs from `D_c⁻¹ R r`.

Do not assume `R P = I`; transpose full weighting of a smooth interpolant generally is not an inverse. Do not renormalize each row to force constant restriction if that breaks the chosen weighted adjoint identity. Derive boundary extension/elimination first, then construct the transpose of the resulting retained-DOF map.

### 5.2 Pressure

Initial pressure correction prolongation copies the parent-cell value to its children. Volume-adjoint restriction therefore averages child pressure residual densities with their cell volumes:

```text
(R_p r)_C = sum(child_volume * r_child) / coarse_volume.
```

This preserves a constant pressure gauge when it exists and conserves the integrated continuity residual. Piecewise-constant pressure is deliberate; it does not promise reproduction of a linear pressure field. Test gradient response, alternating pressure modes, and two-level coupled contraction. Increase pressure interpolation order only if those tests show a specific deficiency and the velocity/pressure pair remains stable.

### 5.3 Velocity: prototype interpolant and conservative reference

First reproduce the successful prototype's velocity transfer as a performance reference:

- In a component's face-normal direction, inject coincident coarse faces and interpolate between neighboring coarse faces at fine interior faces.
- In each transverse direction, use coordinate-based linear interpolation between coarse face-center values. On a uniform 2:1 interior grid, the weights are 3/4 and 1/4.
- Treat physical boundaries through the homogeneous correction BC, not unconditional clamping. Periodic neighbors wrap globally.
- Obtain restriction by the weighted transpose of those exact interpolation weights.

This interpolant is smooth, but it is **not automatically flux conserving or divergence commuting**. Transverse interpolation can change the mean of the fine subfaces covering a coarse face. The prototype's tiny adjoint tests do not establish either property.

Implement a second, simple conservative reference. Copy a coarse face velocity to all transverse child subfaces; interpolate linearly in physical coordinates between opposite coarse faces inside a parent cell. On an unmasked nested Cartesian grid this gives both coarse-face flux preservation and

```text
Div_f P_v = I_p Div_c,
```

where `I_p` injects a parent divergence density to its children. Thus a coarse divergence-free field remains divergence-free. This reference sacrifices transverse linear-field reproduction and is not presumed to be the fastest choice.

For the smooth candidate, measure the normalized commutator defect `Div_f P_v - I_p Div_c`, parent-integrated divergence, and coarse-face flux mismatch. If those defects explain a pressure/continuity regression, develop a compatible smooth transfer using this bounded construction:

1. Reconstruct transverse subface values using fixed linear slopes and true coordinates.
2. Correct each coarse-face subface mean so the area-weighted child flux equals the original coarse-face flux.
3. Interpolate interior child faces.
4. Correct only parent-interior face fluxes so child divergences equal the injected coarse divergence. Flux preservation makes this local constraint compatible; its sum is redundant.
5. Express the chosen minimum-weighted-norm local correction as fixed small stencil weights. For uniform geometry use closed-form/precomputed weights; for nonuniform geometry compute metadata once at setup. Its transpose defines restriction.

This is a proposed transfer design, not a proven property of JR's current interpolant. Keep it linear within an operator epoch: slope limiting or coefficient-dependent switching changes its transpose and must trigger a transfer rebuild. Assemble a tiny local reference only in tests to verify weights, solvability, and the adjoint. No per-cycle host patch factorization is permitted.

**Stage 1 decision:** freeze the least expensive pair that preserves the required boundary/nullspace structure and passes the coupled transfer tests. Exact commuting behavior is mandatory for any pair advertised as compatible. A noncommuting smooth pair may be retained only with its defects reported and a frozen two-grid proof of stable contraction; do not claim flux conservation for it. If neither candidate passes, Stage 2 waits for a corrected transfer design.

### 5.4 Boundary and decomposition contracts

For an error/correction equation, prescribed velocity corrections are zero. Free-slip uses zero normal correction and the correct tangential derivative condition; no-slip fixes both. Physical pure/simple-shear backgrounds belong to the fine state or boundary lifting, not the coarse error equation. Homogeneous traction/free-surface correction conditions must retain their operator-dependent stress coupling.

Test constants only when admissible: a constant pressure gauge or periodic velocity translation is meaningful; an arbitrary nonzero constant velocity violates a no-slip correction BC. Test affine velocity reproduction in the interior separately from boundary correctness.

Every output has one owner. Prolongation gathers coarse values after their halos are valid. Restriction gathers the full fine support of a coarse DOF, including contributions crossing ranks; never omit the neighbor contribution or count replicated overlap twice. The transpose must include the same physical-boundary elimination and periodic-seam identification as prolongation.

### 5.5 Coefficient transfer is a separate operation

Start with the prototype's harmonic averaging as a candidate for positive scalar creep/effective viscosity:

```text
η_C = coarse_volume / sum(child_volume / η_child).
```

Determine whether to coarsen creep viscosity or the effective secant viscosity before forming stress coefficients; those choices need not commute with heterogeneous `G*dt`. Use the effective operator coefficients as the design contract and test the creep reconstruction against them. Vertex/edge coefficients use their actual production staggered locations and boundary extension, with coefficient halos synchronized before averaging.

Transfer compliance `C = 1/η_b` according to the discretized pressure row, not by blindly averaging `η_b`; retain `C=0` for incompressible cells. Restrict augmentation, material parameters, and enabled stabilization consistently with the coarse residual definition. For a uniform scalar augmentation, use the same physical `γ_eff` on all levels rather than recomputing unrelated penalties from coarse spacing. Heterogeneous augmentation needs an explicit policy and product-gradient tests.

Particles, phase labels, and constitutive histories do not move to coarse levels each V-cycle. Restrict the frozen effective coefficients required by the coarse error operator. A coarse phase-ratio container is only needed if the reused kernel demands it; populate it deterministically and verify it produces the intended coefficients. Do not reconstruct a new nonlinear physical problem on every coarse grid.

### 5.6 Stage 1 acceptance gate

Deliver a transfer design record, implementations, and tests covering:

| Proof | Cases / proposed tolerance |
|---|---|
| Shapes and bounds | 2D/3D, every direction tuple, minimal grids, rectangular grids, supported axes; no out-of-bounds reads |
| Coordinates | Same domain; exact nested vertices; correct summed coarse widths and derivative scaling |
| Pressure | Constants, volume conservation, gauge prolongation where applicable |
| Velocity | Admissible constants, normal linear reproduction, declared transverse order, flux/divergence defects |
| Weighted adjointness | Random independent fields; relative discrepancy about 1e-12 in Float64 CPU tests, backend-aware tolerance on GPUs |
| Homogeneous BCs | Free-slip, no-slip, prescribed shear lifting, periodic seams; BCs commute with transfer to the extent claimed |
| Explicit oracle | Tiny assembled P/R from basis responses matches stencil action; actual retained DOF counts, not packed prototype assumptions |
| MPI identity | Serial versus 2/4-rank global action, seam/corner support, unique reductions and cross-rank parent cells |
| GPU identity | CUDA and AMDGPU actions match CPU references within declared precision tolerance; scalar indexing disabled |
| Level isolation | Alternating fine/coarse operations never changes caller IGG geometry or its next fine halo update |

Stage 1 also uses a tiny scalar diffusion and coupled Stokes two-grid analysis to screen the coarse space. An exact coarse solve in this **test-only oracle** distinguishes transfer/coarse-operator defects from smoother/coarse-DR defects. Eliminate verified gauges before computing spectra. Freeze the chosen pair and document all supported geometry/BC restrictions before building the production V-cycle.

## 6. Stage 2A: matrix-free level operator and coefficients

### 6.1 Homogeneous error operator

Provide an `apply_level_operator!(out, level, e)` that evaluates a frozen linear error operator with:

- Zero physical body/source forcing and zero old-pressure/old-stress contributions to the error RHS.
- Homogeneous correction BCs, with the existing operator-dependent surface terms retained where supported.
- Frozen positive secant stress coefficients and physical compliance; no new material iteration.
- Matching augmented numerical pressure, with fresh halos for every differentiated center field.
- Explicit input/output/workspace ownership and no mutation of committed physical state.

Setting histories to zero in the **error workspace** removes affine forcing; it does not erase the real histories in fine nonlinear evaluation. Verify `K(0)=0`, linearity, and agreement with unit-column responses of the corresponding homogeneous JR residual.

Prefer reusing the same low-level stress/divergence/gradient algebra with an explicit level context. Reuse `StokesArrays` where its shapes fit; avoid copying the complete time-step state to every level solely to call a heavyweight constructor. If an existing residual wrapper performs fine IGG exchange or material updates internally, split at that ownership seam before reuse.

For moderate heterogeneous coefficients, use rediscretization. In test diagnostics, compare it against `R K_f P` and an exact tiny reference. Recursive matrix-free Galerkin action is possible but can repeatedly invoke fine-grid work; it is not a free scalable fallback. A strong discrepancy or divergent correction triggers coefficient/transfer analysis, not an unconditional switch to sparse Galerkin matrices.

### 6.2 Secant construction and plasticity

For a Maxwell-like model, compute the **effective strain tensor** before its invariant:

```text
ε_eff = ε + τ_o / (2 G dt)
η_s   = τII / (2 * invariant(ε_eff)).
```

Use JR's exact invariant convention and staggered shear interpolation in both 2D and 3D. The invariant of a tensor sum is not the sum of invariants. Handle zero strain, purely elastic limits, infinite `dt`, nonfinite material data, and finite compliance explicitly.

The prototype recovers creep viscosity through `1/η_creep = 1/η_s - 1/(G dt)`. Near the elastic limit that subtraction is ill-conditioned. Prefer a dedicated homogeneous secant stress evaluation using positive `η_s` directly, with the original fine nonlinear residual unchanged. If creep reconstruction is temporarily used, test its limits and admissibility; do not hide failures under an arbitrary large clamp.

A trial `plastic_shrink` multiplies only the positive preconditioning/secant coefficients in yielding regions. It must not change the yield surface, returned fine stress, physical viscosity audit, or committed plastic history. Compare shrink 1.0 with the prototype's 0.4 and a held-out setting under the same gates. Freeze no default from one shear-band case.

### 6.3 Diagonal and spectral bounds

The smoother needs positive velocity diagonal `D_l` and a reliable upper bound for its fixed-pressure augmented velocity block `A_l`. Start by comparing existing Gershgorin/diagonal kernels with exact tiny level responses. They are reusable only if they describe **this** secant, augmentation, BC, spacing, and stabilization.

Use colored operator probing as an independent setup/reference path when analytic formulas are not yet proven. In the prototype, 3×3 colors per velocity component gave 18 applications in 2D. The coloring is valid only if same-color points do not couple through the actual operator. Periodic seams, short grids, wider stencils, and 3D change that requirement. Use global colors under MPI and verify wraparound conflicts; do not assume local index modulo 3 is universally valid.

For a weighted-symmetric positive velocity block, analyze `D⁻¹/² A D⁻¹/²` in the appropriate DOF inner product. Power iteration with a safety factor is an estimate, not a certified upper bound. Combine it with a valid Gershgorin bound or an explicit safety/restart rule; test underestimation on tiny spectra. If the block is not positive/symmetric, Chebyshev's SPD interval argument does not apply; diagnose or select a separately derived relaxation.

Warm-start spectral vectors across coefficient refreshes with stable global-DOF initialization. All norms and Rayleigh products are global over owned DOFs on the level communicator. Freeze bounds during one smoothing sweep; a material/operator change invalidates its recurrence.

### 6.4 Operator epochs and cache lifetime

A hierarchy cache belongs to a solver workspace; avoid the prototype's global `Dict{Any,...}`. Its key includes backend/device, scalar type, dimension, geometry, partition/communicator identity, coarsening pattern, BC topology, masks, and enabled operator terms.

Separate:

1. **Topology setup:** coordinates, ownership, transfer metadata, buffers, communicators; reuse until structural input changes.
2. **Coefficient refresh:** secant coefficients, their coarse restrictions, diagonals and spectral bounds; reuse allocations.
3. **Iteration state:** residuals, corrections, recurrence, audits; overwritten during the solve.

Every coefficient/transfer/BC revision increments an operator epoch. Restart DR at the revision and, later, clear Anderson history. A scalar-only coefficient update may not need to rebuild transfer metadata. A coefficient-aware transfer or changed cut-cell topology does.

Use fresh coefficients each cycle for the first nonlinear correctness reference. Compare reuse afterwards against that reference: fixed maximum age, globally measured log-coefficient changes, yield/mask changes, and contraction loss can trigger a refresh. Prototype refresh 20 is an experiment seed, not a safety guarantee. Keep constitutive refresh, coefficient refresh, residual audit, and logging cadences separate; do not repeat the overloaded `nout` design.

## 7. Stage 2B: DR smoother and matrix-free V-cycle

### 7.1 Define a smoother, not a nested standalone solve

Do not recursively call `solve_DYREL!` as the smoother. That function owns physical-step initialization, outer stopping, coefficient updates, and history commitment. A smoother is a bounded primitive with explicit fields, RHS, frozen operator epoch, and recurrence workspace.

For a linear correction level, hold pressure fixed during `k` velocity steps. With velocity residual `q_v = rhs_v - (K_l e)_v`, use short Chebyshev-form DR targeting

```text
[a,b] = [λ_upper / spectral_range, λ_upper].
```

This deliberately targets high-frequency error rather than an accurate global estimate of the smallest eigenvalue. It is a second-order polynomial relaxation; it is not identical to production DYREL's adaptive whole-spectrum tuning.

An explicit recurrence, with `θ=(a+b)/2`, `δ=(b-a)/2`, `σ=θ/δ`, is:

```text
ρ = 1/σ
d = D⁻¹ q_v / θ
v += d
for j = 2:k
    recompute q_v at the updated v, with p held fixed
    ρ_new = 1 / (2σ - ρ)
    d = (ρ_new * ρ) * d + (2ρ_new / δ) * D⁻¹ q_v
    v += d
    ρ = ρ_new
end
recompute q_p = rhs_p - (K_l e)_p
p += ω_p * γ_eff * q_p
```

If `a=b`, take the ordinary scaled Richardson limit instead of dividing by zero. Handle `k=1` explicitly. The pressure update uses fresh continuity after the velocity changes, not a stale pre-sweep value. Bounds, `γ_eff`, and diagonal scaling remain fixed during the sweep.

On the finest nonlinear level, recompute production nonlinear stress and augmented residual at every velocity substep. Keep the positive secant diagonal/bounds frozen for that sweep. The linear Chebyshev convergence argument does not prove nonlinear stability; bound/trial safeguards and fresh physical audits remain necessary.

Reset recurrence at the start of each smoothing sweep, after coarse correction, and after a coefficient epoch changes. Never combine a previous direction with residual differences across an intervening pressure/material/coarse update without a derived recurrence. Count those resets in comparisons with production DR.

Candidate seeds from existing experiments are `k=5`, `spectral_range=8`, `ν_pre=ν_post=1` for the matrix-free VEP case and 2 for the earlier frozen hierarchy. Screen a small declared grid of settings, freeze them, and then run hold-outs. They are not universal defaults.

Velocity SPD bounds alone do not guarantee coupled PH/DR stability. Include finite-compliance pressure relaxation and the full coupled one-sweep amplification matrix in tiny tests. Earlier per-substep pressure experiments found a high-frequency instability that damping alone did not cure. Keep pressure updates outside the `k` velocity substeps initially; do not switch to simultaneous updates by convenience.

### 7.2 Linear V-cycle

Use this correction convention on every recursive level:

```text
vcycle!(l, e, rhs):
    enforce homogeneous correction BCs and verified gauge constraints
    if l is coarsest:
        coarse_dr!(l, e, rhs)
        return coarse status

    repeat ν_pre times: dr_smooth!(l, e, rhs)
    r = rhs - K_l(e)
    rhs_c = R_l(r)
    e_c = 0
    recurse vcycle!(l+1, e_c, rhs_c)
    e += P_l(e_c)
    enforce correction BCs; exchange correction halos; reset recurrence
    repeat ν_post times: dr_smooth!(l, e, rhs)
    return status and counters
```

Initialize the coarse correction to zero each visit. Old coarse solution fields are scratch, not physical history. Transfer the complete coupled RHS, not independent scalar velocity Poisson equations with pressure left behind.

The coarsest solver repeats DR/PH sweeps until a globally measured coupled residual has dropped by a prescribed factor, or a hard sweep/work budget is exhausted. Candidate drop factors 0.1 and 0.01 should be compared; the existing note and implementation used different values, so record the actual one. A wider interval such as `spectral_range=100` is a candidate for coarse relaxation, subject to actual bounds and measured convergence.

Normalize velocity and pressure residual blocks with fixed scales so one cannot hide the other. Handle a zero RHS exactly. Return `converged`, `budget_exhausted`, `stagnated`, or `nonfinite` explicitly; a budget-exhausted coarse solve is not success. An inexact finite coarse correction can still be used if the enclosing cycle's safeguards accept it, with its status logged. A nonfinite correction must never be applied.

On MPI, stopping tests, budgets that affect control flow, and exception flags are collective over the level communicator. Do not let one rank exit smoothing while peers enter another exchange. The final fine-level status is communicated to the caller's full solver communicator.

### 7.3 Nonlinear fine cycle

The intended nonlinear method is a nonlinear fine-level smoother with frozen linear coarse error correction:

```text
g_epoch(x):
    perform ν_pre nonlinear fine DR/PH sweeps
    evaluate fresh r_aug(x)
    rhs_2 = R_1 r_aug(x)
    e_2 = linear_vcycle(level=2, initial=0, rhs=rhs_2)
    trial = x + P_1 e_2
    validate finite fields, BCs, gauges and constitutive admissibility
    safeguard the nonlinear coarse correction if needed
    perform ν_post nonlinear fine DR/PH sweeps
    evaluate fresh physical residuals and constitutive closure
    return the resulting state and audit
```

This is a nonlinear defect-correction cycle. It is not FAS: no coarse nonlinear physical equation or FAS tau correction is implied. Coarse coefficients approximate the current fine response; the true fine constitutive law determines acceptance.

For a one-level hierarchy, take the well-defined fine DR path and skip nonexistent level 2. Do not assume all supported grids have at least two levels, as the prototype driver does.

Introduce a transactional safeguard for a nonlinear coarse correction before Anderson exists. Begin with finite/admissible checks and a declared physical-residual merit test. If the full correction fails, try a bounded sequence of reduced corrections, then restore the pre-correction state and continue with a declared fine DR recovery. Count all trial residual evaluations. Tune no acceptance policy on a hold-out case after seeing its outcome.

Do not impose strict decrease after every velocity substep: a second-order smoother can have transient growth. Audit after meaningful sweeps/corrections and use the established physical contract. Conversely, a linear two-grid cycle with an unstable error-propagation spectrum must be repaired rather than hidden behind repeated rejection.

### 7.4 Halo dependency schedule

The actual schedule is derived from kernel reads. At minimum:

| Point | Required valid fields |
|---|---|
| Before strain/divergence | Velocity face halos and physical correction/state BCs |
| After constitutive/stress evaluation | Edge/vertex shear stress halos; any other stress field subsequently read across a rank |
| Before augmented pressure gradient | Pressure, continuity/numerical pressure, plastic pressure correction, and coefficient halos needed by the stencil |
| After velocity/pressure update | Updated independent fields before the next neighboring read |
| Before restriction | Fine residual support, including the transfer's corner/edge reach |
| Before prolongation | Coarse correction support, including transverse interpolation neighbors |
| After coefficient refresh | Coefficient support used by staggered reconstruction, diagonal/bound kernels, and operator action |

Physical BCs apply only on physical faces. MPI exchanges synchronize interior rank boundaries. Verify the ordering against JR's fused kernels: fusion may eliminate a stored field, but it does not eliminate a neighboring data dependency. Coarse helpers that internally call fine IGG exchange must be bypassed or given the explicit level context.

## 8. Stage 2C: GPU, MPI, and robustness hardening before Anderson

### 8.1 GPU execution contract

All level fields, residuals, diagonals, recurrence vectors, transfer buffers, coefficient metadata used by kernels, and later history vectors reside on the selected backend. Allocate through JR's backend constructors/traits. GPU extension methods forward to shared implementation; they do not duplicate cycle algorithms.

Implement transfer, recurrence, copies, linear combinations, mask operations, and coefficient restriction with ParallelStencil conventions initialized in the correct dimension-specific module. Avoid scalar array access, threaded CPU loops, `Vector{Float64}` state assumptions, global concatenations, and full-field `Array(...)` conversion. Host orchestration may manipulate scalars and metadata.

Prefer one writer per destination, fuse operations only after equivalence is proven, and reuse buffers. Repeated V-cycles with a frozen hierarchy should have stable device memory use and no grid-sized fresh allocation. Report persistent memory, peak memory, and cumulative allocation separately. This is especially important in 3D: full coarsening adds about 1/7 of fine-level cells, whereas semicoarsening can approach another full fine-level count, before stencil scratch and histories.

Use device-local reductions followed by the necessary MPI scalar reductions. Support both GPU-aware direct buffers and explicitly managed pinned-host staging for communicating **halo/redistribution buffers** when the MPI runtime needs it. A GPU solve must not silently fall back to CPU operator execution. Synchronize producing GPU work before MPI consumes a buffer, and MPI completion before a GPU reads it; prove correctness before overlapping streams and nonblocking communication.

Run both CUDA and AMDGPU gates on actual hardware. Backend-neutral source and CPU tests establish useful structure but do not establish accelerator correctness. FP64 is the first precision target because current module initialization uses Float64; mixed precision or FP32 is a separate validated extension.

### 8.2 MPI correctness and scaling contract

All residual norms, dot products, gauge means, coefficient-change tests, spectral estimates, trial decisions, and final statuses use owned retained DOFs and the correct communicator. Existing world-communicator helpers can be used only where their semantics match; add MG-scoped reductions for agglomerated levels. Average with global weighted sums/counts, not an unweighted average of local means.

First compare one fixed global manufactured problem under 1, 2, and 4 ranks, including `2×1`, `1×2`, and `2×2` partitions, periodic seams, and cross-rank transfer parents. In 3D include decompositions spanning more than one direction and edge/corner exchange. Poison ghosts before synchronization so accidentally reading stale values causes a failure.

Then validate agglomerated versus unagglomerated cycle action and solutions. Test repeated workspace construction/destruction and two solves with different hierarchies in the same MPI runtime. The solver must leave the caller's fine IGG context usable.

For performance, separate strong scaling of a fixed global problem from weak scaling at fixed owned work per rank. Record level sizes, active ranks, communicated bytes, reduction count, packing time, coarse work, and maximum rank wall time. A flat cycle count with a coarse-communication bottleneck is not scalable time to solution.

### 8.3 High contrast and anisotropy

Define a training and hold-out ladder: contrast 1, 10, 100, 1,000, and 10,000; inclusions resolved at centers and vertices/edges; translated inclusions; aspect 1, 4, and 8; combined contrast and stretch. Do not require the initial transfer to magically pass the entire ladder. Require truthful support limits and a diagnosed failure before adding machinery.

If contraction fails, distinguish:

1. **Invalid operator/bounds:** wrong spacing, coefficients, augmentation, boundary rows, or diagonal scaling.
2. **Poor smoothing:** oscillatory weak-region/interface modes survive the DR sweep.
3. **Poor coarse representation:** modes remain even with exact tiny coarse solves.
4. **Inexact coarse failure:** the transfer/ideal cycle is good but coarse DR exhausts its budget.

Use actual error modes from tiny operators or matrix-free diagnostics. More global damping or a deeper hierarchy does not necessarily repair a pressure/interface mode.

Semicoarsening is the first bounded anisotropy experiment. If necessary, investigate line/region **matrix-free DR** inside the weak region while retaining the requested DR smoother structure. Coefficient-aware/energy-based interpolation is a separate transfer change requiring a new adjoint and support proof. A dense expanding-region Vanka inverse is not an acceptable production substitute for this GPU/MPI matrix-free design.

Do not assume a weak-region DR patch will reproduce the earlier Vanka result. If DR-based hardening cannot pass the requested contrast regime, explicitly narrow the supported envelope or propose a separately approved smoother extension. Anderson is not the repair mechanism for a divergent linear cycle.

### 8.4 Variational/free-surface extension

Add this after standard full-domain GMG is correct, with its own transfer/operator decisions:

- Retain the original `RockRatio` pressure/face/vertex masks and the original eliminated-row convention.
- Coarsen physical rock volumes/face measures; avoid inventing coarse connections between disconnected rock regions or letting pressure leak through eliminated air cells.
- Derive masked transfer and its adjoint on retained DOFs. Simply zeroing a full-grid prolongation afterwards changes its transpose and can lose gauge/connectivity structure.
- Compare coarse geometry/weights with aggregated fine fluxes, especially tiny and shifted cut cells. Reject an unsupported coarsening transition rather than destroying an interface.
- Preserve the enabled free-surface stabilization and homogeneous traction response in coarse operator coefficients; it is not equivalent to a free-slip wall.
- Audit **unweighted retained continuity** as well as the weighted equation residual. Tiny volume fractions must not conceal a physical defect.
- Audit pressure nullspaces per connected retained region and test pressure behavior near the surface.
- Test flat/sloping surfaces, translated interfaces, very small fractions, evolving masks, and an accepted finite-step trajectory. Topology changes invalidate transfers and all recurrence/acceleration history.

The earliest variational milestone may support only masks whose topology survives coarsening. State that restriction explicitly. Full-volume variational tests are useful identity checks, not cut-cell validation. Do not infer 3D variational support from standard 3D DYREL inclusion.

### 8.5 Stage 2 acceptance gate, with Anderson disabled

The gates distinguish correctness, supported-envelope robustness, and performance:

| Gate | Required evidence |
|---|---|
| Operator | Homogeneous linearity; tiny unit-column oracle; verified nullspaces; correct augmentation/compliance/surface terms |
| Smoother | Actual recurrence matches tiny amplification action; positive valid scaling; high-frequency damping and coupled stability |
| Two-grid | Exact-coarse oracle isolates transfer quality; production matrix-free coarse solve gives stable finite correction |
| Hierarchy | Consistent result for 1/2/many levels; explicit status for coarse budget exhaustion; no stale level state |
| Linear refinement | Proposed target: asymptotic physical residual contraction ≤0.5 per V-cycle on isotropic constant-coefficient fixtures; cycle counts through 64²/128²/256² vary by no more than about 25% under frozen settings |
| Nonlinear viscosity | Fresh momentum, continuity, and center/vertex constitutive defects pass the declared target; no false frozen-viscosity acceptance |
| VEP lifecycle | Fine true-stress smoothing; immutable old histories during trials; one accepted commit per step; shear-band trajectory and field comparison |
| Backend/decomposition | CPU, CUDA, AMDGPU, CPU MPI, CUDA+MPI, AMDGPU+MPI actually exercised; precision-aware equality to references |
| Robustness | Supported BCs, geometry, contrast/aspect envelope declared; failures retained; hardening changes re-run transfer/operator gates |
| Cost | All setup, coefficient refresh, transfer, halo, reductions, coarse work, rejection, and final audit charged |

The 25%/0.5 targets are proposed engineering gates for the initial linear family, not claims for plasticity or strong contrast. Establish 3D analogues at feasible global sizes and record unavailable hardware as unvalidated, not passed.

Unaccelerated nonlinear plastic GMG can legitimately have a long convergent tail, as earlier results showed. Record that limitation and its budget honestly. Stage 3 can study acceleration of such a map once its linear correctness, finite-state behavior, physical acceptance, and lifecycle are proven. A frozen unstable cycle, invalid operator, false convergence, or MPI failure blocks Stage 3.

## 9. Final stage: safeguarded Anderson acceleration

### 9.1 Accelerate one complete cycle map

Let `g_epoch(x)` be one complete GMG+DR cycle with fixed topology, transfer, coefficients, controls, and fixed physical histories. Define `f_k = g_epoch(x_k) - x_k`. Use Type-II Anderson:

```text
ΔF = [f_i - f_(i-1)] over the retained history
ΔX = [x_i - x_(i-1)] over the same pairs
γ = argmin || W_x (f_k - ΔF γ) ||₂
x_AA = g_epoch(x_k) - (ΔX + ΔF) γ.
```

This is Anderson's cycle-map least squares. It is different from Stage 2 E9's projected minimization using physical-residual directional probes; do not substitute that method while labeling it Anderson. The standard fixed-point formulation and QR-based implementation are described by [Walker and Ni (2011)](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf).

Initially use full retained velocity and pressure state, plus any explicit constitutive iterate required by Section 3.3. Physical histories are never acceleration unknowns. Use depth 3 first, and compare 0/1/3/5. The old velocity-only result was worse; retain velocity-only acceleration only as an ablation if memory motivates it.

### 9.2 Scaling and least-squares stability

`W_x` includes owned geometric measures and fixed component scales. Choose physical `V_ref`, `P_ref`, and any log-viscosity/multiplier scales at the start of a solve or epoch; a dimensional seed is `P_ref = η_ref * V_ref / L_ref`. Provide deterministic finite fallback scales for zero initial fields. State scaling and **equation residual acceptance scaling** are different objects.

Project history directions onto admissible BC/gauge space. Do not change weights every iteration without transforming or discarding the history. For a decomposed field, exclude ghosts and periodic duplicates from all history products.

Avoid the prototype's unqualified normal equations: a ridge on `ΔFᵀΔF` does not remove their squared-conditioning problem. Use distributed weighted QR with reorthogonalization and a small rank-revealing solve:

1. Orthogonalize device-resident weighted difference fields with global dot products on the fine solver communicator.
2. Reorthogonalize and obtain the tiny triangular factor and projected RHS.
3. Solve the small factor with rank-revealing QR/SVD on the host; discard unresolved directions.
4. Broadcast identical coefficients/decision if the small solve is done on one rank.
5. Apply the resulting field combination on the device.

No full history field moves to the host. Batch independent dot products per orthogonalization pass where possible, respecting the dependencies between QR columns. Pack reductions to reduce latency; do not claim a one-reduction QR implementation without a stability proof.

Use an explicitly tested rank threshold and coefficient/trust bound. Nearly repeated cycle increments, small `f_k`, and nonfinite or excessive coefficients trigger truncation, restart, or a plain-cycle result.

Budget memory before enabling acceleration. A straightforward implementation storing `ΔF`, `ΔG=ΔX+ΔF`, and temporary orthogonal fields needs up to `3m` history-sized fields plus several state/trial buffers. An incremental QR design can reduce that, but must pass history-drop tests. The prototype's 10/14 new-vector counts at depth 3/5 are not a guaranteed production memory bound. Expose estimated/peak device memory and fail construction clearly if requested history cannot fit.

### 9.3 Trial transaction and acceptance

Always retain the ordinary-cycle result `x_base = g_epoch(x_k)` and its fresh physical audit. An Anderson failure must be able to return exactly that result.

Evaluate `x_AA`, or a bounded damped interpolation `x_base + β*(x_AA-x_base)`, in reusable trial workspace with fixed physical histories. Validate finite fields, constitutive admissibility, masks, gauges, and BCs. Recompute all physical residual blocks and constitutive closure; never accept solely because the least-squares predicted norm fell.

Initial acceptance policy: the candidate must improve a fixed coupled physical merit relative to `x_base`, respect declared componentwise growth limits, and fit a state trust bound. A component already meeting its target must stay within its declared target allowance. Freeze the merit scales, growth allowance, damping sequence, and trial budget before measuring hold-outs. This allows component tradeoffs without permitting continuity or rheology to disappear behind momentum reduction.

A rejected/nonfinite trial restores all primary and auxiliary state: velocity, pressure, viscosity, trial stress/strain, plastic multipliers, `ΔPψ`, DR scratch/recurrence, and aliases in `args`. It must not increment accumulated strain, rotate/overwrite old stress, or commit old pressure. Preallocated trial storage replaces repeated `deepcopy` in the production path, with exact state-identity tests.

All ranks decide acceptance collectively. If one rank finds an inadmissible value, peers still complete the scheduled collective status exchange; no rank-local exception may strand other ranks in a halo or QR reduction. On rejection, return `x_base`, log the cost, and clear or truncate history according to the declared policy.

### 9.4 History lifetime

Clear history after physical-step changes, coefficient/transfer epoch changes, mask/connectivity changes, BC/control changes, recurrence recovery that changes the map, and failed trust/rank conditions. Do not keep cross-step or cross-operator history by default.

Coefficients are refreshed **between** accelerated cycle evaluations, never halfway through the nominal map. A deterministic nonlinear constitutive evaluation within a cycle is part of the map; changing lagged coefficient setup is an epoch change. If adaptive stopping makes the coarse work vary, it must be deterministic from map inputs and carry its status; independent rank-local adaptation is invalid.

Useful history needs consecutive cycle evaluations in the same epoch. The initial Stage 2 reference that refreshes coefficients every cycle will therefore repeatedly reset Anderson. Validate a safe reuse window against that reference before claiming an acceleration benefit; do not postpone a required coefficient refresh merely to keep history.

### 9.5 Final-stage gate

Require exact equivalence to Stage 2 when depth is zero, correct tiny affine-map behavior, dependent-column truncation, admissible gauge/BC directions, global least-squares consistency across decompositions, epoch reset, immutable physical histories, and complete rollback after forced NaN/overflow/rejection.

Benchmark four accepted controls under equal physical accuracy: tuned production DR, production DR+Anderson where a valid map is available, GMG+DR without Anderson, and GMG+DR+Anderson. The DR+Anderson control is needed to attribute gains, not to gate the transfer implementation. Count reset, history, QR, reductions, trial audits, and rejections.

Promote acceleration only where repeated total time/work improves and robustness/field accuracy are preserved. Retain an off switch and an unaccelerated fallback. Strong VEP gains and mixed PH-map results justify this final experiment, not default acceleration for every problem.

## 10. JR integration, API, and physical-step lifecycle

### 10.1 Opt-in API

Prefer one opt-in workspace keyword on `solve_DYREL!` over a second public solver family. Preserve existing argument order, plain keywords, and bundled `kwargs` support through `flatten_solver_kwargs`. With no MG workspace supplied, the existing path must reproduce current behavior. The opt-in is a thin route to the new GMG solver implementation in `src/GMG/solver.jl`; GMG iteration logic and helpers do not live in the existing DYREL folder.

Illustrative future API; these names are proposals, not currently callable functions:

```julia
mg = GMG(
    backend, stokes, dyrel, grid, flow_bcs, igg;
    options = GMGOptions(
        coarsest_cells = 16,
        pre_sweeps = 1,
        post_sweeps = 1,
        velocity_steps = 5,
        spectral_range = 8.0,
        anderson_depth = 0,
    ),
)

solve_DYREL!(
    stokes, ρg, dyrel, flow_bcs, phase_ratios, rheology, args, grid, dt, igg;
    multigrid = mg,
    # Existing material, physical-step, and boundary options keep their meaning.
)
```

Interpret `coarsest_cells` as a stopping target per coarsened direction, subject to legal nesting/partition; document its exact semantics before export. Allocate topology/workspace once, and populate step-dependent coefficients at solve time. Keep advanced experimentation controls internal until evidence warrants a public knob. Controls that affect correctness or budgets need explicit validation and diagnostics.

Return existing residual-history fields where compatible and add clearly defined cycle/work counters and convergence status. A cycle is not one DYREL iteration: report smoothing substeps, level operator applications, coarse sweeps, coefficient refreshes, physical audits, trial probes, rejected corrections, halo calls/bytes, and global reductions. Existing `total_iterMax` semantics must not silently become a cycle budget; define a separate MG budget or an explicit documented conversion.

Make recovery visible. On numerical failure, restore the last accepted physical-step iterate, reset numerical recurrence, and optionally invoke the existing DR path under the remaining declared budget. A solve that succeeded only by fallback is `fallback_used`, not an MG performance success. Invalid configuration fails early and collectively; it is not a reason to silently choose a different geometry/backend.

### 10.2 Proposed source ownership

All new GMG solver code and associated functions belong in the dedicated `src/GMG/` folder, separate from `src/DYREL/`. Keep the reusable transfer/communication core separate from constitutive decisions within that folder. These are logical responsibilities; combine small files if that fits repository conventions. Existing source files receive only the necessary includes, exports, trait forwarding, thin opt-in routing, or narrowly scoped reuse of existing primitives.

| Proposed location | Owns |
|---|---|
| `src/GMG/types.jl`, `constructors.jl` | Parametric configuration, level/workspace, smoother, material-context, status, and Anderson structs; backend allocation |
| `src/GMG/levels.jl` | Explicit level geometry, ownership, transitions, cache identity, workspace lifecycle |
| `src/GMG/transfers.jl` | Shared transfer dispatch and weights |
| `src/GMG/transfers2D.jl`, `transfers3D.jl` | Dimension-specific ParallelStencil transfer kernels, if separate definitions are needed |
| `src/GMG/communication.jl` | Level exchange, owned reductions, redistribution/agglomeration |
| `src/GMG/operator.jl` | Homogeneous augmented secant action and coefficient restriction |
| `src/GMG/secant.jl` | GeoParams/JR material evaluation, effective-strain/secant construction, explicitly experimental plastic shrink |
| `src/GMG/smoother.jl` | Restarted DR/PH primitive, bounds, coarsest relaxation |
| `src/GMG/cycle.jl` | Linear recursion, nonlinear fine cycle, statuses/work accounting |
| `src/GMG/solver.jl` | New GMG solve orchestration and accepted-step transaction |
| `src/GMG/anderson.jl` | Final-stage history, stable least squares, collective safeguarded trial |
| `src/GMG/solver_VS.jl` | Later masked-operator integration, with separate support gates |

Before implementation, read JR's `.agents/api.md`, `.agents/grid.md`, `.agents/solvers.md`, and `.agents/mpi.md`. Include reusable shared definitions through `src/common.jl` or its include tree; include dimension-specific kernels only in their matching initialized modules. Keep root types/constructors and dimension-specific kernel definitions consistent with JR's existing split.

Wire all new public names through CPU/CUDA/AMDGPU traits, imports, and exports. Audit `src/JustRelax_CPU.jl`, `src/ext/CUDA/2D.jl`, `src/ext/CUDA/3D.jl`, and both AMDGPU mirrors. A symbol available only inside an extension's private module is not a complete public backend implementation. Verify actual loading order and test with accelerator modules loaded before JR.

### 10.3 Step transaction

Factor only the lifecycle pieces genuinely shared with the existing solver:

1. Begin the physical step once: establish `P0`, frozen old stress/history, material/buoyancy inputs, prescribed BCs, and applicable multiplier initialization.
2. Prepare/refill the MG coefficients without committing stress or accumulated plastic strain.
3. Run bounded GMG cycles, later with optional Anderson trials. Every audit refers to one consistent iterate.
4. Evaluate fresh physical and constitutive acceptance. On budget exhaustion return nonconverged status and preserve the explicit recovery state.
5. Commit accepted constitutive fields and update accumulated strain/volume and `τ_o` once. Recompute diagnostics such as divergence that fused iteration kernels do not store.

Inspect every mutated field and alias, including `args.P`, center/vertex stresses, `λ/λv`, `ΔPψ`, `η/ηv/η_vep`, and DYREL recurrence arrays. A nominally read-only residual function that advances a multiplier or old stress is not suitable for probes without isolated scratch and a defined evaluation order.

### 10.4 Structs, type stability, and repository style

Use structs whenever data has a persistent identity, ownership, invariant, or lifecycle. Prefer parametric immutable structs holding mutable backend arrays, as JR's `StokesArrays` and `DYREL` do. Use a `mutable struct` only where scalar bookkeeping or owned references genuinely need mutation. Keep small statically shaped bundles as concrete `NamedTuple`s or tuples when that already matches JR; do not introduce a wrapper with no useful contract.

Proposed ownership types are `GMGOptions`, `GMGLevel`, `GMGWorkspace`, `GMGMaterialContext`, `DRSmootherState`, `AndersonState`, and a concrete solve-status/work-counter container. Names and exact fields remain implementation decisions. Their fields must expose concrete inferred types through parameters: geometry, arrays, transfer metadata, communication context, rheology, argument bundles, scalar type, and policy objects. Do not store `Any`, abstractly typed arrays or levels, unparameterized containers, or `Function` fields. A callable field is `f::F` with specialization on `F`; an abstract argument annotation is acceptable if the call specializes on the concrete argument and its fields remain inferable.

All new GMG paths must be type stable, including construction, material refresh, transfers, DR, cycle recursion, reductions, rollback, and Anderson. Preserve one concrete return schema across success, budget exhaustion, stagnation, nonfinite trials, and fallback; change status values rather than returning differently shaped objects. Normalize and validate keywords once at entry into typed options, then cross a function barrier into the specialized implementation. Avoid global mutable caches, string-keyed policy lookups, runtime method lookup, and changing an iterate's scalar or container type inside a loop.

The hierarchy must not erase level types. Prefer `Vector{L}` with concrete `L` when all levels share the same representation; changing array sizes does not by itself require changing array types. Keep the nonlinear fine state separate from a homogeneous coarse-level vector when useful. If levels genuinely have different concrete types, use a typed tuple or typed recursive hierarchy and recurse through concrete elements. Do not dynamically index a heterogeneous tuple and assume dispatch stays static. Active/inactive MPI ownership must retain a known concrete communication representation, with participation encoded explicitly; agglomeration is not an excuse for `Vector{Any}`.

Follow JR's existing public-entry, backend-trait forwarding, and shared implementation pattern. Multiple dispatch remains the source-level mechanism, but calls must resolve from concrete types without runtime dynamic dispatch. Resolve backend, dimension, transfer/smoother policy, and material evaluator through concrete traits or typed callables; use `Val(N)` where dimension specialization matches existing kernels. Runtime sweep counts and tolerances remain values rather than creating a new compiled method for every parameter choice.

Reuse JR naming, `!` for mutation, constructor allocation conventions, field-access helpers, docstring style, `flatten_solver_kwargs`, and ParallelStencil kernel organization. Match nearby `src/` code and Runic formatting; do not introduce a second backend abstraction or a configuration framework. Root-level type definitions may be included from `src/GMG/types.jl`; common wrappers/constructors and per-dimension kernels must enter the same module layers as their existing counterparts, without duplicating type definitions or cross-including 2D/3D kernels.

Type stability and absence of dynamic dispatch are explicit review gates. Check representative concrete specializations with `@inferred` where appropriate and inspect inferred IR for unresolved calls, including the body of functions that return `nothing` or a simple status. Use `@code_warntype` as a diagnostic, not as sole proof: a concrete return type alone does not establish an inferable body. Exercise heterogeneous phase tuples, every supported backend/dimension, active/inactive coarse ranks, optional Anderson, and failure/rollback paths. GPU compilation must resolve material/helper calls on device; CPU inference is not accelerator proof. After warm-up, account for allocations and reject avoidable per-cell/per-sweep host allocations and new device work buffers.

### 10.5 GeoParams support is part of the solver contract

Accept JR's existing GeoParams-based material representations: a single `GeoParams.MaterialParams` where the JR material path supports it, and concrete per-phase material tuples with `JustPIC.PhaseRatios` for multiphase problems. Carry the concrete rheology and `args` `NamedTuple` types in `GMGMaterialContext`; preserve phase ordering, scalar precision, units/nondimensionalization, and existing viscosity cutoff/relaxation semantics. Support the single-material route directly or through a tested one-phase adapter, rather than forcing users to replace material laws with hand-built constant-viscosity arrays.

Evaluate the fine physical residual, constitutive closure, and nonlinear DR substeps through JR's existing GeoParams rheology algebra. Reuse the applicable viscosity evaluators in `src/rheology/Viscosity.jl`, phase averaging in `src/phases/`, modulus/thermal helpers in `src/rheology/GeoParams.jl`, and stress/yield primitives used by DYREL. Cover supported linear viscous, nonlinear creep/power-law, elastic/compressible, and plastic/composite material paths. A specialized constant-coefficient fixture may remain a useful oracle, but cannot be the production material interface. Document and reject unsupported material/device combinations early and collectively rather than silently changing their physics.

Coarse levels solve linear correction equations using numeric frozen secant/effective material coefficients derived from those same fine GeoParams evaluations. They do not create replacement coarse material histories or reevaluate a nonlinear material trajectory independently. Restrict derived coefficient fields according to Section 6, retaining compliance and staggered locations; do not average material structs or phase labels as though they were residuals. Old stress, old pressure, accumulated plastic strain, temperature, and other material inputs retain their declared physical-step lifecycle. Rebinding `args.P` to trial pressure must be explicit and reversible, and temperature/material argument fields remain on the selected backend with the correct grid locations.

The device material evaluator must use concrete, GPU-compatible GeoParams objects/callables and JR's existing kernel argument conventions. Validate device compatibility at setup; no host per-cell material loop, CPU material fallback, abstract material collection, or dynamically dispatched callback is allowed. New adapters belong in `src/GMG/` and reuse the established rheology functions rather than duplicating constitutive laws.

Add dedicated GeoParams compatibility tests: single material versus an equivalent one-phase mixture; a heterogeneous phase tuple with different law types; nonlinear viscosity dependent on the supported `T`/`P` arguments; finite `G`/`K` with nonzero old stress/pressure; and supported VEP/composite cases with fresh yield/closure audits. Compare material fields and physical residuals against independent calls to the existing JR path at identical trial states before comparing full solves. Repeat relevant fixtures on CUDA, AMDGPU, MPI, and GPU+MPI, verifying type inference, phase ordering, coarse coefficient refresh, trial rollback, and one-time history commitment. Any missing resource remains unvalidated in the support table.

## 11. Implementation milestones and reviewable units

Implementation should advance in the user's requested order. Parallel hardware verification within a milestone does not change that dependency order.

| Milestone | Concrete deliverable | Exit condition |
|---|---|---|
| M0: contracts | Actual kernel/include map; `src/GMG/` ownership; typed state/policy design; GeoParams interface; physical/augmented residual distinction; DOF ownership and support envelope | No unresolved sign, row-scaling, nullspace, history, source ownership, or material-interface ambiguity in the first fixture |
| M1: geometry/level context | Parametric structs, concrete hierarchy, explicit nested coordinates, coarsening tuples, level communication interface, CPU/GPU allocation | Coarse-spacing regression and fine/coarse isolation pass; inferred level/transfer calls without dynamic dispatch |
| M2: transfer design | Prototype interpolant, conservative reference, pressure transfer, weighted transpose, decision record | Stage 1 tests pass on CPU, both GPUs, and 2/4-rank transfer actions; selected pair frozen |
| M3: frozen two-grid | Homogeneous operator, diagonal/bounds oracle, DR primitive, test-only ideal coarse comparison | Stable coupled action and matching operator/recurrence tests |
| M4: unaccelerated hierarchy | Matrix-free V-cycle and DR coarsest solve, 2D standard integration | Linear refinement gate; full costs and statuses; no matrix/factorization path |
| M5: nonlinear lifecycle | GeoParams single/multiphase evaluation, true fine constitutive smoothing, coefficient epochs, transactions, stopping and recovery | JR material-field/residual parity, accepted power-law/VEP audits, type/dispatch checks, field/step comparisons; limitations recorded with AA off |
| M6: distributed/backend/3D hardening | GPU+MPI cycles, rank agglomeration, standard 3D path, BC coverage | Decomposition invariance and actual CPU/CUDA/AMDGPU resource gates |
| M7: supported-envelope hardening | Semicoarsening, contrast diagnosis, separately scoped variational extension | Honest geometry/rheology/contrast/mask support table; relevant Stage 2 gates pass |
| M8: Anderson | Full-map history, stable distributed least squares, safeguards, resets | Depth-zero equivalence, rollback/collective tests, accepted equal-accuracy benefit |
| M9: release proof | Documentation, examples, full applicable regressions, repeated held-out performance | Final completion contract below |

Stage 1 comprises M0-M2 plus its tiny numerical coarse-space screen. Stage 2 comprises M3-M7. Stage 3 comprises M8; M9 packages the validated result. GPU and MPI transfer validation in M2 prevents designing a serial hierarchy that later requires replacement.

If M7 exposes unresolved very-high-contrast or cut-cell cases, record them as unsupported and keep the core opt-in method scoped to the envelope it has passed. Do not expand into an unrelated nonlinear solver merely to declare all possible geodynamic problems handled. If those cases are required for the release, their failing gate remains open and release waits.

Each review unit includes its contract, smallest meaningful tests, type-inference/dispatch evidence for new paths, actual resource results, and a support-table update covering GeoParams material combinations as well as backend/geometry. Review compliance with the existing `src/` style and dedicated `src/GMG/` ownership at every milestone. Do not commit ignored simulation output, local manifests, generated documentation, or archived benchmark data as implementation source.

## 12. Validation and fair performance evidence

### 12.1 Test matrix

Suggested files follow the existing runner's `test_*` discovery and `*MPI*` separation conventions:

| Test responsibility | Suggested file(s) |
|---|---|
| Coordinates, field locations, transfer identities | `test_gmg_transfers2D.jl`, `test_gmg_transfers3D.jl` |
| Halo, transpose across partitions, level isolation | `test_gmg_transfers_MPI.jl` |
| Operator, scaling, spectral bound, DR recurrence | `test_gmg_dyrel_operator.jl`, `test_gmg_dyrel_smoother.jl` |
| Matrix-free coarse solve and linear hierarchy | `test_gmg_dyrel2D.jl`, `test_gmg_dyrel3D.jl` |
| Communicators, agglomeration, distributed cycle | `test_gmg_dyrel_MPI.jl` |
| Constitutive closure, lifecycle, exact rollback | `test_gmg_dyrel_nonlinear.jl` |
| Concrete state types, inferred bodies, no runtime dynamic dispatch | `test_gmg_types.jl` |
| GeoParams single/multiphase parity, material arguments, device support | `test_gmg_geoparams.jl`, `test_gmg_geoparams_MPI.jl` |
| Masks, retained continuity, traction/interface topology | `test_gmg_variational_dyrel.jl` |
| QR/rank truncation, map resets, depth-zero identity | `test_gmg_anderson.jl`, `test_gmg_anderson_MPI.jl` |

Share invariant/reference tests across backends, but run in correctly initialized backend/dimension processes. Include CUDA/AMDGPU hardware jobs explicitly. Current runner exclusions omit some existing 3D and MPI files on GPU; those exclusions must not accidentally skip the new support proof. GPU+MPI needs its own configured resource job, not a CPU MPI test with a GPU label.

Tiny explicit operators must be independent enough to catch implementation errors: use retained-DOF unit responses of JR's homogeneous physical kernels, check boundary rows, and compare both the augmented action and residual conversion. A test that reconstructs the same transfer formula twice proves little.

Run focused tests while developing, full applicable CPU regressions for solver/kernel/type changes, relevant CUDA/AMDGPU/MPI jobs, Runic on changed Julia files, and `git diff --check`. Build documentation when the public API and guide are added. Verify the actual environment before using a nominal test command: the earlier experiments lacked `test/Project.toml` and root `ParallelTestRunner`, while package extras/targets define test dependencies. Use an isolated proper test environment rather than committing a local manifest workaround. In this workspace, Julia execution should use the Julia tool; CI/MPI jobs use the repository's reproducible `--startup-file=no` launch convention.

### 12.2 Acceptance at one fresh consistent state

Freeze a physical problem-specific contract before comparing solvers. At minimum:

```text
momentum_score  = owned physical momentum RMS / fixed physical force scale
continuity_score = owned physical continuity RMS / fixed rate scale
constitutive_score = fresh original-law defect (centers and vertices/edges)
```

Each score must meet its declared tolerance. Also report max norms for localized residuals, retained/whole-domain counts, and mask-weighted diagnostics where relevant. For power-law viscosity, the earlier useful closure measure was the maximum absolute log-viscosity fixed-point defect from a unit-relaxation refresh using fresh stress. VEP additionally needs the actual yield/return-mapping and history consistency checks, not a viscosity-only surrogate.

Use fixed reference scales established from the physical input or initial state. Do not redefine the reference residual after another PH pass, normalize by a shrinking iterate, or loosen acceptance when a solver stagnates. Every exit, including the final partial budget, runs the required fresh audit. Nonfinite values cannot pass through an empty mask or comparison.

For field accuracy, compare with analytical/manufactured solutions where available and tighter accepted numerical references otherwise. Use gauge removal only for a real gauge. Report velocity, pressure, constitutive fields, stress, and physical-step observables. Near-zero fields need a declared absolute denominator floor. Compare velocity increments as well as total velocity where the VEP sequence requires it.

The old VEP field differences around 1e-4 and pressure around 1.3e-4 are reference observations, not automatic release tolerances. Set and test a field budget suitable for each benchmark and repeat at a stricter residual target. Keep whole-domain and rock-only errors visible; a low rock-only error does not erase the air-field result.

### 12.3 Benchmark ladder

Use the same serialized/frozen physical starts and boundary data for candidate and controls:

1. Manufactured constant-viscosity incompressible Stokes, free-slip/no-slip/periodic as admissible, 2D and 3D.
2. Finite compliance and nonzero old stress/pressure, to prove error-operator affine terms and lifecycle.
3. Heterogeneous inclusion and translated-interface fixtures, then anisotropy and combined contrast/stretch.
4. Nonlinear power-law viscosity with accepted constitutive closure and both frozen starts and finite-step loading.
5. The VEP shear-band sequence: all 20 steps, preserved histories, elastic/plastic breakdown, and tighter field reference.
6. Variational free-surface/cut-cell cases only once their dedicated support gate is implemented.
7. MPI partition changes, rank agglomeration, strong/weak scaling, and multi-GPU runs of accepted fixtures.

Compare against the strongest accepted production DR controls, not a known failed default or a weak idealized baseline. Freeze training controls before hold-outs. A failure receives a failure status and no speedup ratio.

Measure after disposable warm-up with alternating repeated cohorts, initially at least three repetitions and more when timing spread overlaps the claimed improvement. GPU timing includes synchronization; MPI time is the maximum rank time. Preserve raw settings, hardware/runtime versions, input hashes, source revision/diff hash, physical audits, exceptions, and all work counters.

Report both:

- **Cold/setup-inclusive solve:** workspace/topology preparation when applicable, coefficient setup, cycles, communication, trials, and final acceptance audit.
- **Reused-workspace step/trajectory:** all coefficient refresh and iterative costs, while reusing legitimate topology allocations; include total multi-step time.

Separate JIT compilation explicitly. Any independent post-run verification can also be timed separately, but required stopping audits belong in solver time. Report cycle count and volume-weighted operator work alongside wall time; these explain differences but do not replace actual runtime, especially with semicoarsening and agglomeration.

### 12.4 Promotion conditions

Promote an opt-in GMG configuration only after independent acceptance and field/history agreement pass in its declared envelope. A speed claim additionally needs a repeatable setup-inclusive or explicitly amortized gain over the accepted production reference, with the same accuracy. Do not require a universal win on small grids; record the crossover and permit deliberate DR selection.

Promote Anderson separately. Its gain is measured against the already accepted GMG+DR parent and attributed using the DR+Anderson control where valid. A larger cycle reduction that costs more in QR/communication/trials is not a time-to-solution improvement.

## 13. Risk register and response

| Risk | Early detection | Required response |
|---|---|---|
| Fine spacing leaks into coarse grids | Coordinate and derivative oracle | Explicit level geometry; no fine-global size queries in coarse construction |
| Restriction drops/doubles cross-rank contributions | Adjoint and serial/partition action tests | Global parent ownership, correct fine support, unique weighted reductions |
| Smooth transfer injects divergence/pressure artifacts | Flux/commutator and coupled two-grid tests | Compatible transfer repair or documented proven noncommuting pair |
| Harmonic averaging erases a weak interface | Translated inclusion, ideal coarse oracle, mode diagnosis | Coefficient/transfer hardening; restrict support if unresolved |
| Chebyshev interval is invalid | Actual tiny spectra, SPD/symmetry test, finite growth diagnostics | Correct bounds/operator or separately justified relaxation; do not tune blindly |
| Coupled pressure step unstable despite velocity bound | Full one-sweep amplification matrix, compliance tests | Derive coupled relaxation constraint; retain fixed-pressure substep structure |
| Coarse DR dominates or stalls | Coarse work/status counters | Better legal coarsest size/inexact policy; rank agglomeration; never hidden host LU |
| Nonlinear coefficient lag causes NaNs or false convergence | Fresh closure and epoch/reference comparison | Separate refresh cadence, recurrence reset, transactional recovery |
| Plastic secant/shrink changes physics | True-stress/yield/history audit | Limit changes to preconditioning; shrink stays experimental |
| MPI rank exits on local exception | Forced nonfinite and budget tests | Collective status protocol and consistent exchange schedule |
| AA history is ill-conditioned or belongs to old maps | QR rank/epoch tests | Truncate/restart and return the plain accepted cycle |
| Trial evaluation commits history | Complete state/alias rollback identity | Isolated reusable trial workspace and one final physical commit |
| GPU/MPI memory or synchronization failure | Peak memory accounting and multi-GPU tests | Preallocate, size history, explicit stream/buffer dependencies |
| Free-surface coarsening joins disconnected regions | Mask-connectivity and tiny-cut tests | Topology-aware transfers or stop/reject that coarsening |
| Apparent residual speedup has worse field accuracy | Tighter-reference and trajectory comparisons | Tighten/fix the contract and remeasure; retain failed comparisons |

## 14. Completion contract

The work is complete when:

- The transfer/hierarchy design has a documented mathematical contract and Stage 1 proof on the supported CPU/GPU/MPI cases.
- Meaningful solver state/configuration uses concrete parametric structs; all new GMG paths are type stable with no runtime dynamic dispatch, including GeoParams evaluation, MPI participation, and recovery.
- The new solver and associated functions live in `src/GMG/`, follow existing JR source conventions, and use the established backend/module wiring.
- GeoParams single-material and multiphase interfaces preserve JR material fields, physical residuals, argument semantics, and histories on the advertised CPU/GPU/MPI combinations; nonlinear/composite support is demonstrated with dedicated tests.
- GMG+DR uses matrix-free operators and coarse relaxation on every level, with correct coupled RHS, legal BCs/nullspaces, explicit budgets, and no hidden host field solve.
- Standard 2D/3D support, geometry restrictions, contrast/aspect limits, and any variational support are stated from actual validation, not inferred from common source.
- GPU+MPI correctness and coarse-rank scalability have been exercised on the advertised backends; missing resources remain clearly unvalidated.
- Nonlinear stopping includes constitutive closure; trial state is transactional; accepted physical histories advance once.
- The unaccelerated method's correctness and limitations are recorded independently of Anderson.
- Anderson can be disabled exactly, solves a stable globally consistent small least-squares problem, uses device-resident history, and returns the ordinary cycle on rejection.
- Equal-accuracy repeated benchmarks charge all work and attribute GMG, smoother, secant changes, and Anderson separately.
- Existing API behavior remains compatible with MG absent; new exports/traits, documentation, examples, formatting, and applicable regressions pass.

The first coding task is M0/M1 followed by the transfer suite in M2. It should end with a frozen transfer decision and a proven level communication/ownership contract. The next task is the unaccelerated two-grid and V-cycle implementation. Anderson begins only after those foundations satisfy their gates.

## 15. Evidence and source map

These references support provenance and integration; the design above contains the contracts needed to proceed independently.

| Local source | Use |
|---|---|
| [DYREL exploration results](DYREL_EXPLORATION_RESULTS.md) | Frozen transfer/coarse-space, anisotropy, contrast, DR/GMG, nonlinear and VEP findings; tables describe different prototype revisions |
| [Earlier multigrid method note](DYREL_MG_DESIGN.typ) | Matrix-free VEP method and porting pitfalls; this plan replaces its implementation sequencing and strengthens contracts |
| [Stage 2 linear results](miniapps/DYREL2D/free_surface_stabilization/dyrel_vs_variational/STAGE2_RESULTS.md) | Physical acceptance, weak-air field errors, mixed PH Anderson, costs and limitations |
| [Nonlinear power-law results](miniapps/DYREL2D/shear_band/convergence/NONLINEAR_RESULTS.md) | Constitutive closure, recurrence/reset costs, nonlinear false acceptance, field reference audits |
| [Pressure-coupling diagnosis](miniapps/DYREL2D/shear_band/relaxation_exploration/PRESSURE_COUPLING.md) | Actual coupled stability constraints and failed refinement behavior |
| [Matrix-free VEP prototype](miniapps/DYREL2D/shear_band/relaxation_exploration/vep_mg_mf.jl) | Operator applications, transfers, DR coarse solve, cached hierarchy and final AA ablations; CPU-specific assumptions require replacement |
| [Production DYREL solver](src/DYREL/solver.jl) | Backend entry, physical-step lifecycle, dimension helpers, fused kernels, audits and history commit |
| [Variational DYREL solver](src/DYREL/solver_VS.jl) | Masked residual/retained-row and surface-specific lifecycle |
| [Geometry](src/grid/Cartesian.jl) and [grid topology](src/grid/Grid.jl) | Explicit-coordinate construction and fine IGG semantics |
| [Reduction helpers](src/Utils.jl) | Existing world-communicator norms; new level context must account for ownership/agglomeration |
| [CPU module wiring](src/JustRelax_CPU.jl), [CUDA 3D wiring](src/ext/CUDA/3D.jl), [AMDGPU 3D wiring](src/ext/AMDGPU/3D.jl) | Current standard 3D integration, superseding older availability statements |
| [API rules](.agents/api.md), [grid rules](.agents/grid.md), [solver rules](.agents/solvers.md), [MPI rules](.agents/mpi.md) | Repository conventions to recheck before edits |

External context is limited to primary references. [Jodlbauer et al. (2022)](https://arxiv.org/abs/2205.15770) studies matrix-free monolithic Stokes GMG with appropriately scaled Chebyshev-Jacobi smoothing for Taylor-Hood finite elements. It supports the general architectural direction; its transfer, stability, and performance results do not automatically apply to JR's staggered PH/DR discretization. [Walker and Ni (2011)](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf) supplies the Anderson formulation; safeguard, state, GPU, and MPI choices here are implementation proposals. [ImplicitGlobalGrid's official documentation](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) describes its distributed GPU/CPU lifecycle and transport options; verify the installed compatible version before choosing a multilevel adapter.
