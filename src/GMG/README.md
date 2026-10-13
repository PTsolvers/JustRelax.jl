# GMG hierarchy foundation

This directory implements the first hierarchy foundation from
[`GMG_DR_IMPLEMENTATION_PLAN.md`](../../GMG_DR_IMPLEMENTATION_PLAN.md).
The interfaces are internal and are not yet a callable GMG Stokes solver.
The transfer design must pass its own gates before DR cycles or Anderson are added.
Transfer kernels and level halo exchange exist; the Stage 1 transfer pair is frozen
(see the decision record).

## Implemented contracts

- `GMGOptions` holds direction-wise factors of 1 or 2, minimum global coarse cell
  counts, and a maximum hierarchy depth. All levels remain active; a direction
  stops before global nesting, partition alignment, or two owned local cells
  would be lost. Odd directions remain uncoarsened.
- `GMGDomain` owns explicit global host vertex vectors and physical-domain
  metadata. `GMGPartition` separately records owned global pressure cells,
  overlap storage, local owned indices, periodicity, halo widths, and a borrowed
  communicator. Setup does not query, initialize, finalize, or modify IGG.
- Each `GMGLevel` has a JR `Geometry`, separable cell and face-dual measures,
  and three independent coupled workspaces: correction, RHS, and residual.
  Coordinates, measures, pressure arrays, and velocity arrays use the selected
  backend and Float32/Float64 precision. Physical histories are not allocated or
  advanced by this hierarchy.
- Coarse vertices are selected from the previous global vertices. Coarse
  geometry is built explicitly, preserving scalar precision and avoiding the
  fine-global-size query in `Geometry(ni, li)`. `grid.li` describes local storage;
  `domain.li` retains the physical global extent, including under decomposition.
- Pressure ownership excludes overlap. A normal face belongs to the cell on its
  positive side; the nonperiodic terminal face belongs to the terminal owner.
  Transverse velocity ghosts and duplicate periodic faces are excluded from
  `gmg_owned_velocity`. These are unique storage ownership ranges, not yet the
  BC-reduced set of retained correction unknowns.
- Cell volumes are tensor products of stored directional cell widths. Normal
  face dual volumes replace the normal width with the average of its adjacent
  cell widths. Physical terminal dual widths are half-cell widths; periodic
  neighbors wrap through the explicit global coordinates. Numerical/DR diagonal
  coefficients are not geometric measures.
- `level_sum` and `level_max` reduce already computed, locally owned scalar
  contributions on the level communicator. Passing `nothing` means local scope.
  Caller ownership must form a nonoverlapping complete partition for a global
  norm; setup does not infer that partition from array sizes.
- Invalid local ownership/coordinates are rejected collectively. Global
  coordinates, scalar precision, hierarchy controls, periodicity, and halo widths
  must agree before ranks enter a level-building loop. Coordinates broadcast at
  setup are host metadata; no numerical field is gathered to the CPU.
- `GMGMaterialContext` preserves a concrete GeoParams single material or a
  heterogeneous material tuple, a concrete `args` NamedTuple, and an optional
  JustPIC phase-ratio container. Multiphase material counts must match. The laws
  and argument arrays remain borrowed; this is the input contract for later JR
  constitutive evaluation, not a replacement material evaluator.

The constructors produce a homogeneous `Vector{L}` of concrete levels and a
concrete transition vector. No `Any`, abstract array/level fields, global caches,
or stored `Function` callbacks are introduced. Backend selection uses JR's
existing `PTArray` and `backend` traits. Host coordinate-vector construction is
setup work; GPU fields are allocated with backend constructors and `similar`.

## Transfer unit (serial)

- `GMGTransfer(fine, coarse, transition; velocity, no_slip)` builds padded
  one-dimensional index/weight tables on the host from explicit global
  coordinates, once per level pair, and copies them to the level backend.
  `gmg_transfers(hierarchy)` builds all pairs. Prolongation is the tensor
  product of those rows; restriction is built from the same rows as
  `R = M_c⁻¹ Pᵀ M_f`, so the weighted adjoint identity holds by construction.
- `gmg_prolong!(fine, coarse, transfer; β)` and `gmg_restrict!(coarse, fine,
  transfer)` live in `transfer_kernels.jl`, included through `common.jl`, so
  CPU, CUDA, and AMDGPU modules share one dimension-agnostic
  `@parallel_indices (I...)` gather kernel. `β = 0` overwrites; stale or
  nonfinite values never leak through zero-weight padding.
- Pressure: parent-cell injection and volume-weighted restriction.
- Velocity, normal direction: coincident-face injection and coordinate-linear
  interpolation between coarse faces. Nonperiodic boundary normal faces are
  eliminated (prescribed zero correction) on both levels.
- Velocity, transverse directions: `GMGConservativeVelocity()` copies the parent
  value (flux-conserving reference, `Div_f P_v = I_p Div_c` verified);
  `GMGSmoothVelocity()` is the prototype coordinate-linear interpolation between
  coarse centers, with free-slip extension or no-slip linear decay at physical
  walls. The smooth rule is not flux conserving; its commutator defect is not yet
  measured.
- Rows exist only for owned retained DOFs. Prolongation writes nothing to
  velocity ghosts or halos when `β = 1`; restriction zeroes every non-owned
  coarse entry. Reads that fall outside local storage are rejected at setup.

## Level halo exchange

- `GMGExchange(level)` builds a persistent plan from every rank's explicit owned
  ranges (one `Allgather` at setup); the decomposition must be Cartesian and each
  halo entry must be owned by a direct neighbor, otherwise every rank rejects.
- `exchange_level!(exchange, fields)` exchanges pressure and every velocity
  component direction by direction. Each slab spans the full storage of the other
  directions, so edge and corner halos are forwarded by the later stages. Owned
  entries never change; velocity ghost layers beyond storage stay the caller's.
- Buffers are allocated at setup. Device data is staged through host buffers for
  MPI (no CUDA-aware path yet). A periodic direction with one rank exchanges
  with itself without MPI; `comm = nothing` uses the same local path.
- Prolongation needs valid coarse correction halos and restriction needs valid
  fine residual halos; call `exchange_level!` first. The caller's IGG state is
  never used.

## Transfer decision record (Stage 1)

Frozen pair: parent-cell pressure injection with volume-weighted restriction,
and `GMGConservativeVelocity()` (normal-direction linear, transverse parent copy)
with its weighted-transpose restriction. `GMGSmoothVelocity()` remains available
as the prototype reference only.

Evidence, from `test_gmg_two_grid.jl` and the larger exploratory runs it was
condensed from (2D, free slip, constant viscosity, exact coarse solve, 2+2
sweeps, rediscretized coarse operators):

| Screen | Conservative | Smooth |
|---|---|---|
| Velocity block, Jacobi, uniform, n = 8/16/32 | ρ = 0.173/0.190/0.195 | ρ = 0.192/0.211/0.217 |
| Normalized `‖Div_f P_v − I_p Div_c‖/‖Div_c‖`, n = 8/16/32 | 0 (round-off) | 0.89/0.97/0.99 |
| Coupled Stokes, DR/Uzawa sweep, γ = 0, n = 8/16/32 | ρ = 0.414/0.435/0.442 | same to 3 digits |
| Same, γ = 10 | ρ = 0.870/0.881/0.883 | same to 3 digits |
| Smoother alone (4 sweeps), γ = 0, n = 8/16/32 | 0.944/0.986/0.997 | — |

- Both rules give h-independent two-grid factors; neither is better on the
  screens, so the cheaper rule that commutes with divergence is frozen. With it,
  the grad-div part of the coarse operator is exactly Galerkin-consistent
  (`R Dᵀ D P = D_cᵀ D_c` up to measures).
- Degradation with γ is identical for both rules and is a property of the point
  smoother on the grad-div near-kernel, not of the transfers. Large augmentation
  needs a smoother that handles that kernel, or a small `γ_eff`; this is a Stage 2
  smoother question.
- On stretched grids ρ rises with resolution (0.33 to 0.56 for n = 8 to 64) and
  is identical with a Galerkin coarse operator. Uniform cells with aspect 2.3
  give ρ ≈ 0.63 under full coarsening and 0.07 under semicoarsening of the short
  direction; aspect 4 gives 0.85 and 0.065. The growth is the local cell aspect,
  not a transfer defect: coarsening policy must follow cell aspect (Section 4.2).
- Not screened: 3D, no-slip walls, variable viscosity, compressible pressure
  rows, periodic seams, the actual DR recurrence, and Float32.

## Verification of this implementation unit

| Check | Actual result |
|---|---|
| CPU hierarchy/type tests | 469 assertions; 2D/3D, Float32/Float64, all factor tuples, rectangular/nonuniform coordinates, periodic ownership, GeoParams input types, invalid setup |
| CUDA hierarchy/type tests | Same 469 assertions on an NVIDIA GeForce RTX 5070 Ti, with scalar indexing disabled |
| CPU level-context tests | 34 assertions on one rank; 36 per rank on two and four ranks |
| CUDA + MPI level-context tests | 36 assertions per rank on two ranks sharing that GPU; setup and scalar reductions only |
| Existing focused CPU regressions | 265 assertions across type constructors, backend traits, conversions, and JustPIC interfaces |
| Inference | `@inferred` on concrete constructors/helpers; inspected optimized bodies for 40 CPU/MPI construction specializations with no call returning `Any` |
| Formatting | Runic on the six new Julia files; whitespace check on the implementation diff |
| AMDGPU | Not run; the package/hardware environment is unavailable in this session |
| Transfers, CPU | `test_gmg_transfers.jl`: 518 assertions; 2D/3D, Float32/Float64, every tested direction tuple, stretched coordinates, periodic x, no-slip/free-slip walls, both velocity rules; weighted adjointness on random fields, constant gauge, integrated continuity, normal affine reproduction, divergence commuting (conservative rule), assembled 2D `P`/`R` oracle with retained DOF counts, `@inferred` setup and launches |
| Transfers, CUDA | Same 518 assertions on the RTX 5070 Ti with scalar indexing disabled |
| Exchange + distributed transfers, CPU MPI | `test_gmg_exchange_MPI.jl`: 68/136/312 assertions on 1/2/4 ranks; 2D/3D, strip and 2×2-type decompositions, periodic x, edge/corner halos on both levels, serial periodic self-exchange, smooth-rule prolongation/restriction matching the serial owned action to 1e-13 |
| Exchange + distributed transfers, CUDA MPI | Same file on 1/2/4 ranks sharing one RTX 5070 Ti (104/208/312 assertions), host-staged MPI |
| Two-grid screen | `test_gmg_two_grid.jl`: 15 assertions on CPU and on CUDA |
| AMDGPU | Out of scope for now by decision; not run |

The MPI fixtures use explicit strip ownership, check owned volume/face counts,
reject inconsistent coordinates and loop controls collectively, and verify a
separate communicator reduction. They also run with the caller's fine IGG grid
active and verify that its geometry and next fine halo update are unchanged.
The CUDA + MPI run is not a multi-device or coarse GPU-halo transport proof.

The full CPU runner was not run: this workspace has no `test/Project.toml`, and
the active environment lacks `ParallelTestRunner`/`Suppressor`. Focused tests
above run in the existing root environment without changing its dependencies or
committing a manifest. Full applicable regression jobs remain a later gate.

## Stage 2: matrix-free operator, DR/PH smoother, V-cycle

### Operator and smoother (M3)

- `GMGLevelOperator(level; no_slip)` (`operator.jl`) evaluates, matrix free and
  dimension generic, the homogeneous augmented error operator
  `K_v e = -∇·(2η ε_dev) + ∇Π`, `K_p e = ∇·v + C p`, `Π = p - γ(∇·v + C p)`,
  with JR's `ε_dev = ε - ∇·v/3` and the product form `∇(γ F_p)` for a center
  field `γ`. Homogeneous BCs: zero normal correction, free-slip mirror or
  no-slip negated tangential ghosts. `gmg_prepare!` applies BCs then exchanges.
- Coefficients (`GMGCoefficients`): center `η`, edge `η` (harmonic from
  centers), compliance, `γ`, and the PH pressure-step scale, all center fields.
  `gmg_restrict_coefficients!`: harmonic volume average for `η`, volume average
  for the rest (exact Galerkin for compliance and, with the frozen transfer,
  for grad-div).
- `gmg_refresh_bounds!` probes with 3^N global colors per component: exact
  velocity-block diagonal and the exact Gershgorin bound of `D⁻¹A`, reduced on
  the level communicator. Periodic directions need cell counts divisible by 3.
- `gmg_dr_smooth!`: `substeps` Chebyshev velocity steps on
  `[λ/spectral_range, λ]` with pressure fixed, then
  `p += pressure_step · pressure_scale · (rhs_p - K_p e)`. Only owned retained
  entries move; halos come from the exchange.

### Coupled V-cycle (M4)

`GMGMultigrid`, `gmg_vcycle!`, `gmg_coarse_solve!`, `gmg_solve!` with
`GMGStatus` and `GMGCounters` (outer iterations, cycles, sweeps, coarse sweeps,
operator applications, and `work` in finest-level-equivalent applications).

- Fixed: intermediate-level corrections were not reset to zero before
  pre-smoothing. This caused the earlier depth dependence. Constant η, γ = η,
  V(2,2): factor 0.054 at 3/4/5 levels; V(1,1): 0.23. Gate passes.
- Still open: the coupled DR/PH sweep is unstable under viscosity contrast
  (frozen two-grid ρ > 1 with an exact coarse solve, table below). Not pursued.

| Field, contrast | γ = η, step = η | γ, step = mean η (production-like) | γ = 0, step = η, ω = 0.5 |
|---|---|---|---|
| smooth, 1e2 | 1.38 / 0.60 / 0.45 | 0.68 / 0.95 / 0.84 | 0.58 / 0.56 / 0.50 |
| smooth, 1e4 | 57 / 27 / 4.8 | 12 / 36 / 9.1 | 2.9 / 29 / 2.0 |
| inclusion, 1e2 | 0.50 / 5.4 / 5.5 | 0.35 / 1.3 / 2.5 | 0.63 / 0.68 / 0.86 |
| inclusion, 1e4 | 0.50 / 553 / 561 | 0.35 / 113 / 236 | 0.64 / 22 / 74 |

### Decision (2026-10-13): option B, velocity-block GMG inside an outer PH loop

The coupled smoother is set aside. Multigrid acts on the velocity block
`K_vv = K_v(·, p = 0)` (viscous + grad-div); pressure is updated by an outer
PH (inexact augmented Uzawa) iteration. `velocity_cycle.jl`:

- `gmg_velocity_smooth!` (Chebyshev velocity steps, split out of
  `gmg_dr_smooth!`), `gmg_velocity_residual!`, `gmg_velocity_coarse_solve!`,
  `gmg_velocity_vcycle!`, `gmg_velocity_solve!` (stationary V-cycles).
- Energy-minimizing step on each prolongated correction
  (`GMGMultigrid(...; line_search = true)`), in the face-measure inner product
  `GMGLevelOperator.velocity_weight` (the velocity block is symmetric in it).
  Removes the V-cycle divergence under contrast (inclusion 1e4: ρ 13.9 to ~0.6).
- `gmg_velocity_fcg!`: flexible CG (Notay 2000) preconditioned by one V-cycle
  per iteration; scratch in `GMGMultigrid.krylov`.
- `gmg_ph_solve!`: `δv ≈ K_vv⁻¹ r_v` by FCG to `velocity_rtol` (at most
  `velocity_cycles`), `v += δv`, then `p += pressure_step · pressure_scale ·
  r_p` with `pressure_scale = η + γ`. Stops on `rtol`, budget, `patience`
  iterations without a new residual minimum, or nonfinite.

### Baseline: production DYREL vs GMG-PH (2D sinking block, CPU, 4 threads)

Same discrete problem: GMG edge viscosity copied from JR `viscosity.ηv`;
free slip; square block, contrast as listed. Production `solve_DYREL!` with
`ϵ = 1e-8`; GMG-PH stops at equal or better physical residual (plain Stokes
operator, momentum RMS / forcing RMS, continuity RMS / (max|v|/L)). Work: DR
iterations vs GMG finest-equivalent operator applications. GMG-PH solutions
match DYREL to `|ΔVy|/|Vy|` 5e-7 to 4e-5. Script:
`test/gmg_baseline_dyrel.jl` (not a `test_*` file; run manually).

| n, contrast | DYREL iters (time) | Best GMG-PH work (time), setting |
|---|---|---|
| 64, 1e2 | 27 650 (6.9 s, incl. JIT) | 3 128 (0.36 s), γ = η, vtol 0.1 |
| 128, 1e2 | 42 300 (14.1 s) | 5 824 (1.0 s), γ = η, vtol 0.1 |
| 64, 1e4 | 79 050 (13.3 s) | 210 738 (131 s), γ = 5η, vtol 0.01 |
| 128, 1e4 | 51 800 (17.1 s) | 150 116 (103 s), γ = 5η, vtol 0.01 |

Contrast 1e2: GMG-PH needs 7–9× less work than DYREL. Contrast 1e4: 2.9–3×
more work; the velocity FCG needs ~20 V-cycles per outer iteration. γ = 20η
slows the velocity solve more than it speeds the outer loop.

### Evidence

| Check | Result |
|---|---|
| `test_gmg_operator.jl` (CPU, CUDA) | 54/54 |
| `test_gmg_two_grid.jl` (CPU, CUDA) | 15/15 |
| `test_gmg_transfers.jl` (CUDA) | 518/518 |
| `test_gmg_vcycle.jl` (CPU, CUDA) | 20 pass, 1 `@test_broken` (coupled contrast) |
| `test_gmg_velocity.jl` (CPU, CUDA) | 50/50: weighted symmetry 2D/3D; V-cycle h-independent (ρ < 0.15, both line-search modes, 3/4/5 levels); FCG to 1e-8 at contrast 1e2/1e4, 2D/3D; outer PH converges (contrast 1/1e2/1e4 2D, 1e2 3D with step 0.5); inference |
| `test_gmg_operator_MPI.jl` | Distributed `gmg_velocity_dot` check added; **not yet run** after the operator struct change (`velocity_weight` field) |

### Open gates and next steps

1. **Run MPI** (`test_gmg_operator_MPI.jl`, `test_gmg_exchange_MPI.jl`) on
   CPU/CUDA, 1/2/4 ranks; add a distributed velocity V-cycle/FCG/PH check.
2. **Contrast 1e4 velocity solve.** Rediscretized coarse operators with
   harmonic η are the weak point. Candidates: Galerkin (or
   coefficient-aware) coarse velocity operator, stronger smoothing near jumps
   (block/line), more pre/post sweeps on coarse levels.
3. **3D pressure step.** Unit step is near the stability limit under contrast
   (1.5 diverges; 0.5 gives ρ ≈ 0.5). Need a Schur-bound step or outer
   acceleration (Krylov on pressure, or plan M8 Anderson on the PH map).
4. γ = 0 outer PH diverges with step η; production always augments, so
   only augmented settings are supported.
5. Runic not yet applied to this unit's changes.
6. Then: fine JR/IGG layout adapter (wire into `solve_DYREL!`-style entry),
   M5 (GeoParams/nonlinear lifecycle), M6–M8, CUDA-aware MPI, multi-GPU.
