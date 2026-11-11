# Adjoint convergence investigation

Measured on 2026-11-09 with the workspace JustRelax revision `16e1615a`, CPU,
one Julia thread and one MPI rank. These are diagnostic experiments, not production
solver changes. The user's example environment pins a separate Git revision; the
reproductions explicitly load the workspace package.

## Main finding

The global numerical pressure penalty makes the diagonally preconditioned velocity
iteration very stiff in low-viscosity regions. It helps the pressure iteration, but
penalizes the velocity diagonal even for nearly divergence-free motions that do not
experience the pressure penalty itself. A localized velocity objective can excite
those slow motions much more strongly than the forward forcing does.

Schematically, the inner operator is

`D⁻¹ (K_tangent + Bᵀ Γ B)`.

For a divergence-free velocity `v`, `B v ≈ 0`, but `D` still contains the large
penalty contribution. Its effective stiffness is therefore approximately proportional
to the weak viscosity divided by the penalty. Scalar time-step and damping adjustments
do not remove this disparity.

In the fully clipped 8 × 8 inclusion example, the symmetrically scaled inner matrix is
symmetric to relative error `8.4e-17`. Five eigenvalues are `4.0e-6–8.3e-6`, followed
by a gap to `0.0253`. The associated motions are confined to the weak inclusion and
almost divergence-free. The adjoint solution is dominated by that subspace, while the
forward correction to the imposed pure shear has negligible projection onto it.
This is a measured mechanism in that example, not an assertion that every nonlinear
adjoint operator is symmetric or has the same modes.

## Provided subduction model

Source: `/Users/cschule1/Documents/work/JustRelaxAdjointExamples/SubductionShearZone/DoraMairaVerticalWeakZone.jl`.

The objective is the UHP body's phase-weighted mean vertical velocity. The current
file uses smooth constitutive bounds `1e16–1e22` with no hard cutoff, a uniform grid,
`γfact = 20`, tolerance `1e-3`, `nout = 10`, and `rel_drop = 0.5`.

The diagnostic reproduces its first step on a 32 × 64 grid with deterministic
particles. It reads the model without changing its file, disables plotting/output
and autorun, and compares fresh adjoints on one frozen converged forward state.
It also tests the previously reported hard-cutoff configuration by removing the
smooth bounds and imposing `1e18–1e23`. These are different constitutive models;
comparisons between solver methods are made **within each frozen state**.

| Configuration | Forward iterations | Baseline adjoint | Local penalty | Divergence-free patch correction |
|---|---:|---:|---:|---:|
| Hard cutoff `1e18–1e23` | 10,890 | 14,180 | 710 | 3,490 |
| Current smooth bounds `1e16–1e22` | 23,970 | 31,430 | 1,000 | 11,440 |

All adjoint columns use the same production stopping criterion and tolerance `1e-3`.
The hard case has 92 cells at the lower cutoff and 579 at the upper cutoff, out of
2,048 cells. Its inherited penalty is `6.83e22`, even where viscosity is `1e18`.
The smooth case has no active hard cutoffs, a viscosity range approximately
`1.00e16–1.00e22`, and an inherited penalty of `2.76e22`.

The local-penalty experiment replaces the numerical penalty by `γfact * η_i / 2`
for these incompressible, full-rock states, then rebuilds the velocity diagonal and
upper bounds. It uses the converged viscosity, keeps the physical bulk viscosity
unchanged, and retains the production tangent amplification and damping adaptation.
For this model the ratio `Γ_i / η_i` becomes 10 throughout the domain, instead of
reaching about 68,000 in the hard case or 2.75 million in the smooth case.

The patch experiment keeps the global penalty. It constructs four-face,
divergence-free velocity corrections where the four adjoining cells have viscosity
below 1% of the final domain mean. The small projected operator is assembled by
matrix-free adjoint applications, factored once, and applied every 50 iterations.
There are 794 and 785 basis vectors in the hard and smooth runs respectively.
The full Stokes matrix is never assembled for these model experiments, but the
projected matrix and basis are dense, so this prototype is not a scalable implementation.

Against a tighter `1e-6` adjoint solve on the same frozen state, relative
viscosity-field gradient differences were:

| Configuration | Baseline | Patch correction | Local penalty |
|---|---:|---:|---:|
| Hard cutoff | `5.80e-4` | `4.25e-6` | `1.01e-5` |
| Smooth bounds | `8.43e-4` | `2.38e-6` | `1.61e-4` |

These comparisons check solver agreement; they are not end-to-end finite differences
of the nonlinear forward problem. Equal residual tolerances do not imply equal
solution errors in an ill-conditioned system.

For the hard-cutoff model, the local-penalty run also evaluated the actual density,
creep-prefactor `A`, and exponent `n` material-gradient outputs. Against the tighter
reference, the largest phase-wise relative L2 differences were `4.04e-5` for density
and `1.60e-4` for `A`; the returned `n` fields were identical. This is an agreement
check of the returned gradients, not an independent validation of parameter derivatives.

## Additional controlled tests

Synthetic cases use the same frozen state and `1e-7` adjoint tolerance for each method.
“Nonlinear” means that viscosity derivatives are active outside the clipped inclusion.

| Case | Baseline | Local penalty | Local divergence-free correction |
|---|---:|---:|---:|
| Fully clipped, 12 × 12 | 7,500 | 1,500 | 950 |
| Nonlinear, 8 × 8 | 6,800 | 2,850 | 3,400 |
| Nonlinear, 16 × 16 | 10,350 | 2,700 | 3,150 |
| Pressure-difference objective, fully clipped 8 × 8 | 650 | 1,050 | 650 |

The pressure objective is the mean pressure in the weak inclusion minus the mean
outside, so it is invariant to the incompressible pressure gauge. It demonstrates
that local penalty scaling is not uniformly better, even though it did converge.
Enclosed compressible weak bodies and volumetric forcing were not tested here;
the production code's global penalty was deliberately chosen to handle such pressure modes.

Other attempts did not resolve the mechanism: a weighted Rayleigh estimate changed
8,550 iterations to 8,700; fixing damping from the smallest measured eigenvalue
increased it to 30,000. Refreshing the final pressure residual left the count at
8,550. An inner-tolerance floor helped one linear case but worsened a nonlinear one.
Changing only the model's check interval from 10 to 50 changed 14,180 to 14,200 iterations.

## Options

1. **For this shear-zone model, introduce opt-in adjoint-local penalty scaling.**
   Build it from the final viscosity and rebuild its diagonal and upper bounds together.
   Preserve the global option and the physical bulk-compliance treatment. This is the
   smallest promising production change, with 20–31 times fewer adjoint iterations in
   the reduced model tests. Do not assume that improvement transfers to pressure-driven
   or enclosed compressible models.
2. **Add a divergence-free patch/coarse correction while retaining the global penalty.**
   This addresses the slow modes without weakening pressure relaxation. The reference
   experiments support the direction, but a production version needs sparse local
   solves or a multigrid implementation, backend-aware storage, and MPI halo/reduction
   handling. The dense diagnostic should not be copied into large production runs.
3. **For a general solver, use a viscosity-robust multilevel velocity preconditioner.**
   It must treat divergence-free modes appropriately; merely adding generic multigrid
   is not enough. This is the same augmented-Lagrangian difficulty studied by
   [Shih, Stadler and Wechsung](https://arxiv.org/abs/2107.00820), including nonlinear
   tangent viscosities. Their discretization differs from JustRelax, so adaptation
   and validation are required.

No production defaults, physical viscosity laws, or user example files were changed.
Full-resolution evolving models, stretched-grid runs, accelerator backends and
multiple MPI ranks remain untested. Iteration reductions are not wall-time speedup
claims: the patch method has extra applications and setup, and recorded timings are
single samples with Julia compilation and concurrent diagnostic processes.

## Reproduction

From the JustRelax repository, using the root project:

```sh
JUSTRELAX_MODE_METHODS=tight_reference,baseline,local_penalty,curl_patch \
  julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/variational_slow_modes.jl 16 nonlinear

julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/subduction_shearzone_diagnostic.jl 32
julia --project=. --startup-file=no -t 1 scratch/adjoint_solver_experiments/subduction_shearzone_diagnostic.jl 32 smooth
```

`JUSTRELAX_MODEL` can override the example script path. Optional
`JUSTRELAX_MODEL_SAVE_STATE` / `JUSTRELAX_MODEL_LOAD_STATE` paths save/reload the
frozen state for comparisons; MPI is initialized separately when reloading. Use a
cache only with the same model configuration and package revision.
`JUSTRELAX_MODEL_MATERIAL_GRADIENTS=1` also compares the phase-wise density and
dislocation-creep prefactor/exponent gradients with the tighter reference.
