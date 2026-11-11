# DYREL adjoint solver: next steps

The priority is efficient pseudo-transient adjoint convergence for the subduction shear-zone
model with large viscosity contrasts. The next steps follow the fixed-local and adaptive
penalty experiments in `scratch/adjoint_solver_experiments/`.

Production code, permanent tests and user documentation stay in `src/`, `test/` and
`docs/src/man/`. The [experiment-folder guide](../scratch/adjoint_solver_experiments/README.md)
lists the exact production files; the scratch folder contains only temporary investigation
material and is not needed by the solver.

## 1. Add an opt-in local adjoint penalty — implemented

Implemented `adjoint_kwargs = (; penalty_scaling = :local, penalty_factor = 20.0)` in both
2D DYREL adjoints. Keep `:global` as the default, inheriting the forward numerical penalty.
The local option uses the converged cell viscosity, preserves physical bulk viscosity and
rock fractions, and rebuilds the diagonal, Gershgorin bounds and pseudo-time steps before
applying the existing nonlinear tangent amplification. The forward state stays unchanged.

Validation: compare gradients and forward-state preservation for velocity and pressure
objectives; check finite bulk modulus, eliminated cells, periodic boundaries and free-surface
stabilization; reproduce the frozen subduction experiments through the production solver.

## 2. Validate the evolving 64 × 128 subduction model

Use `/Users/cschule1/Documents/work/JustRelaxAdjointExamples/SubductionShearZone/`
`DoraMairaVerticalWeakZone.jl`, loading this workspace version of JustRelax.
Run the model over several time steps with identical forward settings and adjoint
objectives. Compare the existing global option and fixed-local factor 20. Record outer
pressure passes, total velocity iterations, wall time excluding compilation, final residuals,
and gradients. Include both the current smooth rheology and the hard 1e18–1e23 cutoff case.
Check warm starts as the viscosity and active cutoff regions evolve.

Accept the option for this model only if the gain persists and gradients agree with tighter
reference solves. The existing frozen 32 × 64 results do not establish this yet.

## 3. Decide whether adaptive penalty is worth promoting

Keep adaptation experimental until step 2 reveals a need. The current controller rescues
poorly chosen starting factors but adds little to factor 20 in the frozen shear-zone cases.
If needed, adjust the factor only between outer pressure passes: increase it for stalled
pressure contraction and decrease it for repeatedly expensive velocity relaxation. Retain
bounds, a cooldown, rollback on residual growth, and rebuild/reset iteration parameters
when the factor changes. Compare total work against the best fixed factor.

## 4. Broaden validation before changing defaults

Test enclosed weak regions, pressure objectives, compressibility, free surfaces and multiple
contrast geometries. Small local penalties can slow pressure convergence in enclosed weak
regions. Run CUDA and AMDGPU checks on hardware; MPI adjoints remain limited to one rank.
Do not change the global default based only on the shear-zone benchmarks.

## Step 1 validation (2026-11-10)

- CPU focused checks passed: 232 gradient/state checks and 291 kernel/parameter checks.
- The production variational adjoint reproduced 710 iterations for the frozen hard-cutoff
  state and 1,000 for the smooth state, both on 32 × 64 cells at tolerance 1e-3.
  At tolerance 1e-6 these required 1,340 and 2,080 iterations, respectively.
- Material-gradient maximum differences, normalized by each reference field's maximum:

  | State | Density | Creep A | Creep n |
  |---|---:|---:|---:|
  | Hard cutoff | 1.99e-5 | 3.01e-5 | 0 |
  | Smooth rheology | 4.36e-4 | 1.03e-4 | 1.00e-4 |

  These compare the 1e-3 production solve against its 1e-6 reference. Forward viscosity,
  bulk viscosity and sampled forward numerical fields remained unchanged. The gradient
  tests additionally check every forward DYREL field against the default solve.
- Runic checks passed for changed solver implementations, tests and the production diagnostic;
  `git diff --check` passed. The public solver docstring was updated without retaining
  unrelated formatter changes elsewhere in that file.
- Documentation build could not start: Documenter is not installed in the docs environment.
- CUDA/AMDGPU execution and the evolving model in step 2 have not been run.

### Broader CPU regression run

All 60 regular test files completed: 56 passed in the parallel run, including all three
adjoint suites, both periodic DYREL suites, variational DYREL, and the nonlinear cap tests.
The WENO test initially lacked its test-only SpecialFunctions dependency and passed when
rerun with that dependency available. The repository has no `test/Project.toml`; tests used
the root environment plus a temporary environment for installed test-only dependencies.
No repository dependency files were changed.

The full suite is not clean. Separate reruns confirmed these failures outside the new
local-penalty path:

- `test_dyrel.jl`: undefined `JR3` in the existing 3D adjoint allocator test.
- `test_thermalstresses.jl`: JustPIC expects a 34 × 34 grid field but receives 32 × 32.
- `test_shearheating3D.jl`: JustPIC expects a 34 × 34 × 34 grid field but receives
  35 × 35 × 35 during particle-to-grid transfer.

All nine MPI test files passed with two ranks: I/O, 2D/3D multiphase diffusion, DYREL
Taylor–Green, 2D/3D lithostatic pressure, periodic boundary conditions, and 2D/3D shear bands.
These are forward/distributed regression checks; they do not establish multi-rank adjoint
support. The adjoint remains restricted to a single rank.
