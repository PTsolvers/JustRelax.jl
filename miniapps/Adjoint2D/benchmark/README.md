# Adjoint benchmarks

Run the commands below from the **JustRelax repository root**. Use the JustRelax environment
with `--project=.`; the plotting and selected GPU packages must be available in your Julia
environment stack. These commands do not require adding CUDA to JustRelax's `Project.toml`.
The `miniapps` environment can also be used when its dependencies are installed.

## Select a backend

Set the variable at the top of [helpers.jl](helpers.jl):

```julia
const isCUDA = true   # CUDA GPU
# const isCUDA = false  # CPU
```

This selects the matching `backend` and `backend_JP` and initializes ParallelStencil.
The default is CPU. Start a fresh Julia process after changing the switch.
The helper also transfers Taylor-test perturbations to the selected backend.
Plotting and returned diagnostic fields copy data to the CPU.

Standalone miniapps outside this folder have their own `isCUDA` switch.

## Run the Taylor tests

The standard and variational suites check viscosity-field gradients for a viscous sinking
block, a viscoelastic sinking block, and a shear band. They also check the shear-band material
parameters `G`, `C`, `ϕ`, `Ψ`, and `η_vp`.

```sh
# Standard DYREL adjoint
julia --project=. --startup-file=no miniapps/Adjoint2D/benchmark/TaylorTests2D.jl

# Variational DYREL adjoint
julia --project=. --startup-file=no miniapps/Adjoint2D/benchmark/TaylorTests2D_variational.jl
```

Both suites use a 16 × 16 grid by default, with three viscoelastic steps and ten shear-band
steps. Each perturbation acts on the final time step. They report fitted slopes for the
first-order Taylor remainder; a correct gradient should give a slope near two over a suitable
perturbation range, before solver tolerance and round-off dominate.

Plots are written relative to the working directory:

- Standard: `figures/Adjoint2D/TaylorTests/taylor_tests.png`
- Variational: `figures/Adjoint2D/TaylorTests/taylor_tests_variational.png`

## Run individual cases

Use the same backend selection with any case:

```sh
julia --project=. --startup-file=no miniapps/Adjoint2D/benchmark/SinkingBlock2D_adjoint.jl
julia --project=. --startup-file=no miniapps/Adjoint2D/benchmark/SinkingBlock2D_VE_adjoint.jl
julia --project=. --startup-file=no miniapps/Adjoint2D/benchmark/ShearBand2D_adjoint_materials.jl
```

These scripts run their forward and adjoint solves and save figures in
`SinkingBlock2D_adjoint/`, `SinkingBlock2D_VE_adjoint/`, and
`ShearBand2D_adjoint_materials/`, respectively.

## Customize a run

Set `NO_AUTORUN = true` before including a script, then call its main function with keywords.
For example, start a CPU session from the repository root:

```sh
julia --project=. --startup-file=no
```

Then run:

```julia
const NO_AUTORUN = true
include("miniapps/Adjoint2D/benchmark/TaylorTests2D.jl")

n = 32
igg = IGG(init_global_grid(n, n, 1; init_MPI = !JustRelax.MPI.Initialized())...)
results = main(
    igg;
    n,
    ve_steps = 3,
    shear_steps = 10,
    ε = 10.0 .^ (-2:-1:-5),
    material_parameters = (:G, :C),
    figdir = "figures/custom_taylor_tests",
)
finalize_global_grid(; finalize_MPI = false)
```

Use the same pattern with `TaylorTests2D_variational.jl` in a fresh session to customize the
variational suite. Initialize the global grid with the same `n` passed to `main`.

## Check backend helpers

The backend-allocation and device-transfer checks can be run without a full benchmark:

```sh
julia --project=. --startup-file=no test/test_adjoint_miniapp_helpers.jl
```
