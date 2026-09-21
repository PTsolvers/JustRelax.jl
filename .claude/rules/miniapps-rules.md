---
paths:
  - miniapps/**/*.jl
---

# Miniapp Rules

Miniapps are runnable models and benchmarks, and the source of the documentation's examples. Running and validating them: the `miniapps-and-benchmarks` skill; writing a new one: the `new-miniapp` skill.

- All miniapps share the environment `miniapps/Project.toml`: `julia --project=miniapps --startup-file=no <script>`. Its `[sources]` entry points `JustRelax` at an absolute path on the author's machine. To run against the local checkout use `Pkg.develop(path=".")`, and do not commit the resulting `Project.toml` change.
- Follow the closest existing script: `backend` / `backend_JP` constants with the alternatives shown, `@init_parallel_stencil` for that backend, a `main(igg; nx, ny, …)` function, then `nx`/`ny` and the `IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)` guard at the bottom. Keep the default resolution small enough for a smoke test.
- Figures, VTK and checkpoints go to a script-created directory (`figdir`, `figs*`). Never commit output.
- Import what the script uses. `using JustRelax, JustRelax.JustRelax2D` (or `3D`) already brings in ImplicitGlobalGrid and JustPIC names.
- When a public signature, keyword or solver changes, update every affected miniapp (PR checklist): `grep -rl <name> miniapps/`.
- Do not hard-code machine-specific paths.

## Literate-sourced miniapps

`benchmarks/thermal_diffusion/diffusion/diffusion2D_periodic.jl`, `benchmarks/stokes2D/shear_band/ShearBand2D.jl` and `benchmarks/stokes2D/Blankenbach2D/Benchmark2D_sgd.jl` generate documentation pages through `docs/make.jl`; editing them changes the page. In them:

- Open with a `#` title and a short description of what the model does and how to run it.
- A single `#` comment becomes rendered markdown (narrative). Use `##` for a comment that must stay inside the code block.
- Let the code speak: keep narrative concise, and put a heading before each stage (`# ### Model domain`, `# ### Material properties`, …).
- Interleave a figure with setup only when it explains the geometry, initial condition or rheology.
