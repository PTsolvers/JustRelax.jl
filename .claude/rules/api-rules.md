---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# Public API Rules

Preserve the public API unless the task explicitly changes it. Deeper reference: `.agents/api.md`.

- **Deprecate before removing.** Do not remove or rename an exported name or a keyword argument without a deprecation cycle (`Base.@deprecate`, placed next to the new definition). Tag PRs `[BUGFIX]`, `[ADDITION]` or `[DOC]`.
- **Exports** are declared in `src/common.jl` (or the root module for cross-dimensional types) so every module variant picks them up.
- **Three layers.** Every public function is: public entry point → `backend(x)` trait → CPU branch in shared code → GPU branch in `src/ext/{CUDA,AMDGPU}/{2D,3D}.jl` that forwards to the same `_impl`. A new public function needs all layers plus the AMDGPU mirror. The implementation lives in `src/common.jl` or a file it includes, never in a per-backend file.
- **Solver keywords.** `solve!` and `heatdiffusion_PT!` take a *required* `; kwargs` NamedTuple. `solve_VariationalStokes!`, `solve_DYREL!` and `solve_VariationalDYREL!` take `; kwargs...` and normalize through `flatten_solver_kwargs`, so plain keywords and a bundled `kwargs = (; …)` both work. New solver entry points use the second form.
- **Constructors** take the backend tag first and allocate on it: `StokesArrays(backend, ni)`, `ThermalArrays(backend, ni)`, where `ni` is the number of cells.
- **Annotations.** Annotate only as narrowly as the implementation needs. `Matrix{Float64}` locks out GPU arrays, views and `Float32`; use `AbstractArray` or a parametric type. A function that truly needs 1-based indexing declares `Base.require_one_based_indexing`.
- **Documentation follows the export.** Every exported symbol gets a docstring ([docstring-rules](docstring-rules.md)) and appears in the matching `docs/src/man/api/*.md` page ([docs-rules](docs-rules.md)).
