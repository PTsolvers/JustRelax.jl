---
name: add-feature
description: Checklist for adding a solver, public function, kernel, boundary condition or material behavior to JustRelax.jl without breaking the 2D/3D, CPU/GPU and MPI module structure. Use when adding or wiring new functionality.
---

# Add a feature

Read the deep reference for the area first: `.agents/solvers.md` (iteration loops), `.agents/api.md` (dispatch, exports), `.agents/grid.md` (field layout), `.agents/mpi.md` (halos, reductions), plus the "Adding …" sections of `docs/src/man/developer.md`. The path-scoped rules in `.claude/rules/` load automatically for the files you touch.

Most model rheology needs **no** JustRelax change: define phases with GeoParams `SetMaterialParams` in the miniapp and pass `rheology` and `PhaseRatios` to `compute_viscosity!`, `compute_ρg!` and the solver. Add code under `src/rheology/` only for a new derived field or solver-side evaluation.

## Checklist

1. **Pick the home.** Start from the closest existing family: `src/stokes/`, `src/variational_stokes/`, `src/DYREL/`, `src/thermal_diffusion/`, `src/rheology/` or `src/boundaryconditions/`. Code independent of both dimension and device goes in `src/common.jl` or a file it includes. Keep 2D and 3D entry points separate.
2. **Write the kernels** to [kernel-rules](../../rules/kernel-rules.md): `@parallel` / `@parallel_indices (I...)`, `@idx` launches, `@dxi`-style spacing access, MiniKernels helpers, GPU-safe, no scalar indexing. Copy the nearest existing kernel.
3. **Wire the public entry point** in three layers ([api-rules](../../rules/api-rules.md)): public function → `backend(x)` trait → CPU method in shared code → CUDA and AMDGPU methods in `src/ext/{CUDA,AMDGPU}/{2D,3D}.jl` forwarding to the same `_impl`. Prefer the `kwargs...` + `flatten_solver_kwargs` convention for solver entry points.
4. **Update all six module headers** ([backend-rules](../../rules/backend-rules.md)). If shared code uses a name from the root `JustRelax` module or from JustPIC, add it to the `import` blocks of CPU 2D, CPU 3D, CUDA 2D/3D and AMDGPU 2D/3D. Check with `grep -n <name> src/JustRelax_CPU.jl src/ext/*/*.jl`. A miss is a GPU-only `UndefVarError`.
5. **Include the files.** A new source file is `include`d from `src/common.jl` (or from the solver file that needs it) and from every GPU module that must provide the feature. Verify against each module's `include` lines.
6. **Export** in `src/common.jl` (or the root module for cross-dimensional types).
7. **Document.** Docstring per [docstring-rules](../../rules/docstring-rules.md); add the new source file to the `Pages` of the matching `docs/src/man/api/*.md` page; update the relevant manual page. See the `build-docs` skill.
8. **Test** ([testing-rules](../../rules/testing-rules.md)). A focused `test/test_<name>.jl` (auto-collected). Tiny grids; 3D GPU-capable tests keep the launched range ≤ 256 cells. If the test is CPU-only, add it to the GPU exclusion list in `test/runtests.jl`; if it is light, add it to `test_worker`. A halo change needs an MPI test that also passes on one rank.
9. **Miniapp.** Update affected miniapps; add one for a new solver or physics ([`new-miniapp`](../new-miniapp/SKILL.md)). A convergence-affecting change needs a benchmark run, not just unit tests ([`miniapps-and-benchmarks`](../miniapps-and-benchmarks/SKILL.md)).
10. **Format and spelling.** `git runic --inplace` on the changed Julia files only; CI also runs `typos`.
11. **Verify** in this order: the smallest relevant test → the full CPU suite for solver, type, dispatch or kernel changes ([`running-tests`](../running-tests/SKILL.md)) → docs build for public-API or docs changes → `git diff --check` and `git status` (no unrelated files).
12. **Report** exactly which backends and rank counts you ran. CPU-only validation does not establish accelerator or MPI correctness.

## Conventions

- Files: follow the directory's existing naming. Types `PascalCase`, functions `snake_case`, mutating functions end with `!`, and `_impl` for the implementation behind a public function ([style-rules](../../rules/style-rules.md)).
- Do not break the public API; deprecate first.
- Backend-generic code allocates through the backend tag, never with a plain `Array`.
- New dependencies go in `[deps]` **and** `[compat]` of the root `Project.toml`; device and plotting packages stay weakdeps behind the extensions, and `GLMakie` must never appear in the root `Project.toml` (CI's `Check Dependencies` fails on it). A test-only package goes in `[extras]` and `[targets]`.
- New boundary conditions: add the type or kernel under `src/boundaryconditions/`, include it through `BoundaryConditions.jl`, validate incompatible face combinations in the constructor, and apply through `flow_bcs!` / `thermal_bcs!`.

## PR

Title starts with `[BUGFIX]`, `[ADDITION]` or `[DOC]`. Fill in `.github/PULL_REQUEST_TEMPLATE.md`: tests added or updated, miniapps updated, no public-API break, docs added, Runic formatting.
