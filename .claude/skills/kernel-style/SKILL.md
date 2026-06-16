---
name: kernel-style
description: ParallelStencil kernel idioms and GPU-compatibility rules for JustRelax.jl, plus the Runic formatting requirement. Use when writing or editing compute kernels or any Julia source.
---

# Kernel style

## ParallelStencil idioms

- Two kernel forms:
  - `@parallel function f!(A, B, ...)` with `@all(A) = ...`, `@inn`, `@av`, `@d_xa`, etc. (FiniteDifferences2D/3D macros) for pure stencil updates.
  - `@parallel_indices (I...) function f!(...)` when you need explicit indexing, branching per cell, or CellArray access. Always `return nothing` at the end. `I...` is the dimension agnostic way if it is a general kernel. Use `i, j` or `i, j, k` for dimension specific kernels  
- Launch with `@parallel (1:nx, 1:ny) f!(...)` or `@parallel f!(...)` for full-array kernels.
- Custom staggered-grid helpers live in `src/MiniKernels.jl` (e.g. `_d_xa`, averaging mini-kernels) — reuse these instead of writing new index arithmetic. `@dxi, @dx, @dy, @dz` come from JustRelax itself.
- Use `@muladd` (MuladdMacro) where the surrounding code does; match local style.

## GPU-compatibility rules (kernels must compile on CUDA/AMDGPU)

- No allocations inside kernels; use `StaticArrays` for small local vectors/tensors.
- No dynamic dispatch, no `try/catch`, no printing inside kernels.
- Material parameters: kernels receive GeoParams rheology structs; calls like `compute_viscosity`, `compute_ρg` must be type-stable and inlined. Phase ratios come from CellArrays (`@cell` / `@index` access patterns).
- Scalar indexing of device arrays outside kernels is forbidden — operate via kernels or broadcast.

## Formatting

The repo is formatted with **Runic** (checked on PRs by `.github/workflows/format_check.yml`). Before committing Julia changes:

```bash
julia --project=@runic -e 'using Pkg; Pkg.add("Runic")'   # one-time
julia --project=@runic -e 'using Runic; exit(Runic.main(["--inplace", "path/to/changed_file.jl"]))'
```

or `git runic main` if the `git-runic` helper is installed. Respect `#! format: off` blocks (e.g. the banner in `src/JustRelax.jl`).
