---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# Kernel Rules

Kernels are compiled for CPU threads, CUDA and AMDGPU from the same source, in 2D and 3D. CPU-green does not mean GPU-green.

## Form

- Pure stencil updates: `@parallel function f!(A, B, …)` with the FiniteDifferences macros (`@all`, `@inn`, `@av`, `@d_xa`, …).
- Explicit indexing (per-cell branching, CellArray access, staggered offsets): `@parallel_indices (I...) function f!(…)`. `I...` keeps the kernel dimension-agnostic; use `i, j` / `i, j, k` only when the kernel is inherently 2D/3D. End with `return nothing`.
- Launch with `@parallel (@idx ni) f!(…)`, where `@idx` builds the ranges from the cell counts. Never hard-code `1:nx`. Launch over the extent of the array being written: `thermal.T` carries a ghost ring, so its interior is `@idx size(T) .- 2`.
- Reuse `src/MiniKernels.jl` (`center`, `left`, `right`, `next`, `back`, `front`, averaging helpers) instead of writing new index arithmetic.
- Match the neighbouring kernels: `@muladd` (MuladdMacro), `@inbounds` on indexing in kernel bodies, `@inline` on every helper a kernel calls.

## GPU compatibility

- Type-stable and allocation-free. Use `StaticArrays` for small local vectors and tensors.
- No dynamic dispatch, `try/catch`, printing, `error`/`throw`/`@assert` inside a kernel. Validate arguments in the launching function.
- Rheology arrives as GeoParams `NTuple{N, MaterialParams}` and phase ratios as CellArrays. Calls such as `compute_viscosity` and `compute_ρg` must be type-stable and inlined. Read cell arrays with `@index`.
- `@index` is CellArraysIndexing's macro (re-exported by `JustRelax2D`/`JustRelax3D`). KernelAbstractions has an unrelated macro of the same name — do not `using KernelAbstractions` in a file that indexes cell arrays.
- No scalar indexing of device arrays outside kernels, and no bare `for` over cells in solver code: use a kernel or a broadcast. To fill a device array from host data, build it on the host and assign whole: `A .= PTArray(backend)(host)`.
- Do not hard-code `Float64` in kernels; the element type comes from the arrays.
- Fused kernels are register-heavy and have a launch-size limit on GPU (`ERROR_LAUNCH_OUT_OF_RESOURCES`). See the 3D block-size pitfall in [testing-rules](testing-rules.md) before changing a fat kernel or a 3D test grid.

## Staggered grid and indexing

Full layout table: `.agents/grid.md`.

- `ni` is the number of **cells**; vertices are `ni .+ 1`. Never infer an array's staggered location from its size. In 2D: `P`, `τ.xx`, `viscosity.η` are `(nx, ny)`; `V.Vx` is `(nx+1, ny+2)`; `V.Vy` is `(nx+2, ny+1)`; `τ.xy` is `(nx+1, ny+1)`; `thermal.T` is `(nx+2, ny+2)`.
- Read spacing through `@dxi`, `@dx`, `@dy`, `@dz` on `grid._di` (a NamedTuple with `.center`, `.vertex`, `.velocity`); take the entry that matches the field you differentiate. Do not close over a scalar `dx` — it silently breaks on nonuniform grids.
- The vertical direction is the **last** dimension and points up.
- Keep 2D and 3D kernel files separate. Never include a 2D kernel file from a 3D module, or the reverse.
