# Porting JustRelax to JustPIC main

JustPIC main is a KernelAbstractions rewrite (JuliaGeodynamics/JustPIC.jl#291). It drops
ParallelStencil, flattens the `_2D` / `_3D` submodules into dimension-generic kernels,
and dispatches on KernelAbstractions backend types instead of its own. JustRelax does
not load against it:

```
ERROR: LoadError: UndefVarError: `_2D` not defined in `JustPIC`
```

This document records the decisions taken for the port and the order the work runs in.

## Breakage surface

| change | src | test | miniapps | docs |
| :--- | ---: | ---: | ---: | ---: |
| `JustPIC._2D` / `._3D` import sites | 17 | 31 | 90 | 10 |
| `@index` call sites (no longer exported by JustPIC) | 60 | 71 | 321 | 18 |
| `JustPIC.CPUBackend` and friends | 0 | 23 | 74 | 7 |

JustRelax's own ParallelStencil usage (172 `@parallel_indices`, 277 `@parallel`) is
unaffected: ParallelStencil and KernelAbstractions coexist as long as the `@index` name
is resolved explicitly.

Of the ten JustPIC names JustRelax imports, all ten still exist. Five remain exported
(`PhaseRatios`, `cell_index`, `nphases`, `numphases`, `update_phase_ratios!`); five are
now internal (`compute_dx`, `distance`, `face_offset`, `interp1D_extremas`,
`interp1D_inner`).

## Decisions

### `@index` is re-exported, not renamed

JustPIC does `using KernelAbstractions`, so a bare `@index` resolves to
KernelAbstractions' index macro rather than the `CellArraysIndexing` one used for cell
access. JustPIC's own answer is `@cell`.

`JustRelax2D` and `JustRelax3D` import the macro explicitly and re-export it:

```julia
using CellArraysIndexing: @index
export @index
```

Call sites in the package, the tests, the miniapps and the docs are unchanged. The cost
is that a script doing `using KernelAbstractions` alongside JustRelax has an ambiguous
`@index`; such a script must qualify. Migrating to `@cell` is a separate change.

### JustPIC backends are spelled through their owning package

JustPIC no longer defines backend tags. `JustPIC.CPU` is KernelAbstractions' `CPU`, but
the vendor tags are bound inside the package extensions, so `JustPIC.CUDABackend` does
not resolve — the name lives in `JustPICCUDAExt`. Scripts therefore use:

```julia
const backend_JP = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end
```

`CUDA.CUDABackend` and `AMDGPU.ROCBackend` are qualified through the package that owns
them, which also keeps them distinct from JustRelax's own exported `CUDABackend`.

### Internal JustPIC names are qualified at the call site

The five names JustPIC no longer exports are written as `JustPIC.compute_dx`,
`JustPIC.distance`, `JustPIC.face_offset`, `JustPIC.interp1D_extremas` and
`JustPIC.interp1D_inner`, so JustRelax does not depend on upstream's export list.

## Phases

0. **Particle-loss investigation — tracked in JustPIC, not a blocker here.**
   On JustPIC main a uniform velocity field loses particles: with one cell of
   displacement, 444 of 24576 particles disappear and 220 cells are left partially
   filled, scaling with the number of steps and reproducing at 1, 2, 4 and 8 ranks.
   Because it reproduces in serial it is not an MPI fault. Until it is resolved these
   numbers are the reference the port is compared against, so a regression introduced
   here is still distinguishable from the pre-existing loss.
1. **Load.** `src/JustRelax_CPU.jl` and `src/ext/{CUDA,AMDGPU}/{2,3}D.jl`: drop the
   submodule imports, apply the three decisions, move backend dispatch onto the
   KernelAbstractions tags. Done when the package precompiles.
2. **Test suite, CPU, serial.** Done when `Pkg.test()` passes.
3. **MPI tests.** Done when the `*_MPI.jl` tests pass at 2 and 4 ranks.
4. **Miniapps.** Done when one miniapp per family runs (2D/3D, serial/MPI).
5. **Docs.** Done when every `julia` block in `docs/src/man` still runs.
6. **Compat.** `[compat]` bounds, version bump, CI matrix.

## Upstream version collision

JustPIC main carries `version = "0.6.7"`, the same version as the registered release,
while presenting an incompatible API. Any `[compat]` entry that admits the registered
0.6.7 also admits the rewrite and vice versa, so the bound cannot express which of the
two JustRelax needs. `[compat] JustPIC = "0.6.5"` is therefore left as it is until
upstream tags the rewrite as 0.7, at which point phase 6 moves the bound.
