# Julia Execution Rules

## Prefer an MCP Julia REPL over Bash — with one JustRelax caveat

- When an MCP Julia/REPL server is available (a `julia_eval`-style tool), use it for quick checks: loading the package, inspecting types and fields, evaluating a small function, checking array sizes. It has packages loaded and precompiled, so it is much faster than launching Julia.
- Fall back to `julia --project=. --startup-file=no …` through Bash when no REPL tool is available, or when a fresh process is required.
- **`@init_parallel_stencil` runs once per module per session.** A REPL session is therefore bound to one backend and one dimensionality by the first `using JustRelax.JustRelax2D` (or the first script that calls the macro). Restart the REPL to switch backend or 2D↔3D, and never `include` two test files that each call `@init_parallel_stencil` into the same session.
- Run full test files, MPI tests and miniapps through Bash in a fresh process, one file per process.
- Load CUDA or AMDGPU **before** JustRelax if you want the device methods.
