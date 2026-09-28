---
paths:
  - src/**/*.jl
  - ext/**/*.jl
---

# Docstring Rules

Every exported symbol needs a docstring. `docs/make.jl` runs with `checkdocs = :exports`, which reports missing ones (as warnings, since `:missing_docs` is in `warnonly` — still do not add new ones).

## Format

Follow `solve_DYREL!` in `src/DYREL/solver.jl`:

~~~
"""
    name(arg1, arg2; kw = default)

One-sentence summary in the imperative ("Compute …", "Solve …").

# Arguments
- `arg1`: meaning, units, and where it lives on the staggered grid (`(nx+1, ny+2)`, cell centers, …).

# Keyword Arguments
- `kw`: meaning. Default: `default`.
"""
~~~

- Indent the signature four spaces so Documenter renders it as the signature, and list arguments in call order.
- Give array sizes, staggered location and units wherever they matter; the `Velocity` constructors' docstrings are the model.
- Document every form a user can call (e.g. `grid` versus a legacy `di` tuple) and the keyword convention of the solver ([api-rules](api-rules.md)).
- Cross-reference with `` [`name`](@ref) ``. Broken `@ref`s are warnings in this build, but do not introduce new ones.
- Reference existing physics or numerics papers where the method comes from one (e.g. "after Kiss et al. (2023)").
- No doctests are used in this repo. Examples are short, self-contained `julia` code blocks. Do not add `jldoctest` blocks that run solver iterations: they would execute on every docs build.
- A `_`-prefixed implementation function normally needs no docstring; document the public function it backs.
