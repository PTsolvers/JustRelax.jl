---
paths:
  - src/**/*.jl
  - ext/**/*.jl
  - test/**/*.jl
  - miniapps/**/*.jl
  - docs/**/*.jl
---

# Style Rules

## Formatting

- Julia files are formatted with **Runic**; CI (`format_check.yml`) comments the diff on every PR. Format only the files you changed: `git runic --inplace <file>` (`git runic main` previews the diff). Never reformat unrelated files.
- Without `git-runic`, install Runic once and call it directly:

  ```sh
  julia --project=@runic -e 'using Pkg; Pkg.add("Runic")'
  julia --project=@runic -e 'using Runic; exit(Runic.main(["--inplace", "path/to/file.jl"]))'
  ```

- Respect `#! format: off` regions (e.g. the banner in `src/JustRelax.jl`).
- CI also runs `typos` (`_typos.toml`). Spell identifiers and comments correctly; add a word to `_typos.toml` only when it is a legitimate identifier (as `iy`, `nd` are).

## Naming

Match the surrounding code. The package has a strong house vocabulary — reuse it instead of inventing synonyms.

- Physical quantities use Unicode math names (`τ`, `η`, `ε`, `ρg`, `∇V`, `λ`) and the existing field names (`stokes.τ.xx_v`, `η_vep`, `ητ`, `θ_dτ`).
- Grid vocabulary: `ni` number of cells (NTuple), `li` lengths, `di` spacing, `_di` reciprocal spacing, `xci`/`xvi` center/vertex coordinates, `igg` the MPI handle, `I...` or `i, j, k` kernel indices.
- Functions and variables are `snake_case`, types `PascalCase`; a function that mutates an argument ends in `!`.
- A leading `_` marks the implementation behind a same-named public function (`solve!` → `_solve!`, `compute_P!` → `_compute_P!`) or a precomputed reciprocal (`_di`). Do not use it as a generic "private" marker.
- New keyword arguments are `snake_case` English (`viscosity_cutoff`, `free_surface`). Legacy `iterMax`, `iterMin`, `nout` stay as they are — renaming a keyword is an API break — but do not extend the camelCase to new names.

## Comments

Be sparing. Code should read from its names and structure.

- Default to no comment. Add one only for a non-obvious invariant, a staggered-index offset, a sign convention, a numerical-stability detail, or a workaround for a specific upstream bug — one line, exactly at the confusing step.
- Never restate what the next line does, narrate the task ("added for X", "fixes Y"), or refer to callers. No TODOs unless asked.
- ❌ `# loop over cells`
- ✅ `# Vx sits on x-faces: one more entry than cells along x`

## Working tree hygiene

- Preserve unrelated user changes. Do not commit generated docs builds, local manifests, benchmark output or simulation data.
