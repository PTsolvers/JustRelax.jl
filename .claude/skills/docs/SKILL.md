---
name: docs
description: Building and editing JustRelax.jl documentation (Documenter + DocumenterVitepress). Use when changing docs, docstrings, or the docs build.
---

# Documentation

Docs use **Documenter.jl + DocumenterVitepress** (Vitepress theme, deployed by `.github/workflows/Documenter.yml`).

## Build locally

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The Vitepress build requires Node; DocumenterVitepress handles it via NodeJS_jll. Built site lands in `docs/build/`.

## Layout and conventions

- Source pages: `docs/src/` (`index.md`, `man/` for manual pages, `assets/`).
- `docs/make.jl` **auto-generates** `docs/src/man/license.md`, `security.md`, and `authors.md` from the repo-root `LICENSE.md`, `SECURITY.md`, `AUTHORS.md` — never edit those generated files; edit the root files instead.
- New pages must be added to the `pages` list in `docs/make.jl` to appear in navigation.
- The docs env loads `GeoParams` and `JustPIC` too, so doctests/examples can use them.
- `docs/paper/` is the JOSS paper — separate from the user docs, don't mix content.
- PR doc previews are cleaned up by `DocPreviewCleanup.yml`; nothing to do manually.

When changing exported API, update the relevant page in `docs/src/man/` and the docstring in the same PR.
