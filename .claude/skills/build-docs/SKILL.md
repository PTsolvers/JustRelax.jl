---
name: build-docs
description: Build and check the JustRelax.jl documentation (Documenter + DocumenterVitepress) — first-time setup, what the build regenerates, and how to triage its warnings. Use when changing docs, docstrings, API pages or the docs build.
---

# Build documentation

Rules for docs content and docstrings: [docs-rules](../../rules/docs-rules.md) and [docstring-rules](../../rules/docstring-rules.md).

## Steps

1. **First time only** — make the docs environment use the local checkout:

   ```sh
   julia --project=docs --startup-file=no -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
   ```

2. **Build** (the Vitepress theme needs Node, which DocumenterVitepress supplies through `NodeJS_jll`; expect a slow first build):

   ```sh
   julia --project=docs --startup-file=no docs/make.jl
   ```

   The site lands in `docs/build/`. Ask before starting a full build if the change is prose-only: it is heavy.

3. **Triage the output.**
   - `:missing_docs` and `:cross_references` are warn-only in `docs/make.jl`, but do not add new ones. For a missing docstring on a new export: write it, then make sure the source file is listed in the `Pages` of the matching `docs/src/man/api/*.md` `@autodocs` block.
   - Real failures: a Literate error in one of the three miniapps that generate pages, a page missing from `pages`, malformed `@autodocs`/`@ref` syntax.

4. **Review what the build rewrote.** `git status docs/src`. The build regenerates `man/license.md`, `security.md`, `authors.md`, `code_of_conduct.md`, `contributing.md` (from the repo-root files) and the Literate pages `man/diffusion2D_periodic.md`, `man/ShearBand2D.md`, `man/Blankenbach.md` (from miniapps). Edit the sources, not these pages, and commit a regenerated page only when its source changed.

5. **Never commit** `docs/build/`, `docs/site/` or `docs/Manifest*.toml`.

## Layout

- `docs/src/index.md`, `docs/src/man/` — user guide, equations, examples, API reference (`man/api/`), developer guide (`man/developer.md`).
- `docs/make.jl` — page list, generated pages, `checkdocs = :exports`, `modules = [JustRelax, JustRelax2D, JustRelax3D, DataIO]`.
- `docs/paper/` — the JOSS paper; separate from the user docs.
- Deployment: `.github/workflows/Documenter.yml`; PR previews are removed by `DocPreviewCleanup.yml`.

## Notes

- When the exported API changes, update the docstring and the relevant `docs/src/man/` page in the same PR.
- A new user-visible feature usually deserves a mention in the manual page for its area (boundary conditions, backend, grid generation, …) and, for a solver, in `man/developer.md` if it changes how solvers are added.
