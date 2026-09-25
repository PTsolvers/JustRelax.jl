---
name: release
description: Releasing a new JustRelax.jl version — pre-release checks, version and compat bumps, JuliaRegistrator, TagBot. Use when preparing or asked about a release.
---

# Releasing

JustRelax.jl is registered in the Julia General registry. Releases are driven from `main`. Registering is outward-facing: prepare everything, but post the `@JuliaRegistrator register` comment only when the user asks for it.

## Before bumping

1. **CI is green on `main`** across all three systems: GitHub Actions (CPU tests, Runic, typos, docs), Buildkite CUDA and AMDGPU, and the CSCS GH200 pipeline if solver code changed. See the `babysit-ci` skill.
2. **Docs build** and public-API changes carry docstrings and manual updates (`build-docs` skill).
3. **`[compat]` is current.** The tightly coupled companions are GeoParams and JustPIC (both PTsolvers), plus ParallelStencil and ImplicitGlobalGrid; a JustRelax release often follows a compat bump for one of them. Dependabot handles GitHub Actions, not Julia compat, so compat bumps are manual or come from CompatHelper-style PRs. The `julia` compat range should match the CI matrix (`lts` and `1`).

## Steps

1. Bump `version` in `Project.toml` (currently `0.x`: patch for fixes, minor for features or breaking changes, per common 0.x SemVer practice in the Julia ecosystem).
2. Merge the bump to `main`.
3. On the GitHub commit of the version bump, comment:

   ```
   @JuliaRegistrator register

   Release notes:
   - …
   ```

   Breaking changes to exported API must appear in the release notes.
4. JuliaRegistrator opens the General-registry PR. Once it merges, **TagBot** (`.github/workflows/TagBot.yml`) creates the tag and GitHub release. Do not create tags by hand.

## Notes

- `CITATION.cff` and `.zenodo.json` have their own validation workflows (`cff-validator.yml`, `cffconvert.yml`, `validate_zenodo.yml`); update them if the authors change. A release also triggers Zenodo archiving through the repository's Zenodo integration.
