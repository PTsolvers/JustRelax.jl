---
name: release
description: Releasing a new JustRelax.jl version — version bumps, compat bumps, JuliaRegistrator, TagBot. Use when preparing or asked about a release.
---

# Releasing

JustRelax.jl is registered in the Julia General registry. Releases are driven from `main`.

## Steps

1. **Bump `version`** in `Project.toml` (currently `0.x` — patch for fixes, minor for features/breaking, per common 0.x SemVer practice in the Julia ecosystem).
2. Make sure `[compat]` entries are current. The tightly-coupled companion packages are **GeoParams** and **JustPIC** (both PTsolvers); a JustRelax release often follows a compat bump for one of these. Dependabot handles GitHub Actions, not Julia compat — compat bumps are manual or via CompatHelper-style PRs.
3. Ensure CI is green on `main`, including the CSCS GPU pipeline if solver code changed.
4. On the GitHub commit of the version bump (after merge to `main`), comment:

   ```
   @JuliaRegistrator register
   ```

5. JuliaRegistrator opens the General registry PR; once merged, **TagBot** (`.github/workflows/TagBot.yml`) creates the GitHub tag/release automatically. Do not create tags by hand.

## Notes

- `CITATION.cff` has its own validation workflows (`cff-validator.yml`, `validate_zenodo.yml`); update it if authors change. A release also triggers Zenodo archiving via the repo's Zenodo integration.
- Breaking changes to exported API should be noted in the release notes (Registrator picks up the text under the `@JuliaRegistrator register` comment via `Release notes:` block).
