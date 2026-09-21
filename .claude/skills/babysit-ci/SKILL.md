---
name: babysit-ci
description: Monitor CI for the current JustRelax.jl branch or PR (GitHub Actions, Buildkite GPU jobs, CSCS GH200), fix mechanical failures, retrigger flaky jobs, and stop to explain anything that needs judgment. Use when asked to watch, triage or fix CI.
---

# Babysit CI

Watch CI for the current branch or PR. Fix small mechanical problems, retrigger flaky jobs, and pause and describe anything that needs judgment.

## Step 0: Agree on the push policy

Ask once whether you may commit and push mechanical fixes to the PR branch. Without a yes, make the fix in the working tree and show the diff instead. Whatever the answer:

- Never push to `main`, never force-push or rewrite history. Add new commits only.
- Stage specific files; do not sweep in unrelated changes (`git status` first).

## What CI runs

**GitHub Actions** (`gh run list --branch <branch>`, `gh pr checks <PR>`):

| Workflow | Notes |
|---|---|
| `CI` (`ci.yml`) | `Pkg.test` on the CPU. Julia `lts` and `1` on Ubuntu, macOS (Intel and aarch64) and Windows. `pre` jobs are `allow_failure`. New pushes cancel the previous run. |
| `Format` (`format_check.yml`) | Runic. Posts the diff as a **PR comment** and does not fail the check, so read the comment, not the status. |
| `Spell Check` (`SpellCheck.yml`) | `typos`, configured by `_typos.toml`. |
| `Check Dependencies` (`Dependency.yml`) | Fails if `GLMakie` appears in the root `Project.toml`. |
| `Documentation` (`Documenter.yml`) | Docs build on PRs, pushes to `main` and tags; deploys a PR preview. |
| `CITATION.cff`, `cffconvert`, `Validate_Zenodo_Metadata` | Only on citation/metadata changes and pushes to `main`. |

**Buildkite** (`julialang/justrelax-dot-jl`, `.buildkite/`): CUDA (Julia 1.10 and 1) and AMDGPU (Julia 1) run `Pkg.test(…; test_args=["--backend=…"])`, 120-minute timeout. A `forerunner` step launches them only when `src/**`, `ext/**`, `test/**`, `**/*.toml` or `.buildkite/*` changed. Exit code 3 is a soft-failed `Pkg.develop` (instantiate) step.

**CSCS GH200** (`ci/cscs-gh200.yml`, GitLab): CUDA with MPI, four listed test files (`test_diffusion2D_multiphase_MPI`, `test_diffusion3D_multiphase_MPI`, `test_shearband2D_MPI`, `test_shearband3D_MPI`), 30-minute limit, 2 nodes × 4 ranks.

## Step 1: Find the run

```sh
git branch --show-current
gh run list --branch "$(git branch --show-current)" --limit 5
gh pr checks <PR_NUMBER>
```

## Step 2: Read the failure before acting

```sh
gh run view <RUN_ID>
gh run view <RUN_ID> --log-failed
```

Buildkite's UI is JavaScript-rendered, so fetching its page returns nothing useful. The pipeline is public; use the log endpoint (`<jid>` is the `jid=` query parameter of the job link):

```sh
curl -sL -H "Accept: application/json" \
  "https://buildkite.com/organizations/julialang/pipelines/justrelax-dot-jl/builds/<N>/jobs/<jid>/log" -o log.json
```

The JSON has one `output` string holding HTML (`<time>` tags and entities): strip those before reading. `…/builds/<N>.json` gives the build state. Never guess from a job name alone.

## Step 3: Triage

### Fix (mechanical, obvious cause)

| Failure | Fix |
|---|---|
| Runic comment on the PR | `git runic --inplace <changed .jl files>` (or apply the suggested diff) |
| `typos` finding | Correct the spelling. Add to `_typos.toml` only for a legitimate identifier. |
| Whitespace (`git diff --check`) | Remove trailing whitespace; keep one final newline |
| `Check Dependencies` fails | Remove `GLMakie` from the root `Project.toml`; plotting backends belong to the Makie extension and the `miniapps`/`docs` environments |
| Docs: new missing-docstring or `@ref` warning from the change | Add the docstring; list a new source file in the `Pages` of `docs/src/man/api/*.md` |
| A new test never ran | Name it `test_*.jl` (only those are collected) |
| GPU-only `UndefVarError: <name>` while CPU is green | Add `<name>` to the `import JustRelax: …` block of all six modules ([backend-rules](../../rules/backend-rules.md)) |
| `ERROR_LAUNCH_OUT_OF_RESOURCES` in a 3D GPU test | Shrink the test grid so the launched range is ≤ 256 cells ([testing-rules](../../rules/testing-rules.md)) |
| A CPU-only test fails on GPU CI | Add it to the GPU exclusion list in `test/runtests.jl` |

After a fix, commit with a descriptive message (if allowed by Step 0) and push, then go back to Step 1. A push cancels the previous GitHub run.

### Retrigger (likely flaky)

| Signal | Action |
|---|---|
| Network, download, `Pkg` resolve/instantiate error | Retrigger |
| Buildkite exit code 3 (soft-failed instantiate) | Retrigger |
| Timeout with no failing test | Retrigger |
| `pre`-Julia job fails in code the PR did not touch | Informational; do not chase it |
| A job unrelated to the changed files | Retrigger, and say so |

```sh
gh run rerun <RUN_ID> --failed
```

For Buildkite and CSCS use "Rebuild"/retry in their UI (needs a login). An empty commit reruns GitHub Actions, but the GPU pipeline launches only if watched paths differ, so it may not restart there.

### Pause and describe

Stop and explain, without editing, for any of:

- A test assertion failing (not a doctest-style output update) or a **numerical regression**: benchmark reference values, iteration counts, residual history.
- A GPU failure with a real error: dynamic invocation, illegal memory access, scalar indexing, type instability.
- An MPI hang or a failure only on a specific rank count — think missing `update_halo!` or a local instead of global reduction ([mpi-rules](../../rules/mpi-rules.md)).
- A build or precompile failure, or the docs build failing in a Literate miniapp.
- Several interrelated failures that hint at one root cause, or failures unrelated to the PR that may be upstream breakage (JustPIC, GeoParams, ParallelStencil).

Report: which jobs failed, the key error lines (trimmed), your assessment of the cause, and fix options.

## Step 4: Confirm green

```sh
gh pr checks <PR_NUMBER>
```

Report the final state per system (GitHub, Buildkite CUDA/AMDGPU, CSCS) and note anything you skipped or that stayed red (e.g. `pre` jobs).
