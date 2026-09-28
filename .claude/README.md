# `.claude/` — rules and skills for Claude Code

Agent guidance in this repository has three layers:

| Layer | Where | What it is |
|---|---|---|
| Entry point | [`AGENTS.md`](../AGENTS.md) | Project overview, repository map, development rules, commands, verification checklist |
| Deep references | [`.agents/`](../.agents/) | Tool-agnostic explanations of the API, solvers, grid layout and MPI (read before touching that area) |
| **Rules** | `.claude/rules/` | Short, prescriptive constraints, loaded automatically when Claude works on files matching their `paths:` |
| **Skills** | `.claude/skills/<name>/SKILL.md` | Step-by-step procedures, invoked on demand (`/name`) or picked up from their description |

Rules say **what must hold** while editing; skills say **how to do a task**. Rules stay terse and link to `.agents/` for depth instead of repeating it.

## Rules

| Rule | Applies to | Covers |
|---|---|---|
| [`style-rules`](rules/style-rules.md) | all Julia | Runic, typos, naming vocabulary, comments |
| [`kernel-rules`](rules/kernel-rules.md) | `src/`, `ext/` | ParallelStencil idioms, GPU compatibility, staggered grid and spacing |
| [`backend-rules`](rules/backend-rules.md) | `src/`, `ext/` | Module structure, the six import headers, `@init_parallel_stencil`, trait dispatch |
| [`mpi-rules`](rules/mpi-rules.md) | `src/`, `ext/` | Halo exchange, global reductions, rank-0 output |
| [`api-rules`](rules/api-rules.md) | `src/`, `ext/` | Exports, deprecation, three-layer dispatch, solver keywords |
| [`docstring-rules`](rules/docstring-rules.md) | `src/`, `ext/` | Docstring format and content |
| [`docs-rules`](rules/docs-rules.md) | `docs/` | Generated pages, API pages, build warnings |
| [`testing-rules`](rules/testing-rules.md) | `test/` | How `runtests.jl` collects tests, GPU/MPI conventions, tiny 3D grids |
| [`miniapps-rules`](rules/miniapps-rules.md) | `miniapps/` | Script structure, environment, Literate pages |
| [`julia-repl-rules`](rules/julia-repl-rules.md) | always | Prefer an MCP Julia REPL; one backend and dimension per session |

## Skills

| Skill | Use it to |
|---|---|
| [`add-feature`](skills/add-feature/SKILL.md) | Add a solver, public function, kernel, boundary condition or rheology hook end to end |
| [`running-tests`](skills/running-tests/SKILL.md) | Pick, run and debug tests on CPU, GPU and MPI |
| [`build-docs`](skills/build-docs/SKILL.md) | Build the docs and triage warnings |
| [`new-miniapp`](skills/new-miniapp/SKILL.md) | Set up and validate a new model or benchmark |
| [`miniapps-and-benchmarks`](skills/miniapps-and-benchmarks/SKILL.md) | Run existing miniapps to validate a change; which one checks what |
| [`babysit-ci`](skills/babysit-ci/SKILL.md) | Watch CI, fix mechanical failures, retrigger flaky jobs |
| [`release`](skills/release/SKILL.md) | Prepare a release |

## Maintaining these files

- New always-on constraint → a rule with the narrowest `paths:` that makes sense. New procedure → a skill.
- Ground every statement in the code or CI config; when a rule and the code disagree, fix the rule.
- Keep machine-specific facts (a local GPU, personal paths) out of this directory; they belong in personal memory, not in files checked in for every contributor.
