# Contributing

[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) is an open-source project and we are very happy to accept contributions
from the community. Please feel free to open [Issues](https://github.com/PTsolvers/JustRelax.jl/issues/new) (issue templates in [here](https://github.com/PTsolvers/JustRelax.jl/tree/main/.github/ISSUE_TEMPLATE)) or submit [Pull Requests](https://github.com/PTsolvers/JustRelax.jl/pulls) (PR template in [here](https://github.com/PTsolvers/JustRelax.jl/blob/main/.github/PULL_REQUEST_TEMPLATE.md)) to the `main` branch with your contribution. For planned large contributions, it is often
beneficial to get in contact with one of the principal developers first (see
[AUTHORS.md](AUTHORS.md)).

## Getting set up

Load the package from the repository root:

```sh
julia --project=. -e 'using JustRelax'
```

Run the test suite (CPU by default; runs in parallel, plus MPI tests with two ranks):

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=test test/runtests.jl
```

Run a single test file directly when iterating on a focused change:

```sh
JULIA_JUSTRELAX_BACKEND=CPU julia --project=test test/test_diffusion2D.jl
```

The runner also accepts `--backend=CUDA`/`--backend=AMDGPU` for accelerator testing (requires the corresponding hardware):

```sh
JULIA_JUSTRELAX_BACKEND=CUDA julia --project=test test/runtests.jl --backend=CUDA
```

Try out example/benchmark scripts using the `miniapps` environment, e.g.:

```sh
julia --project=miniapps miniapps/subduction/2D/Subduction2D.jl
```

## Code style

Julia files are formatted with [Runic.jl](https://github.com/fredrikekre/Runic.jl), and CI checks formatting on every pull request:

```sh
git runic main              # show formatting differences
git runic --inplace .       # apply formatting
```

Format only the files your change touches, and review the resulting diff before committing.

Kernels should stay backend-agnostic (CPU/CUDA/AMDGPU) and dimension-agnostic (2D/3D) where possible — see `src/common.jl` for where shared solver code lives, and the module structure in `src/JustRelax_CPU.jl` and `ext/`.

## Documentation

If your change affects a public type or function, add or update its docstring, and build the docs locally to check it renders:

```sh
julia --project=docs docs/make.jl
```

## Pull requests

Use a descriptive PR title beginning with the appropriate tag, such as `[BUGFIX]`, `[ADDITION]`, or `[DOC]`. Explain the motivation for the change, and include relevant tests, miniapp updates, and documentation updates alongside it. Note any API compatibility considerations.

[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) and its contributions are licensed under the MIT license. As a contributor, you certify that all your
contributions are in conformance with the *Developer Certificate of Origin
(Version 1.1)*, which is reproduced below.

## Developer Certificate of Origin (Version 1.1)
The following text was taken from
[https://developercertificate.org](https://developercertificate.org):

    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
