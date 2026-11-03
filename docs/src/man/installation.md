# Installation

JustRelax requires Julia 1.10 or later. Install the registered release from the Julia REPL:

```julia
using Pkg
Pkg.add("JustRelax")
using JustRelax
```

To use the unreleased main branch instead:

```julia
Pkg.add(url = "https://github.com/PTsolvers/JustRelax.jl", rev = "main")
```

See [Selecting the backend](@ref) for CPU and GPU setup, then follow the
[Getting started example](./diffusion2D_periodic.md).

## Working with a local checkout

After cloning the repository, run these commands from its root directory.
Activating the project ensures that Julia loads the local source and installs
its dependencies into the correct environment:

```sh
julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'
julia --project=. --startup-file=no -e 'using JustRelax; println("loaded")'
```

## Testing

To test the installed package from Julia:

```julia
using Pkg
Pkg.test("JustRelax")
```

To test a local checkout from its root directory:

```sh
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'
```

The full suite can take some time and includes MPI tests. See
[Contributing](@ref) for focused tests and backend-specific commands.

## Running the miniapps

The `miniapps/` folder contains examples and benchmarks with a separate
project environment. From the repository root, register the local checkout in
that environment and install its dependencies:

```sh
julia --project=miniapps --startup-file=no -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
```

Then run an example in the same environment:

```sh
julia --project=miniapps --startup-file=no miniapps/benchmarks/stokes2D/shear_band/ShearBand2D.jl
```

Check the script's backend selection and resolution before running it; some
miniapps require GPU hardware, MPI, or graphics support. Use a clone of the
repository when working with these examples so their relative data and helper
paths remain intact.
