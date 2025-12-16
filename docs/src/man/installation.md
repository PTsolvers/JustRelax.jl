# Installation

[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) is a registered package and can be added as follows:

```julia
using Pkg; Pkg.add("JustRelax")
```
or
```julia-repl
julia> ]

(@v1.xx) pkg> add JustRelax
```

!!! info "Install from a specific branch"
    However, as the API is changing and not every new feature leads to a new release, one can also clone the main branch of the repository:
    ```julia
    add JustRelax#main
    ```

If you downloaded or cloned the repository manually, you need to instantiate the package to install all dependencies. Navigate to the directory where you have JustRelax.jl and run:
```julia
julia> ]
(@v1.xx) pkg> instantiate
```


After installation, you can test the package by running the following commands:
```julia-repl
using JustRelax

julia> ]

(@v1.xx) pkg> test JustRelax
```
The test will take a while, so grab a ☕️ or 🍵

# Running the miniapps

Available benchmarks and examples can be found in the `miniapps/` folder. These scripts are simple and easy to understand, providing a good basis for more complex applications. These miniapps have their own environment and dependencies, defined in `miniapps/Project.toml`, so they need to be instantiated separately. If you cloned the repository, navigate to `path/to/JustRelax.jl/` and run:
```julia-repl
julia> ]
(@v1.xx) pkg> activate miniapps

(@v1.xx) pkg> instantiate

(@v1.xx) pkg> activate .
```
After that, you can run any of the miniapps, for example:
```julia
julia> using JustRelax

julia> include("miniapps/benchmarks/stokes2D/shear_band/ShearBand2D.jl")
```

If JustRelax.jl is installed via the package manager, the dependencies that are exclusive to the miniapps should be added manually.