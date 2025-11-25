# Installation

[JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl) is a registered package and can be added as follows:

```julia
using Pkg; Pkg.add("JustRelax")
```
or
```julia-repl
julia> ]

(@v1.10) pkg> add JustRelax
```

!!! info "Install from a specific branch"
    However, as the API is changing and not every new feature leads to a new release, one can also clone the main branch of the repository:
    ```julia
    add JustRelax#main
    ```

If you downloaded or cloned the repository manually, you need to instantiate the package to install all dependencies. Navigate to the directory where you have JustRelax.jl and run:
```julia
julia> ]
(@v1.10) pkg> instantiate
```


After installation, you can test the package by running the following commands:
```julia-repl
using JustRelax

julia> ]

(@v1.10) pkg> test JustRelax
```
The test will take a while, so grab a ☕️ or 🍵
