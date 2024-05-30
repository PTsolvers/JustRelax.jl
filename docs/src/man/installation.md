# Installation 

`JustRelax` is a registered package and can be added as follows:

```julia
using Pkg; Pkg.add("JustRelax")
```
or

```julia
julia> ]
(@v1.10) pkg> add JustRelax
```

However, as the API is changing and not every feature leads to a new release, one can also do `add JustRelax#main` which will clone the main branch of the repository.
After installation, you can test the package by running the following commands:

```julia
using JustRelax
julia> ]
  pkg> test JustRelax
```
The test will take a while, so grab a :coffee: or :tea:
