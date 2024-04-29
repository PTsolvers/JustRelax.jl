# Installation instructions

JustRelax.jl is written in the [julia](https://julialang.org) programming language, which is an extremely powerful, modern, scientific computing language. Julia works on all major operating systems, is free, fast, and has a very active user basis (with many useful packages). In case you haven't heard about julia yet, you are not alone. Yet, perhaps a look at [this](https://www.nature.com/articles/d41586-019-02310-3) or [this](https://thenextweb.com/news/watch-out-python-julia-programming-coding-language-coming-for-crown-syndication) article, which explains nicely why it has an enormous potential for computational geosciences as well.

### 1. Install julia
In order to use then package you need to install julia. We recommend downloading from the [julia](https://julialang.org) webpage and follow the instructions .


### 2. Install Visual Studio Code
The julia files itself are text files (just like matlab scripts). You may want to edit or modify them at some stage, for which you can use any text editor for that. We prefer to use the freely available [Visual Studio Code](https://code.visualstudio.com) as it has a build-in terminal and is the comes with the (official) julia debugger (install the Julia extension for that).

### 3. Getting started with julia
You start julia on the command line with:
```
usr$ julia
```
This will start the command-line interface of julia:
```julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.10.2 (2024-03-01)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>

```

From the julia prompt, you start the package manager by typing `]`:
```julia
(@v1.10) pkg>
```
And you return to the command line with a backspace.

Also useful is that julia has a build-in terminal, which you can reach by typing `;` on the command line:
```julia
julia>;
shell>
```
In the shell, you can use the normal commands like listing the content of a directory, or the current path:
```julia
shell> ls
LICENSE         Manifest.toml   Project.toml    README.md       docs            src             test            tutorial
shell> pwd
/home/usr/Documents/JustRelax.jl
```
As before, return to the main command line (called `REPL`) with a backspace.

If you want to see help information for any julia function, type `?` followed by the command.
An example for `Array` is:
```julia
help?> Array
search: Array SubArray BitArray DenseArray StridedArray PermutedDimsArray AbstractArray AbstractRange AbstractIrrational

  Array{T,N} <: AbstractArray{T,N}


  N-dimensional dense array with elements of type T.

  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  Array{T}(undef, dims)
  Array{T,N}(undef, dims)


  Construct an uninitialized N-dimensional Array containing elements of type T. N can either be supplied explicitly, as
  in Array{T,N}(undef, dims), or be determined by the length or number of dims. dims may be a tuple or a series of
  integer arguments corresponding to the lengths in each dimension. If the rank N is supplied explicitly, then it must
  match the length or number of dims. Here undef is the UndefInitializer.

  Examples
  ≡≡≡≡≡≡≡≡

  julia> A = Array{Float64, 2}(undef, 2, 3) # N given explicitly
  2×3 Matrix{Float64}:
   6.90198e-310  6.90198e-310  6.90198e-310
   6.90198e-310  6.90198e-310  0.0

   [...]
```

If you are in a directory that has a julia file (which have the extension `*.jl`), you can open that file with Visual Studio Code:
```julia
shell> code runtests.jl
```
Execute the file with:
```julia
julia> include("runtests")
```
Note that you do not include the `*.jl` extension. Another option is to run the code line by line through the `Julia REPL` via `Shift+Enter`, this is especially useful for debugging purposes.


### 4. Install JustRelax.jl

`JustRelax.jl` is a registered package and can be added as follows:

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
