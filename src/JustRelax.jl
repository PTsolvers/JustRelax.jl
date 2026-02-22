module JustRelax

using Adapt
using Reexport
@reexport using ImplicitGlobalGrid
using LinearAlgebra
using Printf
using MPI
using GeoParams
using HDF5
using CellArrays
using StaticArrays
using Statistics
using TOML, Crayons
@reexport using JustPIC

function solve!() end
#! format: off
function __init__(io::IO = stdout)
    if !isa(stdout, Base.TTY)
        return
    end

    j = string(Crayon(foreground = (50,74,201)))
    u = string(Crayon(foreground = (50,74,201)))
    s = string(Crayon(foreground = (50,74,201)))
    t = string(Crayon(foreground = (50,74,201)))
    r = string(Crayon(foreground = (47,138,29)))
    e = string(Crayon(foreground = (47,138,29)))
    l = string(Crayon(foreground = (189,39,39)))
    a = string(Crayon(foreground = (189,39,39)))
    x = string(Crayon(foreground = (130,63,163)))
    res = string(Crayon(reset = true))

    str = """
     $(j)██╗$(u)██╗   ██╗$(s)███████╗$(t)████████╗$(r)██████╗ $(e)███████╗$(l)██╗      $(a)█████╗ $(x)██╗  ██╗$(res)
     $(j)██║$(u)██║   ██║$(s)██╔════╝$(t)╚══██╔══╝$(r)██╔══██╗$(e)██╔════╝$(l)██║     $(a)██╔══██╗$(x)╚██╗██╔╝$(res)
     $(j)██║$(u)██║   ██║$(s)███████╗$(t)   ██║   $(r)██████╔╝$(e)█████╗  $(l)██║     $(a)███████║$(x) ╚███╔╝$(res)
$(j)██   ██║$(u)██║   ██║$(s)╚════██║$(t)   ██║   $(r)██╔══██╗$(e)██╔══╝  $(l)██║     $(a)██╔══██║$(x) ██╔██╗$(res)
$(j)╚█████╔╝$(u)╚██████╔╝$(s)███████║$(t)   ██║   $(r)██║  ██║$(e)███████╗$(l)███████╗$(a)██║  ██║$(x)██╔╝ ██╗$(res)
 $(j)╚════╝ $(u) ╚═════╝ $(s)╚══════╝$(t)   ╚═╝   $(r)╚═╝  ╚═╝$(e)╚══════╝$(l)╚══════╝$(a)╚═╝  ╚═╝$(x)╚═╝  ╚═╝$(res)
     """
    printstyled(io, "\n\n", str, "\n",
"""
Version: $(TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])
Latest commit: $(try strip(read(`git log -1 --pretty=%B`, String)) catch _ "N/A" end)
Commit date: $(try strip(read(`git log -1 --pretty=%cd`, String)) catch _ "N/A" end)
""", bold=true, color=:default)
end

"""
    versioninfo(io::IO=stdout; verbose::Bool=false)

Print information about the version of JustRelax in use. The output includes:

- JustRelax version and installation method
- Git commit information (if available)
- Platform information
- Julia version
- Key dependencies (verbose mode)
- Environment variables (verbose mode)

The output is controlled with boolean keyword arguments:
- `verbose`: print all additional information including dependencies and environment

See also: `Base.versioninfo()`.
"""
function versioninfo(io::IO=stdout; verbose::Bool=false)
    pkg_dir = dirname(@__DIR__)
    project_file = joinpath(pkg_dir, "Project.toml")

    # Read version
    version = try
        get(TOML.parsefile(project_file), "version", "unknown")
    catch
        "unknown"
    end

    println(io, "JustRelax Version $version")

    # Installation method detection
    is_git = isdir(joinpath(pkg_dir, ".git"))
    is_in_depot = any(depot -> startswith(pkg_dir, joinpath(depot, "packages")), DEPOT_PATH)
    is_dev = any(depot -> startswith(pkg_dir, joinpath(depot, "dev")), DEPOT_PATH)

    installation = if is_dev
        "Pkg.develop() or dev mode"
    elseif is_git
        "Git clone"
    elseif is_in_depot
        "Pkg.add() from registry"
    else
        "Custom location"
    end

    println(io, "Installation: $installation")

    # Show git info if available
    if is_git || is_dev
        try
            commit = strip(read(`git -C $pkg_dir log -1 --pretty=%h`, String))
            date = strip(read(`git -C $pkg_dir log -1 --pretty=%cd --date=short`, String))
            branch = strip(read(`git -C $pkg_dir rev-parse --abbrev-ref HEAD`, String))
            println(io, "  Commit: $commit ($date) on $branch")

            if !isempty(strip(read(`git -C $pkg_dir status --porcelain`, String)))
                println(io, "  ⚠ Uncommitted changes detected")
            end
        catch; end
    end

    # Platform info
    println(io, "\nPlatform Info:")
    os_name = Sys.iswindows() ? "Windows" : Sys.isapple() ? "macOS" :
              Sys.islinux() ? "Linux" : Sys.KERNEL
    println(io, "  OS: $os_name (", Sys.MACHINE, ")")
    println(io, "  CPU: ", length(Sys.cpu_info()), " × ", Sys.cpu_info()[1].model)

    verbose && println(io, "  Memory: $(round(Sys.total_memory()/2^30, digits=2)) GB ($(round(Sys.free_memory()/2^20, digits=2)) MB free)")

    println(io, "  WORD_SIZE: ", Sys.WORD_SIZE)
    println(io, "  Threads: $(Threads.nthreads(:default)) default, $(Threads.nthreads(:interactive)) interactive")
    println(io, "\nJulia Version: $VERSION")

    # Verbose information
    if verbose
        # Helper function for GPU status
        gpu_status(pkg) = if !isdefined(Main, pkg)
            "not loaded"
        elseif try getfield(Main, pkg).functional(); catch; false; end
            "functional"
        else
            "loaded but not functional"
        end

        println(io, "\nBackends:")
        println(io, "  CPU: available")
        println(io, "  CUDA: ", gpu_status(:CUDA))
        println(io, "  AMDGPU: ", gpu_status(:AMDGPU))

        println(io, "\nComputational packages:")
        for pkg in ["GeoParams", "JustPIC", "ParallelStencil", "ImplicitGlobalGrid", "MPI"]
            status = if isdefined(Main, Symbol(pkg))
                try string(Base.pkgversion(getfield(Main, Symbol(pkg)))); catch; "loaded"; end
            else
                "not loaded"
            end
            println(io, "  $pkg: $status")
        end

        # Environment variables
        env_strs = ["  $k = $(ENV[k])" for k in keys(ENV) if occursin(r"^JULIA_|^MPI_|^CUDA_|^HIP_", uppercase(k))]
        !isempty(env_strs) && (println(io, "\nEnvironment:"); foreach(s -> println(io, s), env_strs))

        println(io, "\nPackage location:\n  ", pkg_dir)
    end

    nothing
end

export versioninfo

#! format: on
abstract type AbstractBackend end
struct CPUBackend <: AbstractBackend end
struct AMDGPUBackend <: AbstractBackend end

PTArray() = Array
PTArray(::Type{CPUBackend}) = Array
PTArray(::T) where {T} = error(ArgumentError("Unknown backend $T"))

export PTArray, CPUBackend, CUDABackend, AMDGPUBackend

include("stress_rotation/types.jl")
export unwrap

include("types/stokes.jl")

include("types/heat_diffusion.jl")

include("variational_stokes/types.jl")

include("DYREL/types.jl")

include("types/weno.jl")

include("mask/mask.jl")

include("boundaryconditions/Dirichlet.jl")

include("boundaryconditions/types.jl")
export TemperatureBoundaryConditions,
    DisplacementBoundaryConditions, VelocityBoundaryConditions

include("types/traits.jl")
export BackendTrait, CPUBackendTrait, NonCPUBackendTrait

include("topology/Topology.jl")
export IGG, lazy_grid, Geometry, velocity_grids, x_g, y_g, z_g

include("JustRelax_CPU.jl")

include("IO/DataIO.jl")

include("types/type_conversions.jl")
export Array, copy

function plot_particles end
function plot_field end

export plot_particles, plot_field

end # module
