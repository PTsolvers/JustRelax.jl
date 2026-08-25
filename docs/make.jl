using Documenter
using DocumenterVitepress
using Literate

using JustRelax
using GeoParams, JustPIC

# Get JustRelax.jl root directory
JR_root_dir = dirname(@__DIR__)

# Literate.jl-generated example pages: the miniapp script is the single source,
# this page is generated from it, so it cannot drift from the runnable script.
Literate.markdown(
    joinpath(JR_root_dir, "miniapps", "benchmarks", "thermal_diffusion", "diffusion", "diffusion2D_periodic.jl"),
    joinpath(@__DIR__, "src", "man");
    documenter = true, execute = false,
)

license = read(joinpath(JR_root_dir, "LICENSE.md"), String)
write(joinpath(@__DIR__, "src", "man", "license.md"), license)

security = read(joinpath(JR_root_dir, "SECURITY.md"), String)
write(joinpath(@__DIR__, "src", "man", "security.md"), security)

# Copy list of authors to not need to synchronize it manually
authors_text = read(joinpath(JR_root_dir, "AUTHORS.md"), String)
# authors_text = replace(authors_text, "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
write(joinpath(@__DIR__, "src", "man", "authors.md"), authors_text)

# Copy some files from the repository root directory to the docs and modify them as necessary
# Based on: https://github.com/ranocha/SummationByPartsOperators.jl/blob/0206a74140d5c6eb9921ca5021cb7bf2da1a306d/docs/make.jl#L27-L41
open(joinpath(@__DIR__, "src", "man", "license.md"), "w") do io
    # Point to source license file
    println(
        io, """
        ```@meta
        EditURL = "https://github.com/PTsolvers/JustRelax.jl/blob/main/LICENSE.md"
        ```
        """
    )
    # Write the modified contents
    println(io, "# [License](@id license)")
    println(io, "")
    for line in eachline(joinpath(dirname(@__DIR__), "LICENSE.md"))
        line = replace(line, "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)")
        println(io, "> ", line)
    end
end

open(joinpath(@__DIR__, "src", "man", "code_of_conduct.md"), "w") do io
    # Point to source license file
    println(
        io, """
        ```@meta
        EditURL = "https://github.com/PTsolvers/JustRelax.jl/blob/main/CODE_OF_CONDUCT.md"
        ```
        """
    )
    # Write the modified contents
    println(io, "# [Code of Conduct](@id code-of-conduct)")
    println(io, "")
    for line in eachline(joinpath(dirname(@__DIR__), "CODE_OF_CONDUCT.md"))
        line = replace(line, "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)")
        println(io, "> ", line)
    end
end

open(joinpath(@__DIR__, "src", "man", "contributing.md"), "w") do io
    # Point to source license file
    println(
        io, """
        ```@meta
        EditURL = "https://github.com/PTsolvers/JustRelax.jl/blob/main/CONTRIBUTING.md"
        ```
        """
    )
    # Write the modified contents
    for line in eachline(joinpath(dirname(@__DIR__), "CONTRIBUTING.md"))
        line = replace(line, "[LICENSE.md](LICENSE.md)" => "[License](@ref)")
        line = replace(line, "[AUTHORS.md](AUTHORS.md)" => "[Authors](@ref)")
        println(io, line)
    end
end
@info "Making documentation..."

makedocs(;
    sitename = "JustRelax.jl",
    authors = "Albert de Montserrat, Pascal Aellig and contributors",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/PTsolvers/JustRelax.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    modules = [JustRelax],
    checkdocs = :exports,
    # :missing_docs stays a warning: JustRelax2D.Data/JustRelax3D.Data are
    # ParallelStencil.@init_parallel_stencil-generated submodules whose docstring
    # carries @ref links into ParallelStencil, which this build does not document.
    warnonly = [:missing_docs],
    pages = [
        "Home" => "index.md",
        "Getting started" => "man/diffusion2D_periodic.md",
        "User guide" => Any[
            "Installation" => "man/installation.md",
            "Backend" => "man/backend.md",
            "Grid generation" => "man/grid_generation.md",
            "Core objects" => "man/core_objects.md",
            "Equations" => Any[
                "Governing equations" => "man/equations_basic.md",
                "Constitutive equations" => "man/constitutive_equations.md",
                "APT equations" => "man/equations_APT.md",
                "Discretization" => "man/equations_discretization.md",
                "Material physics" => "man/material_physics.md",
            ],
            "Boundary conditions" => "man/boundary_conditions.md",
            "Advection" => "man/advection.md",
        ],
        "Examples" => Any[
            "Blankenbach" => "man/Blankenbach.md",
            "Shear Bands" => "man/ShearBands.md",
            "Subduction 2D" => Any[
                "Model setup" => "man/subduction2D/setup.md",
                "Rheology" => "man/subduction2D/rheology.md",
                "Setting up the model" => "man/subduction2D/subduction2D.md",
            ],
            "Plume 3D" => Any[
                "Rheology" => "man/plume3D/rheology.md",
                "Setting up the model" => "man/plume3D/plume3D.md",
            ],
            "Self-tuned APT solver" => "man/DYREL.md",
            "Checkpointing/Restart" => Any[
                "Checkpointing" => "man/checkpointing.md",
                "Restart" => "man/restart.md",
            ],
        ],
        "API reference" => Any[
            "Stokes" => "man/api/stokes.md",
            "Thermal" => "man/api/thermal.md",
            "Boundary conditions" => "man/api/boundary_conditions.md",
            "Rheology and phases" => "man/api/rheology_phases.md",
            "I/O and checkpointing" => "man/api/io.md",
            "Grid" => "man/api/grid.md",
            "Index" => "man/listfunctions.md",
        ],
        "Citing JustRelax.jl" => "man/citing.md",
        "References" => Any[
            "JustPIC" => "man/JustPIC.md",
            "GeoParams" => "man/GeoParams.md",
        ],
        "Authors" => "man/authors.md",
        "Contributing" => "man/contributing.md",
        "Code of Conduct" => "man/code_of_conduct.md",
        "Security" => "man/security.md",
        "License" => "man/license.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/PTsolvers/JustRelax.jl",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main", # or master, trunk, ...
    push_preview = true,
)
