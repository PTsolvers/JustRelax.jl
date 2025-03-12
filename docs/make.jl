using Documenter, JustRelax
push!(LOAD_PATH, "../src/")

# Get JustRelax.jl root directory
JR_root_dir = dirname(@__DIR__)

license = read(joinpath(JR_root_dir, "LICENSE.md"), String)
write(joinpath(@__DIR__, "src", "man", "license.md"), license)

security = read(joinpath(JR_root_dir, "SECURITY.md"), String)
write(joinpath(@__DIR__, "src", "man", "security.md"), security)

# Copy list of authors to not need to synchronize it manually
authors_text = read(joinpath(JR_root_dir, "AUTHORS.md"), String)
# authors_text = replace(authors_text, "in the [LICENSE.md](LICENSE.md) file" => "under [License](@ref)")
write(joinpath(@__DIR__, "src", "man", "authors.md"), authors_text)

# Copy some files from the repository root directory to the docs and modify them
# as necessary
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
    modules = [JustRelax],
    format = Documenter.HTML(;
        assets = ["assets/favicon.ico"],
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold_ignore = ["man/listfunctions.md"]
    ), # easier local build

    warnonly = Documenter.except(:footnote),
    pages = [
        "Home" => "index.md",
        "User guide" => Any[
            "Installation" => "man/installation.md",
            "Backend" => "man/backend.md",
            "Equations" => "man/equations.md",
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
        ],
        "List of functions" => "man/listfunctions.md",
        "Authors" => "man/authors.md",
        "Contributing" => "man/contributing.md",
        "Code of Conduct" => "man/code_of_conduct.md",
        "Security" => "man/security.md",
        "License" => "man/license.md",
    ],
)

deploydocs(; repo = "github.com/PTsolvers/JustRelax.jl", devbranch = "main")
