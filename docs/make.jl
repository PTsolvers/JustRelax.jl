using Documenter, JustRelax
push!(LOAD_PATH, "../src/")

@info "Making documentation..."
makedocs(;
    sitename="JustRelax.jl",
    authors="Albert de Montserrat and contributors",
    modules=[JustRelax],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true",
    size_threshold_ignore = ["man/listfunctions.md"]), # easier local build

    warnonly = Documenter.except(:footnote),
    pages=[
        "Home"      => "man/index.md",
        "User guide"=> Any[
            "Installation" => "man/installation.md",
            "Backend"   => "man/backend.md",
            "Equations" => "man/equations.md",
            "Advection" => "man/advection.md",
            ],
        "Examples"  => Any[
            "Blankenbach" => "man/Blankenbach.md",
            "Shear Bands" => "man/ShearBands.md",
            ],
        "List of functions" => "man/listfunctions.md",
    ],
)

deploydocs(; repo="github.com/PTsolvers/JustRelax.jl", devbranch="main")
