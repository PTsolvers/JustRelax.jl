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
        "Home"      => "index.md",
        "User guide"=> Any[
            "Installation" => "man/installation.md",
            "Backend"   => "man/backend.md",
            "Equations" => "man/equations.md",
            "Advection" => "man/advection.md",
            ],
        "Examples"  => Any[
            "Blankenbach"   => "man/Blankenbach.md",
            "Shear Bands"   => "man/ShearBands.md",
            "Subduction 2D" => Any[
                "Model setup"          => "man/subduction2D/setup.md",
                "Rheology"             => "man/subduction2D/rheology.md",
                "Setting up the model" => "man/subduction2D/subduction2D.md",
                ]
            ],
        "List of functions" => "man/listfunctions.md",
    ],
)

deploydocs(; repo="github.com/PTsolvers/JustRelax.jl", devbranch="main")
