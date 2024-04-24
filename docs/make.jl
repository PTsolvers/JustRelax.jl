using Documenter, JustRelax
push!(LOAD_PATH, "../src/")

@info "Making documentation..."
makedocs(;
    sitename="JustRelax.jl",
    authors="Albert de Montserrat and contributors",
    modules=[JustRelax],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"), # easier local build
    warnonly = Documenter.except(:footnote),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(; repo="github.com/JuliaGeodynamics/JustPIC.jl", devbranch="main")
