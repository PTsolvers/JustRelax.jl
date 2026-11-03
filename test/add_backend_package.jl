using Pkg

# Pkg 1.13.2 precompiles the indirect dependencies an `add` changed by looking them up as direct
# dependencies and fails with "Unknown package" (JuliaLang/Pkg.jl#4866); add without
# auto-precompilation, then precompile the whole environment.
function add_backend_package(name)
    withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
        Pkg.add(name)
    end
    return Pkg.precompile()
end
