using JustRelax

using Pkg

push!(LOAD_PATH, "..")

function parse_flags!(args, flag; default=nothing, typ=typeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(typ ≡ nothing || typ <: AbstractString)
                @show typ val
                val = parse(typ, val)
            end
        else
            val = default
        end

        filter!(x -> x != f, args)
        return true, val
    end
    return false, default
end

function runtests()
    testdir = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(
        filter(
            istest,
            vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...),
        ),
    )
    nfail = 0
    printstyled("Testing package JustRelax.jl\n"; bold=true, color=:white)

    f0 = ("test_traits.jl", "test_types.jl", "test_arrays_conversions.jl")
    for f in f0
        include(f)
    end

    testfiles = [f for f in testfiles if basename(f) ∉ f0]

    for f in testfiles
        occursin("burstedde", f) && continue
        occursin("VanKeken", f) && continue

        println("")
        println("Running tests from $f")
        try
            run(`$(Base.julia_cmd()) -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, f))`)
        catch ex
            nfail += 1
        end
    end

    return nfail
end

_, backend_name = parse_flags!(ARGS, "--backend"; default="CPU", typ=String)

@static if backend_name == "AMDGPU"
    Pkg.add("AMDGPU")
    ENV["JULIA_JUSTRELAX_BACKEND"] = "AMDGPU"
elseif backend_name == "CUDA"
    Pkg.add("CUDA")
    ENV["JULIA_JUSTRELAX_BACKEND"] = "CUDA"
elseif backend_name == "CPU"
    ENV["JULIA_JUSTRELAX_BACKEND"] = "CPU"
end

exit(runtests())
