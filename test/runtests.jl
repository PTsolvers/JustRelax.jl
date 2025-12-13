using JustRelax

using Pkg
using MPI
using Test

push!(LOAD_PATH, "..")

function parse_flags!(args, flag; default = nothing, type = typeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(type ≡ nothing || type <: AbstractString)
                @show type val
                val = parse(type, val)
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
    printstyled("Testing package JustRelax.jl\n"; bold = true, color = :white)

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
        if occursin("MPI", f)
            try
                @testset "$(basename(f))" begin
                    n = 2
                    p = run(`$(mpiexec()) -n $n $(Base.julia_cmd()) -O3 --startup-file=no $(joinpath(testdir, f))`)
                    @test success(p)
                end
            catch ex
                nfail += 1
            end
        else
            try
                run(`$(Base.julia_cmd()) -O3 -t auto --startup-file=no $(joinpath(testdir, f))`)
            catch ex
                nfail += 1
            end
        end
    end

    return nfail
end

_, backend_name = parse_flags!(ARGS, "--backend"; default = "CPU", type = String)

@static if backend_name == "AMDGPU"
    Pkg.add("AMDGPU")
    ENV["JULIA_JUSTRELAX_BACKEND"] = "AMDGPU"
    using AMDGPU; AMDGPU.versioninfo()
elseif backend_name == "CUDA"
    Pkg.add("CUDA")
    ENV["JULIA_JUSTRELAX_BACKEND"] = "CUDA"
    using CUDA; CUDA.versioninfo()
elseif backend_name == "CPU"
    ENV["JULIA_JUSTRELAX_BACKEND"] = "CPU"
end

exit(runtests())
