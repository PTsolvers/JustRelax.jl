using JustRelax

using Pkg
using MPI
using Test, ParallelTestRunner

push!(LOAD_PATH, "..")

function parse_flags!(args, flag; default = nothing, type = typeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(type ≡ nothing || type <: AbstractString)
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

# light tests that require simple or no PS/IGG.
function test_worker(name)
    if name in (
            "test_traits", "test_types", "test_arrays_conversions",
            "test_mask", "test_mini_kernels", "test_Interpolations",
            "test_boundary_conditions2D", "test_boundary_conditions3D",
        )
        return nothing
    end
    return addworker()
end

function runtests(args)
    printstyled("Testing package JustRelax.jl\n"; bold = true, color = :white)

    testsuite = find_tests(@__DIR__)
    for k in collect(keys(testsuite))
        startswith(basename(k), "test_") || delete!(testsuite, k)
    end

    # Separate MPI tests – always run sequentially via mpiexec
    mpi_keys = [k for k in keys(testsuite) if occursin("MPI", k)]
    mpi_files = Dict(k => joinpath(@__DIR__, k * ".jl") for k in mpi_keys)
    for k in mpi_keys
        delete!(testsuite, k)
    end

    # Parse args for ParallelTestRunner and apply exclusions
    pt_args = ParallelTestRunner.parse_args(args)
    if ParallelTestRunner.filter_tests!(testsuite, pt_args)
        for k in collect(keys(testsuite))
            (occursin("burstedde", k) || occursin("VanKeken", k)) && delete!(testsuite, k)
        end
    end

    nfail = 0
    # Parallel tests
    printstyled("\nRunning parallel tests with ParallelTestRunner\n"; bold = true, color = :white)
    try
        ParallelTestRunner.runtests(JustRelax, pt_args; testsuite, test_worker)
    catch ex
        nfail += 1
    end

    # MPI tests
    printstyled("\nRunning MPI tests sequentially\n"; bold = true, color = :cyan)
    for k in sort(mpi_keys)
        f = mpi_files[k]
        isfile(f) || continue
        println("\nRunning MPI test: $f")
        try
            @testset "$k" begin
                n = 2
                project = abspath(@__DIR__)
                p = run(`$(mpiexec()) -n $n $(Base.julia_cmd()) --project=$project --startup-file=no $f`)
                @test success(p)
            end
        catch ex
            @warn "MPI test $k failed: $ex"
            nfail += 1
        end
    end

    return nfail
end

args = copy(ARGS)
_, backend_name = parse_flags!(args, "--backend"; default = "CPU", type = String)

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

exit(runtests(args))
