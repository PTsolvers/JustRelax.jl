push!(LOAD_PATH, "..")

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

    for f in testfiles
        occursin("burstedde", f) && continue

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

exit(runtests())
