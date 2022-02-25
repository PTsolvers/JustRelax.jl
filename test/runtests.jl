push!(LOAD_PATH, "..")

function runtests()
  exename = joinpath(Sys.BINDIR, Base.julia_exename())
  testfiles = filter(x->endswith(x, ".jl") && contains(x, "test_" ), readdir(@__DIR__, join=true))

  nfail = 0
  printstyled("Testing package JustRelax.jl\n"; bold=true, color=:white)

  for f in testfiles
    println("")
    println("Running tests from $f")
    try
      run(`$exename -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, f))`)
    catch ex
      nfail += 1
    end
  end
  return nfail
end

exit(runtests())
