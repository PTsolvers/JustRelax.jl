using Test

function runtests()
  exename = joinpath(Sys.BINDIR, Base.julia_exename())
  testdir = pwd()
  testfiles = [joinpath(testdir, "test_placeholder.jl")]
  nfail = 0
  for f in testfiles
    println("")
    try
      run(`$exename -O3 --startup-file=no --check-bounds=no $(joinpath(testdir, f))`)
    catch ex
      nfail += 1
    end
    return nfail
  end
end

exit(runtests())
