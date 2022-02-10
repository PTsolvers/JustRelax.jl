# Cribs from the ParallelStencil.jl tests, which in turn crib from the MPI.jl tests

# A placeholder for a miniapp, meaning a Julia script that
# would in practice be run as a top-level application,
# and for now may have to be run that way because of the details
# of supporting multiple backends, etc. with ParallelStencil

using Test

# Include the miniapp logic here, perhaps
# also setting a flag or command line argument to not run the
# default case there, so that we can run our tests cases below

# Run tests cases and check properties of the output
# Note that failing tests throw exceptions, which in turn return non-zero exit codes,
# which are then picked up as exceptions by Julia's run(), allowing
# the top level runtests.jl to execute subsidiary tests files like this
@testset begin
  @test 1 == 1
  @test 2 == 2
end
