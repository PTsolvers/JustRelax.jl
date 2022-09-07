push!(LOAD_PATH, "..")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

ParallelStencil.@reset_parallel_stencil

model = PS_Setup(:cpu, Float32, 2)
environment!(model)

@testset begin
    @test true  # Success if no exception is thrown before this point
end
