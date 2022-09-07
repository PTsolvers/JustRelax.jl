push!(LOAD_PATH, "..")
using Pkg; Pkg.activate("C:\\Users\\albert\\Desktop\\JustRelax.jl")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 3)
environment!(model)

ParallelStencil.@reset_parallel_stencil

model = PS_Setup(:cpu, Float32, 3)
environment!(model)

@testset begin
    @test true  # Success if no exception is thrown before this point
end
