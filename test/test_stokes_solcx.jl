push!(LOAD_PATH, "..")

using Test
using ParallelStencil
using JustRelax
using ParallelStencil.FiniteDifferences2D

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("../miniapps/benchmarks/stokes/solcx/SolCx.jl")

function check_convergence_case1()
  nx = 64
  ny = 64
  Δη = 1e6
  geometry, stokes, iters, ρ = solCx(Δη, nx=nx, ny=ny)
  iters_expected = (iter = 4000,
                    err_evo1 = [2.687660078359529, 0.24711842751138513, 0.011292850755750898, 0.0004920705891406875, 2.139574580520437e-5, 9.302233455551638e-7, 4.04431715618076e-8, 1.7583442731006563e-9],
                    err_evo2 = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
                   )
  return iters.iter == iters_expected.iter && iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
  @test check_convergence_case1()
end
