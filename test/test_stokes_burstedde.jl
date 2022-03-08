push!(LOAD_PATH, "..")

using Test
using ParallelStencil
using ImplicitGlobalGrid
using JustRelax

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

include("../miniapps/benchmarks/stokes3D/burstedde/Burstedde.jl")

function check_convergence_case1()
  nx = 16
  ny = 16
  nz = 16
  _, _, iters = burstedde(;nx=nx, ny=ny, nz=nz, init_MPI = true, finalize_MPI = true)
  iters_expected = (
    iter = 10001,
    err_evo1 = [0.0001926889212057774, 0.00019268875672200957, 0.00019268875672138723, 0.00019268875672138845, 0.00019268875672140043, 0.0001926887567213953, 0.000192688756721381, 0.00019268875672139282, 0.00019268875672140862, 0.00019268875672139501, 0.00019268875672139873, 0.00019268875672138694, 0.00019268875672137853, 0.00019268875672138764, 0.0001926887567214029, 0.00019268875672138233, 0.00019268875672138902, 0.0001926887567213886, 0.00019268875672137748, 0.00019268875672139144],
    err_evo2 = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000],
  )
  return iters.iter == iters_expected.iter && iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
  @test check_convergence_case1()
end




