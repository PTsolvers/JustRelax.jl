import ParallelStencil

module DiffusionNonlinearSolvers_Threads_Float64_3D
  using ParallelStencil
  using ParallelStencil.FiniteDifferences3D
  @init_parallel_stencil(Threads, Float64, 3)
  include("DiffusionNonlinearSolvers_common_3D.jl")
end

ParallelStencil.@reset_parallel_stencil()

module DiffusionNonlinearSolvers_CUDA_Float64_3D
  using ParallelStencil
  using ParallelStencil.FiniteDifferences3D
  @init_parallel_stencil(CUDA, Float64, 3)
  include("DiffusionNonlinearSolvers_common_3D.jl")
end

ParallelStencil.@reset_parallel_stencil()
