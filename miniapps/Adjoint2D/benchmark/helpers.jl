const isCUDA = false

@static if isCUDA
    using CUDA
    CUDA.allowscalar(false)
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using JustPIC
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend = @static if isCUDA
    JustRelax.CUDABackend
else
    JustRelax.CPUBackend
end

const backend_JP = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

# Taylor-test directions are host arrays; transfer them once before the forward solve.
adjoint_device_multiplier(::Nothing) = nothing
adjoint_device_multiplier(multiplier) = map(PTArray(backend), multiplier)

using CellArrays
adjoint_host_array(A) = Array(A)
adjoint_host_array(A::CellArrays.CellArray) = Array(CellArrays.CPUCellArray(A))
