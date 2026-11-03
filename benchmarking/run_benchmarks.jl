const BACKEND_FLAGS = filter(startswith("--backend="), ARGS)
const BACKEND = isempty(BACKEND_FLAGS) ? "CPU" : split(only(BACKEND_FLAGS), '='; limit = 2)[2]
BACKEND in ("CPU", "CUDA", "AMDGPU") ||
    error("Unknown backend $(repr(BACKEND)); use --backend=CPU|CUDA|AMDGPU")

using JustRelax
using JustRelaxBenchmarks
@static if BACKEND == "CUDA"
    using CUDA
elseif BACKEND == "AMDGPU"
    using AMDGPU
end

# Separate top-level statement: the vendor bindings exist only after `using` has run.
# JustRelax and JustPIC tag backends with different types; CUDA.jl exports a `CUDABackend`
# of its own, so both are qualified.
backend, backend_JP, device = if BACKEND == "CUDA"
    JustRelax.CUDABackend, CUDA.CUDABackend, CUDA.name(CUDA.device())
elseif BACKEND == "AMDGPU"
    JustRelax.AMDGPUBackend, AMDGPU.ROCBackend, string(AMDGPU.device())
else
    JustRelax.CPUBackend, JustPIC.CPU, nothing
end

JustRelaxBenchmarks.main(
    filter(!startswith("--backend="), ARGS); backend, backend_JP, backend_name = BACKEND, device
)
