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
backend, device = if BACKEND == "CUDA"
    CUDABackend, CUDA.name(CUDA.device())
elseif BACKEND == "AMDGPU"
    AMDGPUBackend, string(AMDGPU.device())
else
    CPUBackend, nothing
end

JustRelaxBenchmarks.main(filter(!startswith("--backend="), ARGS); backend, backend_name = BACKEND, device)
