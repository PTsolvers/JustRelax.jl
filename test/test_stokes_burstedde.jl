push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    CPUBackend
end

include("../miniapps/benchmarks/stokes3D/burstedde/Burstedde.jl")

# The PT residual reaching its tolerance only says the discrete system was solved, not that
# the discrete system is the right one, so the errors against the analytical solution are
# checked at two resolutions and their ratio is required to show the expected order.
function check_convergence_case1()
    errors = map((8, 16)) do n
        geometry, stokes, iters = burstedde(; nx = n, ny = n, nz = n, init_MPI = n == 8, finalize_MPI = n == 16)
        iters.err_evo1[end] < 1.0e-8 || error("PT iterations did not converge at nx = $n")
        error_norms(stokes, geometry)
    end

    L2_p, L2_vx, L2_vy, L2_vz = last(errors)
    order = log2.(first(errors) ./ last(errors))

    return all(order[2:end] .> 1.4) && max(L2_vx, L2_vy, L2_vz) < 3.0e-2 && L2_p < 2.0e-1
end

@testset "Burstedde" begin
    @suppress begin
        @test check_convergence_case1()
    end
end
