push!(LOAD_PATH, "..")

using Test
using JustRelax

model = PS_Setup(:cpu, Float64, 3)
environment!(model)

include("../miniapps/benchmarks/thermal_diffusion/diffusion/diffusion3D.jl")

function check_convergence_case1()
    nx = ny = nz = 32
    L = 100e3
    _, _, iters = diffusion_3D(;
        nx=nx,
        ny=ny,
        nz=nz,
        lx=L,
        ly=L,
        lz=L,
        ρ=3.3e3,
        Cp=1.2e3,
        K=3.0,
        init_MPI=MPI.Initialized() ? false : true,
        finalize_MPI=false,
    )
    iters_expected = (
        iter=7,
        err_evo1=[
            1.357515229623902e-6,
            5.966784784078521e-7,
            2.5985987041627143e-7,
            1.1411039300743566e-7,
            4.9739497068255585e-8,
            2.1825268358281862e-8,
            9.52007457847057e-9,
        ],
        err_evo2=[1, 2, 3, 4, 5, 6, 7],
    )
    println(iters)
    println(iters_expected)
    return iters.iter == iters_expected.iter &&
           iters.err_evo1[end] ≈ iters_expected.err_evo1[end]
end

@testset begin
    @test check_convergence_case1()
end
