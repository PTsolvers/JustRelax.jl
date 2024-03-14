push!(LOAD_PATH, "..")

using Test, Suppressor
using JustRelax
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("../miniapps/benchmarks/stokes2D/elastic_buildup/Elastic_BuildUp.jl")

function check_convergence_case1()
    # model specific parameters
    nx, ny = 32, 32
    lx, ly = 100e3, 100e3 # length of the domain in meters
    endtime = 10 # duration of the model in kyrs
    η0 = 1e21 # viscosity
    εbg = 1e-14 # background strain rate (pure shear boundary conditions)
    G = 10e9 # shear modulus
    # run model
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    _, _, av_τyy, sol_τyy, t, = elastic_buildup(;
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        endtime=endtime,
        η0=η0,
        εbg=εbg,
        G=G,
        init_MPI=init_mpi,
        finalize_MPI=false,
    );

    err =
        sum(abs(abs.(av_τyy[i]) - sol_τyy[i]) / sol_τyy[i] for i in eachindex(av_τyy)) /
        length(av_τyy)

    println("mean error $err")
    return err ≤ 5e-3
end

@testset "Elastic Build-Up" begin
    @suppress begin
        @test check_convergence_case1()
    end
end
