using LinearAlgebra, CairoMakie
using JustRelax, JustRelax.JustRelax3D
using MPI: MPI

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

const backend_JR = CPUBackend

# choose benchmark
benchmark = :Burstedde

# model resolution (number of gridpoints)
nx, ny, nz = 16, 16, 16

# set MPI
finalize_MPI = false

if benchmark == :taylorGreen
    # benchmark reference:
    #   FVCA8 benchmark for the Stokes and Navier-Stokes
    #       equations with the TrioCFD code – benchmark session
    #       P.-E. Angeli, M.-A. Puscas, G. Fauchet, A. Cartalade
    #   HAL Id: cea-02434556
    #   https://hal-cea.archives-ouvertes.fr/cea-02434556

    # include benchmark
    include("taylor_green/TaylorGreen.jl")

    # run benchmark
    geometry, stokes, iters = taylorGreen(;
        nx = nx,
        ny = ny,
        nz = nz,
        init_MPI = MPI.Initialized() ? false : true,
        finalize_MPI = finalize_MPI,
    )

    # plot results
    f = plot(stokes, geometry; cmap = :vik)

    # compute error
    L2_p, L2_vx, L2_vy, L2_vz = error(stokes, geometry)

elseif benchmark == :Burstedde
    # benchmark reference:
    #   C. Burstedde, G. Stadler, L. Alisic, L. C. Wilcox, E. Tan, M. Gurnis, and O. Ghattas.
    #   Large-scale adaptive mantle convection simulation. Geophysical Journal International, 2013

    # include benchmark
    include("burstedde/Burstedde.jl")

    # run benchmark
    geometry, stokes, iters = burstedde(;
        nx = nx,
        ny = ny,
        nz = nz,
        init_MPI = MPI.Initialized() ? false : true,
        finalize_MPI = finalize_MPI,
    )

    # plot results
    f = plot(stokes, geometry; cmap = :vik)

    # compute error
    L2_p, L2_vx, L2_vy, L2_vz = error(stokes, geometry)

elseif benchmark == :solvi
    # Benchmark reference:
    #   D. W. Schmid and Y. Y. Podladchikov. Analytical solutions for deformable elliptical inclusions in
    #   general shear. Geophysical Journal International, 155(1):269–288, 2003.

    # include benchmark
    include("solvi/solVi3D.jl") # need to call this again if we switch from gpu <-/-> cpu

    # model specific parameters
    Δη = 1.0e-3 # viscosity ratio between matrix and inclusion
    rc = 1.0e0 # radius of the inclusion
    εbg = 1.0e0 # background strain rate
    lx, ly, lz = 1.0e1, 1.0e1, 1.0e1 # domain siye in x and y directions

    # run model
    geometry, stokes, iters = solVi3D(;
        Δη = Δη,
        nx = nx,
        ny = ny,
        nz = nz,
        lx = lx,
        ly = ly,
        lz = lz,
        rc = rc,
        εbg = εbg,
        init_MPI = MPI.Initialized() ? false : true,
        finalize_MPI = finalize_MPI,
    )
    # plot results
    f = plot(stokes, geometry, rc; cmap = :vik)

else
    throw("Benchmark not available.")
end
