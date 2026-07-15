using Pkg; Pkg.activate("miniapps")

# choose benchmark
benchmark = :solcx

# model resolution (number of gridpoints), used for runtype = :single
nx, ny = 63, 63

# :single for a single run model with nx, ny resolution
# :multiple for a grid sensitivity error sweep (saves h/L2_vx/L2_vy/L2_p to a jld2
# file next to the benchmark script, read by ../convergence_comparison_figure.jl)
runtype = :single

if benchmark == :solcx
    # need to `include` again if you switch benchmark within the same session
    include("solcx/SolCx_DYREL.jl")

    Δη = 1.0e6 # viscosity contrast
    if runtype == :single
        geometry, stokes, iters = solCx_DYREL(;
            Δη = Δη, nx = nx, ny = ny,
            init_MPI = !JustRelax.MPI.Initialized(), finalize_MPI = false,
            figdir = "SolCx_DYREL",
        )
    elseif runtype == :multiple
        f = multiple_solCx_DYREL(; Δη = Δη, nrange = 6:10) # nx = ny = 2^nrange - 1
    end

elseif benchmark == :solkz
    include("solkz/SolKz_DYREL.jl")

    Δη = 1.0e6 # viscosity contrast
    if runtype == :single
        geometry, stokes, iters = solKz_DYREL(;
            Δη = Δη, nx = nx, ny = ny,
            init_MPI = !JustRelax.MPI.Initialized(), finalize_MPI = false,
            figdir = "SolKz_DYREL",
        )
    elseif runtype == :multiple
        f = multiple_solKz_DYREL(; Δη = Δη, nrange = 4:10) # nx = ny = 2^nrange - 1
    end

elseif benchmark == :solvi
    include("solvi/SolVi_DYREL.jl")

    Δη = 1.0e-3 # viscosity ratio between matrix and inclusion
    rc = 0.2    # radius of the inclusion
    εbg = 1.0e0 # background strain rate
    lx, ly = 2.0e0, 2.0e0
    if runtype == :single
        geometry, stokes, iters = solVi_DYREL(;
            Δη = Δη, nx = nx, ny = ny, lx = lx, ly = ly, rc = rc, εbg = εbg,
            init_MPI = !JustRelax.MPI.Initialized(), finalize_MPI = false,
            figdir = "SolVi_DYREL",
        )
    elseif runtype == :multiple
        f = multiple_solVi_DYREL(; Δη = Δη, lx = lx, ly = ly, rc = rc, εbg = εbg, nrange = 4:8) # nx = ny = 2^nrange - 1
    end

else
    throw("Benchmark not available.")
end
