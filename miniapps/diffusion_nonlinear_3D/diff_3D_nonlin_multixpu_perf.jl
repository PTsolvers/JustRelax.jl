# TODO if nothing better, use the dumb duplicative way to support the GPU backend, as in ParllelRandomFields.jl
# TODO run it on a GPU machine (geocomp1 seems easiest) and confirm perf
# TODO add a reference test that calls this
# TODO add a perf test
# TODO run the tests locally and on a GPU machine
# TODO run the tests with CI (crib from ParallelKernels perhaps)


# This is a replication of the solver in
# scripts/diff_3D
# at https://github.com/PTSolvers/PseudoTransientDiffusion.jl,
#
# It should produce numerically identical results and equally-good performance.
#
# Recall that to check perf, do something like
# julia --project=. --check-bounds=no -O3 diff_3D_nonlin_multixpu_perf.jl

const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : false

const do_viz  = haskey(ENV, "DO_VIZ")  ? parse(Bool, ENV["DO_VIZ"])  : false
const do_save = haskey(ENV, "DO_SAVE") ? parse(Bool, ENV["DO_SAVE"]) : false
const do_save_viz = haskey(ENV, "DO_SAVE_VIZ") ? parse(Bool, ENV["DO_SAVE_VIZ"]) : false

const nx_in = haskey(ENV, "NX") ? parse(Int, ENV["NX"]) : 64
const ny_in = haskey(ENV, "NY") ? parse(Int, ENV["NY"]) : 64
const nz_in = haskey(ENV, "NZ") ? parse(Int, ENV["NZ"]) : 64

# Packages used for the "grid" (Stage 2)
using ImplicitGlobalGrid  # Note that ParallelStencil is only used within the solver, hence not here
import MPI
@static if USE_GPU
    using JustRelax.DiffusionNonlinearSolvers_CUDA_Float64_3D
else
    using JustRelax.DiffusionNonlinearSolvers_Threads_Float64_3D
end

using ParallelStencil
# ParallelStencil forces you to use immediate initialization with known-at-parse-time
# values passed into @init_parallel_stencil
# FIXME: this is awkward and ruins how Julia is supposed to work. ParallelStencil should not force the user to hard code these parameters, if at all possible.
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

# Packages used by the application for output
@static if do_viz || do_save_viz
  using Plots, MAT
end
using Printf

# A misc. macro used for visualization
@views inn(A) = A[2:end-1,2:end-1,2:end-1]

@views function diffusion_3D()

    # This main function for "Application" code is divided into 7 stages which
    # are intended to cover usage with the GPU4GEO project and potential users
    # of the software developed within it.

    # 1. Quantities needed to describe "where the problem lives", in terms of (parallel) topology
    # 2. Initialize tools which can represent this domain concretely in parallel (IGG here, could be PETSc/DM)
    # 3. Concrete representations of data and population of values
    #    - Includes information on embedding/coordinates
    # 4. Tools, dependent on the data representation, to actually solve a particular physical problem (here JustRelax.jl, but could be PETSc's SNES)
    #    - Note that here, the physical timestepping scheme is baked into this "physical problem"
    # 5. Analysis and output which depends on the details of the solver
    # 6. "Application" Analysis and output which does not depend on the details of the solver
    # 7. Finalization/Cleanup

    # In a real application, steps 4., 5., and 6. will likely be repeated
    # multiple times and be interspersed with other logic (e.g. a particle advection step).

    # Note: CompGrids.jl combines 1. 2., and part of 3. (coordinates, and the identity of the fields)

    # 1. Description of problem setting
    # This could be further decomposed into
    # information describing
    # - the global continuous problem domain's topology
    # - the global discretization (discrete topology)
    # - the definition of overlapping local patches (the "atlas")
    # But for regular grids this likely isn't worth it.

    # Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    nx, ny, nz = nx_in, ny_in, nz_in
    # Additional information about boundary conditions would logically go here

    # (Physical) Time domain and discretization
    ttot = 0.4              # total simulation time
    dt = 0.2              # physical time step

    # 2. Parallel representation
    # This is constructed based on the more abstract information in the previous stage.
    # This is where the details of the hardware come into play. Which actual RAM does it live in?
    # (host or device, and on how many nodes/GPUs?)
    # Here, this means that we introduce both ParallelStencil and ImplicitGlobalGrid,
    # as both are required to
    me, dims, nprocs = init_global_grid(nx, ny, nz) # MPI initialisation
    @static if USE_GPU
        select_device()
    end    # select one GPU per MPI local rank (if >1 GPU per node)


    # 3. Instantiation and population of unknown and coefficient fields

    # We think of additional information about the coordinates (that is, the way the discrete mesh
    # is embedded into R^3) as just another field. Here, that field
    # can be defined implicitly with a few parameters for a regular grid.
    lx, ly, lz = 10.0, 10.0, 10.0                      # domain sizes
    dx, dy, dz = lx / nx_g(), ly / ny_g(), lz / nz_g() # cell sizes

    # We only have one solution field for this problem. Additional solution fields
    # as well as coefficient fields would be defined here.
    H = zeros(nx, ny, nz)
    H = Data.Array([exp(-(x_g(ix, dx, H) - 0.5 * lx + dx / 2) * (x_g(ix, dx, H) - 0.5 * lx + dx / 2) - (y_g(iy, dy, H) - 0.5 * ly + dy / 2) * (y_g(iy, dy, H) - 0.5 * ly + dy / 2) - (z_g(iz, dz, H) - 0.5 * lz + dz / 2) * (z_g(iz, dz, H) - 0.5 * lz + dz / 2)) for ix = 1:size(H, 1), iy = 1:size(H, 2), iz = 1:size(H, 3)])


    # (Pre) 5./6.
    # This is before the solve to fail earlier on OOM, but could be
    # moved after the solve without modification.
    if do_viz || do_save_viz
        if (me == 0)
            ENV["GKSwstype"] = "nul"
            if do_viz
                !ispath("../../figures") && mkdir("../../figures")
            end
        end
        nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2], (nz - 2) * dims[3]
        if (nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory())
            error("Not enough memory for visualization.")
        end
        H_v = zeros(nx_v, ny_v, nz_v) # global array for visu
        H_inn = zeros(nx - 2, ny - 2, nz - 2) # no halo local array for visu
        z_sl = Int(ceil(nz_g() / 2))     # Central z-slice
        Xi_g, Yi_g = dx+dx/2:dx:lx-dx-dx/2, dy+dy/2:dy:ly-dy-dy/2 # inner points only
        xc, yc, zc = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny), LinRange(dz / 2, lz - dz / 2, nz)
    end

    # 4. Invoke a method to solve a particular physical problem (repeatedly)
    # JustRelax.jl exists to provide these solvers!
    # Note that we do not attempt to define some solver-independent description
    # of the physical problem

    solver = DiffusionNonlinearSolver(
        nx, ny, nz,                # 1.
        me, dims, nprocs,          # 2.
        lx, ly, lz, dx, dy, dz, dt # 3.
        # Coefficient fields from 3. could go here
    )

    # Physical time loop
    t = 0.0
    it = 0
    ittot = 0
    nt = Int(ceil(ttot / dt))
    niter = 0  # for perf
    first_solve = true
    while it < nt
        if (it == 1)
            # GC.gc()  # could include this here
            tic()
            niter = 0
        end
        # Solver, with H as an explicit argument to emphasize that H is being updated
        iter = solve!(solver, H, first_solve)
        first_solve = false
        ittot += iter
        it += 1
        t += dt
        niter += iter
    end
    t_toc = toc()

    # 5. Output and analysis depending on the details of the solver
    # We make this division explicit as some envisioned applications will
    # want to be able to use a different solver in the previous stage.
    A_eff = (2 * 4 + 2) / 1e9 * nx * ny * nz * sizeof(Data.Number) # Effective main memory access per iteration [GB]
    t_it = t_toc / niter                                           # Execution time per iteration [s]
    T_eff = A_eff / t_it                                           # Effective memory throughput [GB/s]
    if (me == 0)
        @printf("PERF: Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits = 3), niter)
    end
    if (me == 0)
        @printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits = 2), it, nx_g(), ittot)
    end

    # Wrap up any solver-specific information we want to include in the "application" output
    if do_viz
        solver_info_string = "iters=$ittot"
    end

    # 6. "Application" postprocessing and output which does not depend on the details of the solver.
    # Note that we still allow information from the solver to be displayed here
    # by standardizing it in such a way that swapping out the solver can be accomplished only by changing
    # the previous 2 stages, not anything in this stage.

    # Visualise
    if do_viz || do_save_viz
        H_inn .= inn(H)
        gather!(H_inn, H_v)
        if solver.me == 0 && do_viz
            p = heatmap(Xi_g, Yi_g, H_v[:, :, z_sl]', dpi = 150, aspect_ratio = 1, framestyle = :box, xlims = (Xi_g[1], Xi_g[end]), ylims = (Yi_g[1], Yi_g[end]), xlabel = "lx", ylabel = "ly", c = :viridis, clims = (0, 1), title = "nonlinear diffusion (nt=$it, $solver_info_string)")
            display(p)
            savefig("../../figures/diff_3D_nonlin_multixpu_perf_$(nx_g()).png")
        end
    end

    if me == 0 && do_save
        !ispath("../../output") && mkdir("../../output")
        open("../../output/out_diff_3D_nonlin_multixpu_perf.txt", "a") do io
            println(io, "$(nprocs) $(nx_g()) $(ny_g()) $(nz_g()) $(ittot) $(t_toc) $(A_eff) $(t_it) $(T_eff)")
        end
    end
    if me == 0 && do_save_viz
        !ispath("../../out_visu") && mkdir("../../out_visu")
        matwrite("../../out_visu/diff_3D_nonlin_multixpu_perf.mat", Dict("H_3D" => Array(H_v), "xc_3D" => Array(xc), "yc_3D" => Array(yc), "zc_3D" => Array(zc)); compress = true)
    end

    # 7. Finalization
    finalize_global_grid()
    return
end

diffusion_3D()
