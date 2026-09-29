push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
else
    @init_parallel_stencil(Threads, Float64, 3)
end

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

# HELPER FUNCTIONS ---------------------------------------------------------------
@parallel_indices (i, j, k) function init_T!(T, z)
    T[i, j, k + 1] = z[k] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, zc, r, xci)

    @parallel_indices (i, j, k) function _elliptical_perturbation!(T, x, y, z)
        if (((x[i] - xc))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
            T[(i, j, k) .+ 1...] += δT
        end
        return nothing
    end
    ni = size(T) .- 2
    return @parallel (@idx ni) _elliptical_perturbation!(T, xci...)
end

function diffusion_3D(;
        nx = 32,
        ny = 32,
        nz = 32,
        lx = 100.0e3,
        ly = 100.0e3,
        lz = 100.0e3,
        ρ0 = 3.3e3,
        Cp0 = 1.2e3,
        K0 = 3.0,
        init_MPI = JustRelax.MPI.Initialized() ? false : true,
        finalize_MPI = true,
    )

    kyr = 1.0e3 * 3600 * 24 * 365.25
    Myr = 1.0e6 * 3600 * 24 * 365.25
    ttot = 1 * Myr # total simulation time
    dt = 50 * kyr # physical time step

    # Physical domain
    ni = (nx, ny, nz)
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    origin = 0, 0, -lz # nodes at the center and vertices of the cells
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_MPI)...) # init MPI
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    # Define the thermal parameters with GeoParams
    rheology = SetMaterialParams(;
        Phase = 1,
        Density = PT_Density(; ρ0 = 3.1e3, β = 0.0, T0 = 0.0, α = 1.5e-5),
        HeatCapacity = ConstantHeatCapacity(; Cp = Cp0),
        Conductivity = ConstantConductivity(; k = K0),
    )

    # fields needed to compute density on the fly
    P = @zeros(ni...)

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(backend, ni)
    thermal.H .= 1.0e-6
    # physical parameters
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ

    # Boundary conditions
    Ttop = 300.0
    Tbot = 3500.0
    pt_thermal = PTThermalCoeffs(backend, K, ρCp, dt, di, li; CFL = 0.95 / √3.1)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false, front = true, back = true),
        constant_value = (left = true, right = true, top = Ttop, bot = Tbot, front = true, back = true),
    )

    @parallel (1:(nx + 2), 1:(ny + 2), 1:nz) init_T!(thermal.T, xci[3])

    # Add thermal perturbation
    δT = 100.0e0 # thermal perturbation
    r = 10.0e3 # thermal perturbation radius
    center_perturbation = lx / 2, ly / 2, -lz / 2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xci)

    t = 0.0
    it = 0

    # Physical time loop
    while it < 10
        args = (; P = P, T = thermal.T)
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs = (;
                igg,
                verbose = false,
            ),
        )

        t += dt
        it += 1
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return thermal
end

@testset "heatdiffusion_PT! runs on a non-uniform grid" begin
    # src/thermal_diffusion/DiffusionPT_solver.jl: update_T!/check_res! must divide the
    # flux divergence by the thermal cell width, `grid._di.vertex` (one entry per cell).
    # `grid._di.center` is one entry shorter (the gaps between interior cell centers)
    # and throws a BoundsError as soon as the grid spacing is a genuine vector instead
    # of a repeated scalar.
    xv = [0.0, 0.2, 0.5, 0.6, 1.0, 1.3] .* 1.0e3
    yv = [0.0, 0.1, 0.35, 0.45, 0.7] .* 1.0e3
    zv = [0.0, 0.3, 0.4, 0.8] .* 1.0e3
    nx, ny, nz = length(xv) - 1, length(yv) - 1, length(zv) - 1
    ni = (nx, ny, nz)
    li = (xv[end] - xv[1], yv[end] - yv[1], zv[end] - zv[1])
    di = (minimum(diff(xv)), minimum(diff(yv)), minimum(diff(zv)))
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_mpi)...)
    grid = Geometry(PTArray(backend), xv, yv, zv)

    thermal = ThermalArrays(backend, ni)
    thermal.T .= 1000.0
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = true, bot = true, front = true, back = true),
    )
    K = @fill(3.0, ni...)
    ρCp = @fill(3.1e3 * 1.2e3, ni...)
    dt = 1.0e3
    pt_thermal = PTThermalCoeffs(backend, K, ρCp, dt, di, li; CFL = 0.95 / √3.1)

    try
        @suppress begin
            result = heatdiffusion_PT!(
                thermal,
                pt_thermal,
                thermal_bc,
                K,
                ρCp,
                dt,
                grid;
                kwargs = (; igg = igg, nout = 1, iterMax = 20, verbose = false),
            )
            @test !isempty(result.iter_count)
        end
    finally
        finalize_global_grid(; finalize_MPI = false)
    end
end

# @testset "Diffusion_3D" begin
#     @suppress begin
#         nx = 32
#         ny = 32
#         nz = 32
#         thermal = diffusion_3D(; nx = nx, ny = ny, nz = nz)
#         if backend == CPUBackend
#             @test thermal.T[Int(ceil(nx / 2)), Int(ceil(ny / 2)), Int(ceil(nz / 2))] ≈ 1813.2470160788096 rtol = 1.0e-3
#             @test (@view thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)])[Int(ceil(nx / 2)), Int(ceil(ny / 2)), Int(ceil(nz / 2))] ≈ 1831.2568044653274 rtol = 1.0e-3
#         else
#             @test true == true
#         end
#     end
# end
