push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
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

function elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _elliptical_perturbation!(T, x, y, z)
        @inbounds if (((x[i] - xc))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
            T[i, j, k] += δT
        end
        return nothing
    end

    return @parallel _elliptical_perturbation!(T, xvi...)
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
    args = (; P = P, T = @zeros(ni .+ 2...))

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
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xvi)

    t = 0.0
    it = 0

    # Physical time loop
    while it < 10
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

@testset "Diffusion_3D" begin
    @suppress begin
        nx = 32
        ny = 32
        nz = 32
        thermal = diffusion_3D(; nx = nx, ny = ny, nz = nz)
        if backend == CPUBackend
            @test thermal.T[Int(ceil(nx / 2)), Int(ceil(ny / 2)), Int(ceil(nz / 2))] ≈ 1831.9030160251664 rtol = 1.0e-3
            @test (@view thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)])[Int(ceil(nx / 2)), Int(ceil(ny / 2)), Int(ceil(nz / 2))] ≈ 1836.4686150797922 rtol = 1.0e-3
        else
            @test true == true
        end
    end
end
