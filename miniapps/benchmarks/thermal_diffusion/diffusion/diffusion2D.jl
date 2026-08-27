const isCUDA = false
# const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D
using Pkg; Pkg.activate("miniapps")

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, CellArrays


# using JustPIC
# const backend = JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend

@parallel_indices (i, j) function init_T!(T, z)
    if z[j] == maximum(z)
        T[i + 1, j + 1] = 300.0
    elseif z[j] == minimum(z)
        T[i + 1, j + 1] = 3500.0
    else
        T[i + 1, j + 1] = z[j] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _elliptical_perturbation!(T, δT, xc, yc, r, x, y)
        if (((x[i] - xc))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i + 1, j + 1] += δT
        end
        return nothing
    end
    ni = size(T) .- 2
    return @parallel (@idx ni) _elliptical_perturbation!(T, δT, xc, yc, r, xvi...)
end

function diffusion_2D(; nx = 32, ny = 32, lx = 100.0e3, ly = 100.0e3, ρ0 = 3.3e3, Cp0 = 1.2e3, K0 = 3.0)
    kyr = 1.0e3 * 3600 * 24 * 365.25
    Myr = 1.0e3 * kyr
    ttot = 1 * Myr # total simulation time
    dt = 50 * kyr # physical time step
    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    # Physical domain
    ni = (nx, ny)
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    origin = 0, -ly
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
    thermal = ThermalArrays(backend, ni)
    thermal.H .= 1.0e-6 # radiogenic heat production
    # physical parameters
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ

    pt_thermal = PTThermalCoeffs(backend, K, ρCp, dt, di, li)
    Ttop, Tbot = 300.0, 3500.0
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
        constant_value = (left = false, right = false, top = Ttop, bot = Tbot),
    )
    @parallel (1:(nx + 2), 1:ny) init_T!(thermal.T, xci[2])

    # Add thermal perturbation
    δT = 100.0e0 # thermal perturbation
    r = 10.0e3 # thermal perturbation radius
    center_perturbation = lx / 2, -ly / 2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xci)

    # Time loop
    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    while it < nt
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            grid;
            kwargs = (;
                verbose = false,
            ),
        )

        t += dt
        it += 1
    end

    return thermal
end

diffusion_2D()
