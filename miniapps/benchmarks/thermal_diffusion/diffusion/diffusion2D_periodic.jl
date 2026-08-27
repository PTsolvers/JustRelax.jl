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
using GeoParams


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

function circular_perturbation!(T, δT, xc, yc, r, xci)
    @parallel_indices (i, j) function _circular_perturbation!(T, x, y)
        if (x[i] - xc)^2 + (y[j] - yc)^2 ≤ r^2
            T[i + 1, j + 1] += δT
        end
        return nothing
    end
    ni = size(T) .- 2
    return @parallel (@idx ni) _circular_perturbation!(T, xci...)
end

function diffusion_2D_periodic(;
        nx = 32, ny = 32, lx = 100.0e3, ly = 100.0e3,
        ρ0 = 3.3e3, Cp0 = 1.2e3, K0 = 3.0,
    )
    kyr = 1.0e3 * 3600 * 24 * 365.25
    dt = 50 * kyr
    nt = 20
    init_mpi = !JustRelax.MPI.Initialized()
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    ni = (nx, ny)
    li = (lx, ly)
    di = @. li / ni
    grid = Geometry(ni, li; origin = (0, -ly))
    (; xci) = grid

    rheology = SetMaterialParams(;
        Phase = 1,
        Density = PT_Density(; ρ0 = 3.1e3, β = 0.0, T0 = 0.0, α = 1.5e-5),
        HeatCapacity = ConstantHeatCapacity(; Cp = Cp0),
        Conductivity = ConstantConductivity(; k = K0),
    )
    thermal = ThermalArrays(backend, ni)
    thermal.H .= 1.0e-6
    K = @fill(K0, ni...)
    ρCp = @fill(ρ0 * Cp0, ni...)
    args = (; P = @zeros(ni...), T = @zeros(ni .+ 2...))
    pt_thermal = PTThermalCoeffs(backend, K, ρCp, dt, di, li)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = false, right = false, top = false, bot = false),
        constant_value = (left = false, right = false, top = 300.0, bot = 3500.0),
        periodic = (left = true, right = true, top = false, bot = false),
    )

    @parallel (1:nx, 1:ny) init_T!(thermal.T, xci[2])
    thermal_bcs!(thermal, thermal_bc)
    circular_perturbation!(thermal.T, 100.0, lx / 2, -ly / 2, 10.0e3, xci)

    for _ in 1:nt
        heatdiffusion_PT!(
            thermal, pt_thermal, thermal_bc, rheology, args, dt, grid;
            kwargs = (; igg, verbose = false),
        )
    end

    return thermal
end

diffusion_2D_periodic()
