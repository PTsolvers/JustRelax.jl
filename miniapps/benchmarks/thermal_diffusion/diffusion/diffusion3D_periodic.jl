using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

using GeoParams
using JustRelax, JustRelax.JustRelax3D
using Pkg; Pkg.activate("miniapps")

const backend = CPUBackend

@parallel_indices (i, j, k) function init_T!(T, z, lz)
    T[i, j, k + 1] = z[k] * (1900.0 - 1600.0) / (-lz) + 1600.0
    return nothing
end

function spherical_perturbation!(T, δT, xc, yc, zc, r, xci)
    @parallel_indices (i, j, k) function _spherical_perturbation!(T, x, y, z)
        if (x[i] - xc)^2 + (y[j] - yc)^2 + (z[k] - zc)^2 ≤ r^2
            T[(i, j, k) .+ 1...] += δT
        end
        return nothing
    end
    ni = size(T) .- 2
    return @parallel (@idx ni) _spherical_perturbation!(T, xci...)
end

function diffusion_3D_periodic(;
        nx = 32, ny = 32, nz = 32,
        lx = 100.0e3, ly = 100.0e3, lz = 100.0e3,
        ρ0 = 3.3e3, Cp0 = 1.2e3, K0 = 3.0,
        init_MPI = !JustRelax.MPI.Initialized(), finalize_MPI = false,
    )
    kyr = 1.0e3 * 3600 * 24 * 365.25
    dt = 50 * kyr
    nt = 20
    ni = (nx, ny, nz)
    li = (lx, ly, lz)
    di = @. li / ni
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_MPI)...)
    grid = Geometry(ni, li; origin = (0, 0, -lz))
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
    args = (; P = @zeros(ni...), T = thermal.T)
    pt_thermal = PTThermalCoeffs(backend, K, ρCp, dt, di, li; CFL = 0.95 / √3.1)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (
            left = false, right = false, front = false, back = false,
            top = false, bot = false,
        ),
        constant_value = (
            left = false, right = false, front = false, back = false,
            top = 300.0, bot = 3500.0,
        ),
        periodic = (
            left = true, right = true, front = true, back = true,
            top = false, bot = false,
        ),
    )

    @parallel (1:nx, 1:ny, 1:nz) init_T!(thermal.T, xci[3], lz)
    thermal_bcs!(thermal, thermal_bc)
    spherical_perturbation!(thermal.T, 100.0, lx / 2, ly / 2, -lz / 2, 10.0e3, xci)

    for _ in 1:nt
        heatdiffusion_PT!(
            thermal, pt_thermal, thermal_bc, rheology, args, dt, grid;
            kwargs = (; igg, verbose = false),
        )
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)
    return (ni = ni, xci = xci, li = li, di = di), thermal
end

diffusion_3D_periodic()
