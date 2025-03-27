using JustRelax, JustRelax.JustRelax3D


const backend_JR = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

using JustPIC, JustPIC._3D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GeoParams

@parallel_indices (i, j, k) function init_T!(T, z)
    if z[k] == maximum(z)
        T[i, j, k] = 300.0
    elseif z[k] == minimum(z)
        T[i, j, k] = 3500.0
    else
        T[i, j, k] = z[k] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
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

function init_phases!(phases, particles, xc, yc, zc, r)
    ni = size(phases)
    center = xc, yc, zc

    @parallel_indices (I...) function init_phases!(phases, px, py, pz, index, center, r)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, I...]) == 0 && continue

            x = @index px[ip, I...]
            y = @index py[ip, I...]
            z = @index pz[ip, I...]

            # plume - rectangular
            if (((x - center[1]))^2 + ((y - center[2]))^2 + ((z - center[3]))^2) ≤ r^2
                @index phases[ip, I...] = 2.0

            else
                @index phases[ip, I...] = 1.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, center, r)
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
        finalize_MPI = false,
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
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            Density = PT_Density(; ρ0 = 3.0e3, β = 0.0, T0 = 0.0, α = 1.5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp0),
            Conductivity = ConstantConductivity(; k = K0),
            RadioactiveHeat = ConstantRadioactiveHeat(1.0e-6),
        ),
        SetMaterialParams(;
            Phase = 2,
            Density = PT_Density(; ρ0 = 3.3e3, β = 0.0, T0 = 0.0, α = 1.5e-5),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp0),
            Conductivity = ConstantConductivity(; k = K0),
            RadioactiveHeat = ConstantRadioactiveHeat(1.0e-7),
        ),
    )

    # fields needed to compute density on the fly
    P = @zeros(ni...)
    args = (; P = P)

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(backend_JR, ni)
    thermal.H .= 1.0e-6
    # physical parameters
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ

    # Boundary conditions
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false, front = true, back = true),
    )

    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[3])

    # Add thermal perturbation
    δT = 100.0e0 # thermal perturbation
    r = 10.0e3 # thermal perturbation radius
    center_perturbation = lx / 2, ly / 2, -lz / 2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xvi)

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 20, 1
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi...
    )
    pPhases, = init_cell_arrays(particles, Val(1))
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, center_perturbation..., r)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # PT coefficients for thermal diffusion
    args = (; P = P, T = thermal.Tc)
    pt_thermal = PTThermalCoeffs(backend_JR, K, ρCp, dt, di, li; CFL = 0.75 / √3.1)

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    # Physical time loop
    while it < nt
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )

        t += dt
        it += 1
    end

    finalize_global_grid(; finalize_MPI = finalize_MPI)

    return (ni = ni, xci = xci, li = li, di = di), thermal
end

diffusion_3D()
