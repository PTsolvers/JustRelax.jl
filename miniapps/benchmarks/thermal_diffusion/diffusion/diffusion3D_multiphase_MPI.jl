using JustRelax, JustRelax.JustRelax3D
using Pkg; Pkg.activate("miniapps")


const backend_JR = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

using JustPIC
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPU # Options: CPUBackend, CUDABackend, AMDGPUBackend

using GeoParams, CairoMakie

@parallel_indices (i, j, k) function init_T!(T, z, lz)
    T[i, j, k + 1] = z[k] * (1900.0 - 1600.0) / (-lz) + 1600.0
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
    ni = nx, ny, nz
    li = lx, ly, lz  # domain length in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI = init_MPI)...) # init MPI
    di = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    origin = 0, 0, -lz # nodes at the center and vertices of the cells
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
    Ttop = 300.0
    Tbot = 3500.0
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false, front = true, back = true),
        constant_value = (left = true, right = true, top = Ttop, bot = Tbot, front = true, back = true),
    )

    @parallel (1:(nx + 2), 1:(ny + 2), 1:nz) init_T!(thermal.T, xci[3], lz)

    # Add thermal perturbation
    δT = 100.0e0 # thermal perturbation
    r = 10.0e3 # thermal perturbation radius
    center_perturbation = lx / 2, ly / 2, -lz / 2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xci)
    update_halo!(thermal.T)
    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 20, 20, 1
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    pPhases, = init_cell_arrays(particles, Val(1))
    particle_args = (pPhases)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, center_perturbation..., r)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    update_cell_halo!(particles.coords..., particle_args)
    update_cell_halo!(particles.index)
    # ----------------------------------------------------

    # PT coefficients for thermal diffusion
    args = (; P = P, T = thermal.T)
    pt_thermal = PTThermalCoeffs(backend_JR, K, ρCp, dt, di, li; CFL = 0.75 / √3.1)

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    # Visualization global arrays
    ni_v = ni * igg.dims
    T_v = zeros(ni_v...)
    # local array without halo
    T_nohalo = zeros(ni...)

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
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )

        @views T_nohalo .= Array(thermal.T[2:(end - 1), 2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)

        if igg.me == 0
            slice_j = ny_v >>> 1
            fig, = heatmap(T_v[:, slice_j, :])
            save("temperature_3D_it_$(it)_MPI.png", fig)
            println("\n SAVED TEMPERATURE \n")
        end

        t += dt
        it += 1
    end

    finalize_global_grid(; finalize_MPI = true)

    return thermal
end


diffusion_3D()
