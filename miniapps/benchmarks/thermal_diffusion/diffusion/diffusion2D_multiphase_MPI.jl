using JustRelax, JustRelax.JustRelax2D
const backend_JR = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)  #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using ImplicitGlobalGrid
using MPI: MPI

using GeoParams, CairoMakie
using JustPIC, JustPIC._2D
const backend = JustPIC.CPUBackend

distance(p1, p2) = mapreduce(x -> (x[1] - x[2])^2, +, zip(p1, p2)) |> sqrt

@parallel_indices (i, j) function init_T!(T, z, ly)
    T[i, j + 1] = -z[j] * (1900.0 - 1600.0) / ly + 1600.0
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

function init_phases!(phases, particles, xc, yc, r)
    ni = size(phases)
    center = xc, yc

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, center, r)
        @inbounds for ip in cellaxes(phases)
            # quick escape
            @index(index[ip, i, j]) == 0 && continue

            x = @index px[ip, i, j]
            y = @index py[ip, i, j]

            # plume - rectangular
            if (((x - center[1]))^2 + ((y - center[2]))^2) ≤ r^2
                @index phases[ip, i, j] = 2.0
            else
                @index phases[ip, i, j] = 1.0
            end
        end
        return nothing
    end

    return @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, center, r)
end

function diffusion_2D(igg, figdir; nx = 32, ny = 32, lx = 100.0e3, ly = 100.0e3, Cp0 = 1.2e3, K0 = 3.0)
    kyr = 1.0e3 * 3600 * 24 * 365.25
    Myr = 1.0e3 * kyr
    ttot = 1 * Myr # total simulation time
    dt = 50 * kyr # physical time step

    # Physical domain
    ni = nx, ny
    li = lx, ly  # domain length in x- and y-
    origin = 0.0, -ly
    # igg = IGG(init_global_grid(nx, ny, 1; init_MPI = true)...) #init MPI
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
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

    # Allocate arrays needed for every Thermal Diffusion
    thermal = ThermalArrays(backend_JR, ni)
    Ttop = 300.0
    Tbot = 3500.0
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
        constant_value = (left = true, right = true, top = Ttop, bot = Tbot),
    )
    @parallel (1:nx+2, 1:ny) init_T!(thermal.T, xci[2], ly)
    thermal_bcs!(thermal, thermal_bc)
    update_halo!(thermal.T)

    # Add thermal perturbation
    δT = 100.0e0 # thermal perturbation
    r = 10.0e3 # thermal perturbation radius
    center_perturbation = lx / 2, -ly / 2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xci)

    update_halo!(thermal.T)
    thermal_bcs!(thermal, thermal_bc)

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 40, 1
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, grid.xi_vel...
    )
    pPhases, = init_cell_arrays(particles, Val(1))
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles, center_perturbation..., r)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    # update_halo!(particles.coords..., pPhases, particles.index)
    update_cell_halo!(particles.coords..., pPhases)
    update_cell_halo!(particles.index)
    # ----------------------------------------------------

    # PT coefficients for thermal diffusion
    args = (; P = P, T = thermal.T)
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 0.65 / √2
    )

    # Time loop
    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    # global array
    nx_v = (nx) * igg.dims[1]
    ny_v = (ny) * igg.dims[2]
    T_v = zeros(nx_v, ny_v)
    # local array without halo
    T_nohalo = zeros(nx, ny)
    # Time loop
    ## IO -----------------------------------------------
    take(figdir)
    # ---------------------------------------------------

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
                igg = igg,
                phase = phase_ratios,
                iterMax = 1.0e3,
                nout = 1.0e2,
            )
        )

        it += 1
        t += dt

        @views T_nohalo .= Array(thermal.T[2:(end - 1), 2:(end - 1)]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)

        if igg.me == 0
            fig, = heatmap(T_v, colorrange = (1500, 2000))
            save(joinpath(figdir, "temperature_it_$it.png"), fig)
        end
    end

    return (ni = ni, xci = xci, xvi = xvi, li = li, di = di), thermal
end

figdir = "MPI_Diffusion2D"
n = 32
igg = IGG(init_global_grid((n, n)..., 1; init_MPI = true, select_device = false)...) #init MPI
diffusion_2D(igg, figdir; nx = n, ny = n)
