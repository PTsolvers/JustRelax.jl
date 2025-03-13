# using CairoMakie
using JustRelax, JustRelax.JustRelax2D
const backend_JR = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)  #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)


using GeoParams

using CairoMakie

@parallel_indices (i, j) function init_T!(T, z, lz)
    if z[j] ≥ 0.0
        T[i, j] = 300.0
    elseif z[j] == -lz
        T[i, j] = 3500.0
    else
        T[i, j] = z[j] * (1900.0 - 1600.0) / (-lz) + 1600.0
    end
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _elliptical_perturbation!(T, δT, xc, yc, r, x, y)
        if (((x[i] - xc))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i + 1, j] += δT
        end
        return nothing
    end
    nx, ny = size(T)
    return @parallel (1:(nx - 2), 1:ny) _elliptical_perturbation!(T, δT, xc, yc, r, xvi...)
end

function diffusion_2D(
        figdir;
        nx = 32,
        ny = 32,
        lx = 100.0e3,
        ly = 100.0e3,
        ρ0 = 3.3e3,
        Cp0 = 1.2e3,
        K0 = 3.0
    )
    kyr = 1.0e3 * 3600 * 24 * 365.25
    Myr = 1.0e3 * kyr
    ttot = 1 * Myr # total simulation time
    dt = 50 * kyr # physical time step

    # Physical domain
    ni = nx, ny
    li = lx, ly  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    origin = 0.0, -ly
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = true, select_device = false)...) #init MPI
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
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
    args = (; P = P, T = @zeros(ni .+ 1...))

    ## Allocate arrays needed for every Thermal Diffusion
    thermal = ThermalArrays(backend_JR, ni)
    thermal.H .= 1.0e-6 # radiogenic heat production
    # physical parameters
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ

    pt_thermal = PTThermalCoeffs(backend_JR, K, ρCp, dt, di, li)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2], ly)

    # Add thermal perturbation
    δT = 100.0e0 # thermal perturbation
    r = 10.0e3 # thermal perturbation radius
    center_perturbation = lx / 2, -ly / 2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xvi)
    temperature2center!(thermal)

    # global array
    nx_v = (nx - 2) * igg.dims[1]
    ny_v = (ny - 2) * igg.dims[2]
    T_v = zeros(nx_v, ny_v)
    T_nohalo = zeros(nx - 2, ny - 2)

    # Time loop
    t = 0.0
    it = 0
    ## IO -----------------------------------------------
    take(figdir)
    # ---------------------------------------------------

    while it < 10
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
                b_width = (4, 4, 1),
            )
        )

        temperature2center!(thermal)

        @views T_nohalo .= Array(thermal.Tc[2:(end - 2), 2:(end - 1)]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)

        if igg.me == 0
            fig, = heatmap(T_v, colorrange = (1500, 2000))
            save(joinpath(figdir, "temperature_it_$it.png"), fig)
        end

        t += dt
        it += 1
    end

    return nothing
end

figdir = "MPI_Diffusion2D"
n = 32
diffusion_2D(figdir; nx = n, ny = n)
