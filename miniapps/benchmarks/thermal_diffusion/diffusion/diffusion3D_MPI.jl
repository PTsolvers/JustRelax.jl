using JustRelax, GeoParams
using GLMakie

# setup ParallelStencil.jl environment
dimension = 3 # 2 | 3
device = :cpu # :cpu | :CUDA | :AMDGPU
precision = Float64
model = PS_Setup(device, precision, dimension)
environment!(model)

@parallel_indices (i, j, k) function init_T!(T, z, lz)
    if z[k] ≥ 0.0
        T[i, j, k] = 300.0
    elseif z[k] == -lz
        T[i, j, k] = 3500.0
    else
        T[i, j, k] = z[k] * (1900.0 - 1600.0) / (-lz) + 1600.0
    end
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _elliptical_perturbation!(T, x, y, z)
        @inbounds if (((x[i]-xc))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
            T[i, j, k] += δT
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, xvi...)
end

function diffusion_3D(;
    nx           = 32,
    ny           = 32,
    nz           = 32,
    lx           = 100e3,
    ly           = 100e3,
    lz           = 100e3,
    ρ0           = 3.3e3,
    Cp0          = 1.2e3,
    K0           = 3.0,
    init_MPI     = JustRelax.MPI.Initialized() ? false : true,
    finalize_MPI = false,
)

    kyr      = 1e3 * 3600 * 24 * 365.25
    Myr      = 1e6 * 3600 * 24 * 365.25
    ttot     = 10 * Myr # total simulation time
    dt       = 50 * kyr # physical time step

    # Physical domain
    ni           = (nx, ny, nz)
    li           = (lx, ly, lz)  # domain length in x- and y-
    di           = @. li / ni # grid step in x- and -y
    origin       = 0, 0, -lz # nodes at the center and vertices of the cells
    igg          = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI, select_device=false)...) # init MPI
    di           = @. li / (nx_g(), ny_g(), nz_g()) # grid step in x- and -y
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells


    # Define the thermal parameters with GeoParams
    rheology = SetMaterialParams(;
        Phase        = 1,
        Density      = PT_Density(; ρ0=3.1e3, β=0.0, T0=0.0, α = 1.5e-5),
        HeatCapacity = ConstantHeatCapacity(; cp=Cp0),
        Conductivity = ConstantConductivity(; k=K0),
    )

    # fields needed to compute density on the fly
    P          = @zeros(ni...)
    args       = (; P=P)

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal    = ThermalArrays(ni)
    thermal.H .= 1e-6
    # physical parameters
    ρ          = @fill(ρ0, ni...)
    Cp         = @fill(Cp0, ni...)
    K          = @fill(K0, ni...)
    ρCp        = @. Cp * ρ

    # Boundary conditions
    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; CFL = 0.75 / √3.1)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux     = (left = true , right = true , top = false, bot = false, front = true , back = true),
        periodicity = (left = false, right = false, top = false, bot = false, front = false, back = false),
    )

    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[3], lz)

    # return nothing 

    # Add thermal perturbation
    δT                  = 100e0 # thermal perturbation
    r                   = 10e3 # thermal perturbation radius
    center_perturbation = lx/2, ly/2, -lz/2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xvi)

    # Visualization global arrays
    nx_v     = ((nx + 1)-2) * igg.dims[1]
    ny_v     = ((ny + 1)-2) * igg.dims[2]
    nz_v     = ((nz + 1)-2) * igg.dims[3]
    T_v      = zeros(nx_v, ny_v, nz_v)             # plotting is done on the CPU
    T_nohalo = zeros((nx+1)-2, (ny+1)-2, (nz+1)-2) # plotting is done on the CPU

    t  = 0.0
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
            di;
            igg
        )

        @views T_nohalo .= Array(thermal.T[2:end-1, 2:end-1, 2:end-1]) # Copy data to CPU removing the halo
        gather!(T_nohalo, T_v)

        if igg.me == 0
            slice_j = ny_v >>> 1
            fig, = heatmap(T_v[:, slice_j, :])
            save("temperature_3D_it_$it.png", fig)
            println("\n SAVED TEMPERATURE \n")
        end

        t  += dt
        it += 1
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return nothing
end

n  = 32
nx = n
ny = n
nz = n
diffusion_3D(; nx=n, ny=n, nz=n)
