using JustRelax, Printf, LinearAlgebra, GeoParams, GLMakie

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

@parallel_indices (i, j) function init_T!(T, z)
    T[i, j] = if z[j] == maximum(z)
        300.0
    elseif z[j] == minimum(z)
        3500.0
    else
        z[j] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end


function diffusion_2D(; nx=32, ny=32, lx=100e3, ly=100e3, ρ0=3.3e3, Cp0=1.2e3, K0=3.0)
    kyr = 1e3 * 3600 * 24 * 365.25
    Myr = 1e3 * kyr
    ttot = 1 * Myr # total simulation time
    dt = 50 * kyr # physical time step

    # Physical domain
    ni = (nx, ny)
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=(0, -ly)) # nodes at the center and vertices of the cells

    # Define the thermal parameters with GeoParams
    rheology = SetMaterialParams(;
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.1e3, β=0.0, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=Cp0),
        Conductivity      = ConstantConductivity(; k=K0),
    )
    # fields needed to compute density on the fly
    P = @zeros(ni...)
    args = (; P=P)

    ## Allocate arrays needed for every Thermal Diffusion
    thermal = ThermalArrays(ni)
    # physical parameters
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ

    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux = (left = true, right = true, top = false, bot = false), 
    )
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2])
    @copy thermal.Told thermal.T
    T1 = deepcopy(thermal.T)

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    # Time loop
    while it < nt
        # heatdiffusion_PT!(
        #     thermal,
        #     pt_thermal,
        #     thermal_bc,
        #     rheology,
        #     args,
        #     dt,
        #     di,
        #     nothing
        # )

        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            K,
            ρCp,
            dt,
            di,
            nothing
        )

        it += 1
        t += dt
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), thermal, T1
end

# Model size 
lx = ly = L = 100e3 # [m]
# model resolution (number of gridpoints)
nx, ny = 64, 64
ρ0  = 3.3e3
Cp0 = 1.2e3
K0  = 3.0

# start model
geometry, thermal, T = diffusion_2D(;
    nx=nx, ny=ny, lx=L, ly=L, ρ0=3.3e3, Cp0=1.2e3, K0=3.0
);

Yv = [y for x in geometry.xvi[1], y in geometry.xvi[2]][:];

fig, ax, = lines(
    T[2:end-1, :][:],
    Yv,
    color=:black,
)

lines!(
    ax,
    thermal.T[2:end-1, :][:],
    Yv,
    color=:red,
)
fig