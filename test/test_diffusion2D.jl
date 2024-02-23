push!(LOAD_PATH, "..")

using Test, Suppressor
using GeoParams, CellArrays
using JustRelax, JustRelax.DataIO
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)

# setup ParallelStencil.jl environment
model  = PS_Setup(:Threads, Float64, 2)
environment!(model)

# HELPER FUNCTIONS ---------------------------------------------------------------
@parallel_indices (i, j) function init_T!(T, z)
    if z[j] == maximum(z)
        T[i, j] = 300.0
    elseif z[j] == minimum(z)
        T[i, j] = 3500.0
    else
        T[i, j] = z[j] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end

function elliptical_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _elliptical_perturbation!(T, δT, xc, yc, r, x, y)
        @inbounds if (((x[i]-xc ))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i, j]  += δT
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, r, xvi...)
end
# MAIN SCRIPT --------------------------------------------------------------------
function diffusion_2D(; nx=32, ny=32, lx=100e3, ly=100e3, ρ0=3.3e3, Cp0=1.2e3, K0=3.0)
    kyr      = 1e3 * 3600 * 24 * 365.25
    Myr      = 1e3 * kyr
    ttot     = 1 * Myr # total simulation time
    dt       = 50 * kyr # physical time step

    init_mpi = JustRelax.MPI.Initialized() ? false : true
    igg    = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

    # Physical domain
    ni           = (nx, ny)
    li           = (lx, ly)  # domain length in x- and y-
    di           = @. li / ni # grid step in x- and -y
    origin       = 0, -ly
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    # Define the thermal parameters with GeoParams
    rheology = SetMaterialParams(;
        Phase        = 1,
        Density      = PT_Density(; ρ0=3.1e3, β=0.0, T0=0.0, α = 1.5e-5),
        HeatCapacity = ConstantHeatCapacity(; Cp=Cp0),
        Conductivity = ConstantConductivity(; k=K0),
    )
    # fields needed to compute density on the fly
    P          = @zeros(ni...)
    args       = (; P=P)

    ## Allocate arrays needed for every Thermal Diffusion
    thermal    = ThermalArrays(ni)
    thermal.H .= 1e-6 # radiogenic heat production
    # physical parameters
    ρ          = @fill(ρ0, ni...)
    Cp         = @fill(Cp0, ni...)
    K          = @fill(K0, ni...)
    ρCp        = @. Cp * ρ

    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2])

    # Add thermal perturbation
    δT                  = 100e0 # thermal perturbation
    r                   = 10e3 # thermal perturbation radius
    center_perturbation = lx/2, -ly/2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xvi)
    temperature2center!(thermal)

    # Time loop
    t  = 0.0
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
            di;
            verbose=false,
        )

        t  += dt
        it += 1
    end

    return thermal
end

@testset "Diffusion_2D" begin
    @suppress begin
        nx=32;
        ny=32;
        thermal = diffusion_2D(nx = nx, ny = ny)
        @test thermal.T[Int(ceil(size(thermal.T)[1]/2)), Int(ceil(size(thermal.T)[2]/2))] ≈ 1823.6076461523571 rtol=1e-1
        @test thermal.Tc[Int(ceil(size(thermal.Tc)[1]/2)), Int(ceil(size(thermal.Tc)[2]/2))] ≈ 1828.3169386441218 rtol=1e-1
    end
end
