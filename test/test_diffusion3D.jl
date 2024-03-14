push!(LOAD_PATH, "..")

using Test, Suppressor
using GeoParams, CellArrays
using JustRelax, JustRelax.DataIO
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 3)

# setup ParallelStencil.jl environment
model  = PS_Setup(:Threads, Float64, 3)
environment!(model)

# HELPER FUNCTIONS ---------------------------------------------------------------
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
    ttot     = 1 * Myr # total simulation time
    dt       = 50 * kyr # physical time step

    # Physical domain
    ni           = (nx, ny, nz)
    li           = (lx, ly, lz)  # domain length in x- and y-
    di           = @. li / ni # grid step in x- and -y
    origin       = 0, 0, -lz # nodes at the center and vertices of the cells
    igg          = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI)...) # init MPI
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

    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[3])

    # Add thermal perturbation
    δT                  = 100e0 # thermal perturbation
    r                   = 10e3 # thermal perturbation radius
    center_perturbation = lx/2, ly/2, -lz/2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xvi)

    t  = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    # Physical time loop
    while it < 10
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di,;
            igg,
            verbose=false,
        )

        t  += dt
        it += 1
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return thermal
end

@testset "Diffusion_3D" begin
    @suppress begin
        nx=32;
        ny=32;
        nz=32;
        thermal = diffusion_3D(; nx = nx, ny = ny, nz = nz)
        @test thermal.T[Int(ceil(nx/2)), Int(ceil(ny/2)), Int(ceil(nz/2))] ≈ 1824.614400703972 rtol=1e-3
        @test thermal.Tc[Int(ceil(nx/2)), Int(ceil(ny/2)), Int(ceil(nz/2))] ≈ 1827.002299288895 rtol=1e-3
    end
end
