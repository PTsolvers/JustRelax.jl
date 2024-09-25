push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test, Suppressor
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend_JR = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
    CPUBackend
end


using GeoParams

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
    igg      = IGG(init_global_grid(nx, ny, 1; init_MPI = init_mpi)...)

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
    args       = (; P=P, T=@zeros(ni.+1...))

    ## Allocate arrays needed for every Thermal Diffusion
    thermal    = ThermalArrays(backend_JR, ni)
    thermal.H .= 1e-6 # radiogenic heat production
    # physical parameters
    ρ          = @fill(ρ0, ni...)
    Cp         = @fill(Cp0, ni...)
    K          = @fill(K0, ni...)
    ρCp        = @. Cp * ρ

    pt_thermal = PTThermalCoeffs(backend_JR, K, ρCp, dt, di, li; CFL = 0.95 / √2.1)
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
            kwargs = (;
                verbose = false
            ),
        )

        t  += dt
        it += 1
    end

    return thermal
end

@testset "Diffusion_2D" begin
    @suppress begin
        nx, ny  = 32, 32
        thermal = diffusion_2D(; nx = nx, ny = ny)

        nx_T, ny_T = size(thermal.T)
        @test  Array(thermal.T)[nx_T >>> 1 + 1, ny_T >>> 1 + 1] ≈ 1823.6076461523571 atol=1e-1
        @test Array(thermal.Tc)[  nx >>> 1    ,   nx >>> 1    ] ≈ 1828.3169386441218 atol=1e-1
    end
end
