push!(LOAD_PATH, "..")

using Test, Suppressor
using GeoParams
using JustRelax, JustRelax.JustRelax2D

const backend_JR = CPUBackend

using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)  #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

import JustRelax.@cell

distance(p1, p2) = mapreduce(x->(x[1]-x[2])^2, +, zip(p1, p2)) |> sqrt

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

function init_phases!(phases, particles, xc, yc, r)
    ni = size(phases)
    center = xc, yc

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, center, r)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = JustRelax.@cell py[ip, i, j]

            # plume - rectangular
            if (((x - center[1] ))^2 + ((y - center[2]))^2) ≤ r^2
                JustRelax.@cell phases[ip, i, j] = 2.0

            else
                JustRelax.@cell phases[ip, i, j] = 1.0
            end
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(phases, particles.coords..., particles.index, center, r)
end

@parallel_indices (I...) function compute_temperature_source_terms!(H, rheology, phase_ratios, args)

    args_ij = ntuple_idx(args, I...)
    H[I...] = fn_ratio(compute_radioactive_heat, rheology, phase_ratios[I...], args_ij)

    return nothing
end

function diffusion_2D(; nx=32, ny=32, lx=100e3, ly=100e3, Cp0=1.2e3, K0=3.0)

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
    grid         = Geometry(ni, li; origin = (0, -ly))
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    # Define the thermal parameters with GeoParams
    rheology = (
        SetMaterialParams(;
            Phase           = 1,
            Density         = PT_Density(; ρ0=3e3, β=0.0, T0=0.0, α = 1.5e-5),
            HeatCapacity    = ConstantHeatCapacity(; Cp=Cp0),
            Conductivity    = ConstantConductivity(; k=K0),
            RadioactiveHeat = ConstantRadioactiveHeat(1e-6),
        ),
        SetMaterialParams(;
            Phase           = 2,
            Density         = PT_Density(; ρ0=3.3e3, β=0.0, T0=0.0, α = 1.5e-5),
            HeatCapacity    = ConstantHeatCapacity(; Cp=Cp0),
            Conductivity    = ConstantConductivity(; k=K0),
            RadioactiveHeat = ConstantRadioactiveHeat(1e-7),
        ),
    )

    # fields needed to compute density on the fly
    P          = @zeros(ni...)
    args       = (; P=P)

    ## Allocate arrays needed for every Thermal Diffusion
    thermal    = ThermalArrays(backend_JR, ni)
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

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 40, 40, 1
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...
    )
    # temperature
    pPhases,     = init_cell_arrays(particles, Val(1))
    init_phases!(pPhases, particles, center_perturbation..., r)
    phase_ratios = PhaseRatio(backend_JR, ni, length(rheology))
    phase_ratios_center(phase_ratios, particles, grid, pPhases)
    # ----------------------------------------------------

    @parallel (@idx ni) compute_temperature_source_terms!(thermal.H, rheology, phase_ratios.center, args)

    # PT coefficients for thermal diffusion
    args       = (; P=P, T=thermal.Tc)
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.65 / √2
    )

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
            kwargs = (
                phase   = phase_ratios,
                iterMax = 1e3,
                nout    = 10,
                verbose=false,
            )
        )

        it += 1
        t  += dt
    end

    return thermal
end

@testset "Diffusion_2D_Multiphase" begin
    @suppress begin
        nx=32;
        ny=32;
        thermal = diffusion_2D(; nx = nx, ny = ny)

        nx_T, ny_T = size(thermal.T)
        @test thermal.T[nx_T >>> 1 + 1, ny_T >>> 1 + 1] ≈ 1819.2297931741878 atol=1e-1
        @test thermal.Tc[ nx >>> 1    ,   nx >>> 1    ] ≈ 1824.3532934301472 atol=1e-1
    end
end
