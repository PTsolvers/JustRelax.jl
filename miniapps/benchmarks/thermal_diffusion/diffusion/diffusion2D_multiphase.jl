using Printf, LinearAlgebra, GeoParams, SpecialFunctions, CellArrays, StaticArrays
using JustRelax
using ParallelStencil
@init_parallel_stencil(Threads, Float64, 2)  #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2)  #or (:CUDA, Float64, 2) or (:AMDGPU, Float64, 2)
environment!(model)

import JustRelax.@cell

# @inline init_particle_fields(particles) = @zeros(size(particles.coords[1])...)
# @inline init_particle_fields(particles, nfields) = tuple([zeros(particles.coords[1]) for i in 1:nfields]...)
# @inline init_particle_fields(particles, ::Val{N}) where N = ntuple(_ -> @zeros(size(particles.coords[1])...) , Val(N))
# @inline init_particle_fields_cellarrays(particles, ::Val{N}) where N = ntuple(_ -> @fill(0.0, size(particles.coords[1])..., celldims=(cellsize(particles.index))), Val(N))

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
        @inbounds if distance((xc, yc), (x[i], y[i])) ≤ r^2
            T[i, j]  += δT
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, r, xvi...)
end

function init_phases!(phases, particles, xc, yc, r)
    ni = size(phases)
    center = xc, yc

    @parallel_indices (i, j) function init_phases!(phases, px, py, index)
        @inbounds for ip in JustRelax.JustRelax.cellaxes(phases)
            # quick escape
            JustRelax.@cell(index[ip, i, j]) == 0 && continue

            x = JustRelax.@cell px[ip, i, j]
            y = JustRelax.@cell py[ip, i, j]

            # plume - rectangular
            if distance(center, (x, y)) ≤ r^2
                JustRelax.@cell phases[ip, i, j] = 2.0

            else
                JustRelax.@cell phases[ip, i, j] = 1.0
            end
        end
        return nothing
    end

    @parallel (JustRelax.@idx ni) init_phases!(phases, particles.coords..., particles.index)
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
            HeatCapacity    = ConstantHeatCapacity(; cp=Cp0),
            Conductivity    = ConstantConductivity(; k=K0),
            RadioactiveHeat = ConstantRadioactiveHeat(1e-6),
        ),
        SetMaterialParams(;
            Phase           = 1,
            Density         = PT_Density(; ρ0=3.3e3, β=0.0, T0=0.0, α = 1.5e-5),
            HeatCapacity    = ConstantHeatCapacity(; cp=Cp0),
            Conductivity    = ConstantConductivity(; k=K0),
            RadioactiveHeat = ConstantRadioactiveHeat(1e-7),
        ),
    )

    # fields needed to compute density on the fly
    P          = @zeros(ni...)
    args       = (; P=P)

    ## Allocate arrays needed for every Thermal Diffusion
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2])

    # Add thermal perturbation
    δT                  = 100e0 # thermal perturbation
    r                   = 10e3 # thermal perturbation radius
    center_perturbation = lx/2, -ly/2
    elliptical_perturbation!(thermal.T, δT, center_perturbation..., r, xvi)
    @parallel (JustRelax.@idx size(thermal.Tc)...) temperature2center!(thermal.Tc, thermal.T)

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 40, 40, 1
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi..., di..., ni...
    )
    # temperature
    pPhases,     = init_cell_arrays(particles, Val(1))
    init_phases!(pPhases, particles, center_perturbation..., r)
    phase_ratios = PhaseRatio(ni, length(rheology))
    @parallel (@idx ni) JustRelax.phase_ratios_center(phase_ratios.center, particles.coords..., xci..., di, pPhases)
    # ----------------------------------------------------

    @parallel (@idx ni) compute_temperature_source_terms!(thermal.H, rheology, phase_ratios.center, args)

    # PT coefficients for thermal diffusion
    args       = (; P=P, T=thermal.Tc)
    pt_thermal = JustRelax.PTThermalCoeffs(
        rheology, phase_ratios, args, dt, ni, di, li; ϵ=1e-5, CFL=0.65 / √2
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
            phase   = phase_ratios,
            iterMax = 1e3,
            nout    = 10,
        )

        it += 1
        t  += dt
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), thermal
end

diffusion_2D()
