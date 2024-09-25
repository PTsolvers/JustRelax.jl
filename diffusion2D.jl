using JustRelax, ParallelStencil
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

# using JustPIC
# using JustPIC._2D
# # Threads is the default backend,
# # to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# # and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
# const backend = CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# setup ParallelStencil.jl environment
model = PS_Setup(:Threads, Float64, 2) #or (:CUDA, Float64, 2) or (:AMDGPU, Float64, 2)
environment!(model)

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie

include("diffeq.jl")

@parallel_indices (i, j) function init_T!(T, z)
    depth = abs(z[j])

    T[i, j] = if depth < 0e0
        273e0
    elseif 0e0 ≤ (depth) < 120e3
        dTdZ        = (1273-273)/120e3
        offset      = 273e0
        (depth) * dTdZ + offset
    elseif (depth) ≥ 120e3
        offset      = 273e0
        1000.0 + offset
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

function diffusion_2D(dt, nt; nx=32, ny=32, lx=1000e3, ly=1000e3, ρ0=4e3, Cp0=1.25e3, K0=5.0)

    # Physical domain
    ni           = (nx, ny)
    li           = (lx, ly)  # domain length in x- and y-
    di           = @. li / ni # grid step in x- and -y
    origin       = 0, -ly
    grid         = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    # Define the thermal parameters with GeoParams
    α = 2.5e-5
    rheology = SetMaterialParams(;
        Phase        = 1,
        Density      = PT_Density(; ρ0=4000.0, T0=273, α = α, β = 0.0),
        HeatCapacity = ConstantHeatCapacity(; Cp=Cp0),
        Conductivity = ConstantConductivity(; k=K0),
    )
    # fields needed to compute density on the fly
    P          = @zeros(ni...)
    args       = (; P=P)

    ## Allocate arrays needed for every Thermal Diffusion
    thermal    = ThermalArrays(ni)
    ρg         = @fill(ρ0, ni...)
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2])
    @parallel (@idx size(thermal.Tc)) temperature2center!(thermal.Tc, thermal.T)
    @parallel (JustRelax.@idx ni) compute_ρg!(ρg, rheology, (T=thermal.Tc, P=P))
    ρ  =  ρg ./ 9.81
    
    # thermal.H .= 1e-6 # radiogenic heat production
    # physical parameters
    # ρ          = @fill(ρ0, ni...)
    Cp         = @fill(Cp0, ni...)
    K          = @fill(K0, ni...)
    ρCp        = @. Cp * ρ
    
    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; CFL= 1e-1 / √2.1)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )

    # Time loop
    t  = 0.0
    it = 0
    # nt = Int(ceil(ttot / dt))
    # nt = 25
    while it < nt
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            nout = 10
        )

        t  += dt
        it += 1
    end
    # lines!(ax1, thermal.T[2,:],  xvi[2]./1e3)
    # display(fig)

    return thermal.T, xvi[2]
end

nx  = ny = 100
lx  = ly = 1000e3
ρ0  = 4e3
Cp0 = 1.25e3
K0  = 5.0

igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end

kyr   = 1e3 * 3600 * 24 * 365.25
dt    = 100 * kyr # physical time step
nt    = 5000;      # number of timesteps to take

T_JR,    z = diffusion_2D(dt, nt; nx=nx, ny=ny); # JustRelax
# T_DE_const = diffeq_constant_density(ny, dt, nt);  # DiffEq.jl
T_DE_PT    = diffeq_Tdep_density(ny, dt, nt);      # DiffEq.jl

fig = Figure(size = (1200, 900))
ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
ax2 = Axis(fig[1,2], aspect = 2/3, title = "(T_JR - T_DE) / T_DE")
lines!(ax1, T_DE_const[1],  z./1e3, label = "initial T")

lines!(ax1, T_JR[2,:],  z./1e3, label = "JustRelax")
# lines!(ax1, T_DE_const[end],  z./1e3, label = "DiffEq (const. ρ)")
lines!(ax1, T_DE_PT[end],  z./1e3, label = "DiffEq (T dep ρ)")

lines!(ax2, (T_JR[2,:].-T_DE_PT[end]) ./ T_DE_PT[end] * 100,  z./1e3,)
ylims!(ax1, minimum(z)./1e3, 0)
ylims!(ax2, minimum(z)./1e3, 0)
axislegend(ax1, position = :lb)
fig