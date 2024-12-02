using Adapt, MuladdMacro

include("types/constructors/stokes.jl")
export StokesArrays, PTStokesCoeffs

include("types/constructors/heat_diffusion.jl")
export ThermalArrays, PTThermalCoeffs

include("types/constructors/weno.jl")
export WENO5

include("Utils.jl")
export @allocate,
    @add,
    @idx,
    @copy,
    @velocity,
    @displacement,
    @strain,
    @plastic_strain,
    @stress,
    @tensor,
    @shear,
    @normal,
    @stress_center,
    @strain_center,
    @tensor_center,
    @qT,
    @qT2,
    @residuals,
    compute_dt,
    multi_copy!,
    take

include("types/displacement.jl")
export velocity2displacement!, displacement2velocity!

include("mask/constructors.jl")

include("boundaryconditions/BoundaryConditions.jl")
export AbstractBoundaryConditions,
    TemperatureBoundaryConditions,
    AbstractFlowBoundaryConditions,
    DisplacementBoundaryConditions,
    VelocityBoundaryConditions,
    flow_bcs!,
    thermal_bcs!,
    pureshear_bc!

include("MiniKernels.jl")

include("phases/phases.jl")
export fn_ratio

include("rheology/BuoyancyForces.jl")
export compute_ρg!

include("rheology/Viscosity.jl")
export compute_viscosity!

include("rheology/Melting.jl")
export compute_melt_fraction!

include("particles/subgrid_diffusion.jl")
export subgrid_characteristic_time!

include("Interpolations.jl")
export vertex2center!,
    center2vertex!, temperature2center!, velocity2vertex!, velocity2center!

include("advection/weno5.jl")
export WENO_advection!

# Stokes

include("rheology/GeoParams.jl")

include("rheology/StressUpdate.jl")

include("stokes/StressKernels.jl")
export tensor_invariant!

include("stokes/PressureKernels.jl")
export rotate_stress_particles!

include("stokes/VelocityKernels.jl")

# variational Stokes
include("variational_stokes/mask.jl")
export RockRatio, update_rock_ratio!

include("variational_stokes/PressureKernels.jl")

include("variational_stokes/MiniKernels.jl")

include("variational_stokes/StressKernels.jl")

include("variational_stokes/VelocityKernels.jl")

include("stress_rotation/constructors.jl")
export StressParticles

include("stress_rotation/stress_rotation_particles.jl")
export rotate_stress!, stress2grid!

# thermal diffusion

include("thermal_diffusion/DiffusionPT.jl")
export PTThermalCoeffs, heatdiffusion_PT!, compute_shear_heating!
