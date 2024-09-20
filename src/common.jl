include("types/constructors/stokes.jl")
export StokesArrays, PTStokesCoeffs

include("types/constructors/heat_diffusion.jl")
export ThermalArrays, PTThermalCoeffs


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

# include("thermal_diffusion/DiffusionExplicit.jl")
# export ThermalParameters

include("particles/subgrid_diffusion.jl")
export subgrid_characteristic_time!

include("Interpolations.jl")
export vertex2center!, center2vertex!, temperature2center!, velocity2vertex!

include("advection/weno5.jl")
export WENO5, WENO_advection!

# Stokes

include("rheology/GeoParams.jl")
include("rheology/StressUpdate.jl")
include("stokes/StressRotation.jl")
include("stokes/StressKernels.jl")
export tensor_invariant!

include("stokes/PressureKernels.jl")
include("stokes/VelocityKernels.jl")

# thermal diffusion

include("thermal_diffusion/DiffusionPT.jl")
export PTThermalCoeffs, heatdiffusion_PT!, compute_shear_heating!
