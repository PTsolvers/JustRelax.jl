include("types/constructors/stokes.jl")
export StokesArrays, PTStokesCoeffs

include("types/constructors/heat_diffusion.jl")
export ThermalArrays, PTThermalCoeffs

include("types/constructors/phases.jl")
export PhaseRatio

include("Utils.jl")
export @allocate,
    @add,
    @idx,
    @copy,
    @velocity,
    @displacement,
    @strain,
    @strain_plastic,
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
    take,
    detect_args_size,
    _tuple,
    indices

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
    pureshear_bc!,
    apply_free_slip!

include("MiniKernels.jl")

include("phases/phases.jl")
export fn_ratio, phase_ratios_center!, numphases, nphases

include("rheology/BuoyancyForces.jl")
export compute_ρg!

include("rheology/Viscosity.jl")
export compute_viscosity!

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
