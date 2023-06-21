struct PS_Setup{B,C}
    device::Symbol

    function PS_Setup(device::Symbol, precission::DataType, nDim::Integer)
        return new{precission,nDim}(device)
    end
end

function environment!(model::PS_Setup{T,N}) where {T,N}
    gpu = model.device == :gpu ? true : false

    # call appropriate FD module
    Base.eval(@__MODULE__, Meta.parse("using ParallelStencil.FiniteDifferences$(N)D"))
    Base.eval(Main, Meta.parse("using ParallelStencil.FiniteDifferences$(N)D"))

    # start ParallelStencil
    if model.device == :gpu
        eval(:(@init_parallel_stencil(CUDA, $T, $N)))
        Base.eval(Main, Meta.parse("using CUDA"))
        if !isconst(Main, :PTArray)
            eval(:(const PTArray = CUDA.CuArray{$T,$N}))
        end
    else
        @eval begin
            @init_parallel_stencil(Threads, $T, $N)
            if !isconst(Main, :PTArray)
                const PTArray = Array{$T,$N}
            end
        end
    end

    # CREATE ARRAY STRUCTS
    make_velocity_struct!(N) # velocity
    make_symmetrictensor_struct!(N) # (symmetric) tensors
    ## Stokes
    make_residual_struct!(N) # residuals
    make_stokes_struct!() # Arrays for Stokes solver
    make_PTstokes_struct!()
    ## thermal diffusion
    make_thermal_arrays!(N) # Arrays for Thermal Diffusion solver
    make_PTthermal_struct!() # PT Thermal Diffusion coefficients

    # includes and exports
    @eval begin
        export USE_GPU, PTArray
        export Velocity, SymmetricTensor, Residual, StokesArrays, PTStokesCoeffs
        export ThermalArrays, PTThermalCoeffs
        export AbstractStokesModel,
            AbstractElasticModel, Viscous, ViscoElastic, ViscoElastoPlastic
        export solve!

        include(joinpath(@__DIR__, "Utils.jl"))
        export @allocate, @add, @idx, @copy
        export @velocity, @strain, @stress, @tensor, @shear, @normal, @stress_center, @strain_center, @tensor_center
        export compute_dt, assign!, tupleize, compute_maxloc!

        include(joinpath(@__DIR__, "boundaryconditions/BoundaryConditions.jl"))
        export pureshear_bc!, FlowBoundaryConditions, flow_bcs!
        export TemperatureBoundaryConditions, thermal_boundary_conditions!, thermal_bcs!
        export free_slip_x!, free_slip_y!, free_slip_z!, apply_free_slip!

        include(joinpath(@__DIR__, "rheology/BuoyancyForces.jl"))
        export compute_ρg!
        
        include(joinpath(@__DIR__, "rheology/Viscosity.jl"))
        export compute_viscosity!

        include(joinpath(@__DIR__, "stokes/Stokes2D.jl"))
        export solve!

        include(joinpath(@__DIR__, "stokes/Stokes3D.jl"))
        export solve!

        include(joinpath(@__DIR__, "thermal_diffusion/DiffusionExplicit.jl"))
        # include(joinpath(@__DIR__, "thermal_diffusion/Diffusion.jl"))
        export ThermalParameters

        include(joinpath(@__DIR__, "Interpolations.jl"))
        export vertex2center!, center2vertex!, temperature2center!


        
    end

    # conditional submodule load
    module_names = if N === 1
        (Symbol("ThermalDiffusion$(N)D"),)
    elseif N === 2
        (Symbol("Stokes$(N)D"), Symbol("ThermalDiffusion$(N)D"))
    else
        (Symbol("Stokes$(N)D"), Symbol("ThermalDiffusion$(N)D"))
    end

    for m in module_names
        Base.@eval begin
            @reexport import .$m
        end
    end
end

function ps_reset!()
    Base.eval(Main, ParallelStencil.@reset_parallel_stencil)
    Base.eval(@__MODULE__, ParallelStencil.@reset_parallel_stencil)
    return nothing
end
