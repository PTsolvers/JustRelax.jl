struct PS_Setup{B,C}
    device::Symbol

    function PS_Setup(device::Symbol, precission::DataType, nDim::Integer)
        return new{precission,nDim}(device)
    end
end

function environment!(model::PS_Setup{T,N}) where {T,N}
    gpu = model.device == :gpu ? true : false

    # environment variable for XPU
    @eval begin
        const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : $gpu
    end

    # call appropriate FD module
    Base.eval(@__MODULE__, Meta.parse("using ParallelStencil.FiniteDifferences$(N)D"))
    Base.eval(Main, Meta.parse("using ParallelStencil.FiniteDifferences$(N)D"))

    # start ParallelStencil
    global PTArray
    if model.device == :gpu
        eval(:(@init_parallel_stencil(CUDA, $T, $N)))
        Base.eval(Main, Meta.parse("using CUDA"))
        eval(:(PTArray = CUDA.CuArray{$T,$N}))
    else
        @eval begin
            @init_parallel_stencil(Threads, $T, $N)
            PTArray = Array{$T,$N}
        end
    end

    # CREATE ARRAY STRUCTS
    make_velocity_struct!(N) # velocity
    make_symmetrictensor_struct!(N) # (symmetric) tensors
    ## Stokes
    make_residual_struct!(N) # residuals
    make_stokes_struct!() # Arrays for Stokes solver
    make_PTstokes_struct!() # numeric parameter for stokes
    ## Two Phase Flow
    make_pressure_struct!() # pressure
    make_P_residual_struct!() # residuals for pressure
    make_TPF_struct!() # Arrays for two phase flow solver
    make_PTTPF_struct!() # numeric parameter for two phase flow
    make_TPF_parameter_struct!() # background and initial parameters
    ## thermal diffusion
    make_thermal_arrays!(N) # Arrays for Thermal Diffusion solver
    make_PTthermal_struct!() # PT Thermal Diffusion coefficients

    # includes and exports
    @eval begin
        export USE_GPU, PTArray
        export Velocity, SymmetricTensor, Residual, StokesArrays, PTStokesCoeffs
        export TPF_Pressure, P_Residual, TPFArrays, PTTPFCoeffs, PTTPFParams
        export ThermalArrays, PTThermalCoeffs
        export AbstractStokesModel, Viscous, ViscoElastic, PoreEvolution
        export solve!

        include(joinpath(@__DIR__, "boundaryconditions/BoundaryConditions.jl"))
        export pureshear_bc!, free_slip_x!, free_slip_y!, free_slip_z!, apply_free_slip!, zero_y!

        include(joinpath(@__DIR__, "stokes/Stokes.jl"))
        export stress

        include(joinpath(@__DIR__, "TwoPhaseFlow/Direct.jl"))

        include(joinpath(@__DIR__, "TwoPhaseFlow/PT.jl"))

        include(joinpath(@__DIR__, "Utils.jl"))

        include(joinpath(@__DIR__, "stokes/Elasticity.jl"))

        include(joinpath(@__DIR__, "thermal_diffusion/Diffusion.jl"))
        export ThermalParameters
    end

    # conditional submodule load
    module_names = if N === 1
        (Symbol("ThermalDiffusion$(N)D"),)
    elseif N === 2
        (Symbol("Stokes$(N)D"), Symbol("Elasticity$(N)D"), Symbol("ThermalDiffusion$(N)D"),Symbol("TwoPhaseFlow$(N)D"))
    else
        (Symbol("Elasticity$(N)D"), Symbol("ThermalDiffusion$(N)D"))
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
