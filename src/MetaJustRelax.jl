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
    eval(Meta.parse("using ParallelStencil.FiniteDifferences$(N)D"))

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

    # create array structs
    make_velocity_struct!(N) # velocity
    make_symmetrictensor_struct!(N) # (symmetric) tensors
    make_residual_struct!(N) # residuals
    make_stokes_struct!() # Arrays for Stokes solver
    make_PTstokes_struct!()

    # includes and exports
    @eval begin
        include(joinpath(@__DIR__, "stokes/Stokes.jl"))
        include(joinpath(@__DIR__, "boundaryconditions/BoundaryConditions.jl"))
        include(joinpath(@__DIR__, "Utils.jl"))

        export USE_GPU,
            PTArray, Velocity, SymmetricTensor, Residual, StokesArrays, PTStokesCoeffs
        export AbstractStokesModel, Viscous, ViscoElastic
        export pureshear_bc!, free_slip_x!, free_slip_y!, free_slip_z!, apply_free_slip!
        export smooth!, stress, solve!

        include(joinpath(@__DIR__, "stokes/Elasticity.jl"))
    end

    # conditional submodule load
    module_names = [Symbol("Elasticity$(N)D")]
    for m in module_names
        Base.@eval begin
            @reexport import .$m
        end
    end
end

function ps_reset!()
    Base.eval(Main, ParallelStencil.@reset_parallel_stencil)
    return Base.eval(@__MODULE__, ParallelStencil.@reset_parallel_stencil)
end
