struct PS_Setup{B,C}
    device::Symbol

    function PS_Setup(device::Symbol, precission::DataType, nDim::Integer)
        new{precission, nDim}(device)
    end
end

function environment!(model::PS_Setup{T, N}) where {T, N}

    gpu = model.device == :gpu ? true : false

    # environment variable for XPU
    @eval begin
        const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : $gpu 
        if USE_GPU # select one GPU per MPI local rank (if >1 GPU per node)
            select_device()
        end
    end
    
    # call appropriate FD module
    eval(
        Meta.parse("using ParallelStencil.FiniteDifferences$(N)D")
    )

    Base.eval( @__MODULE__, Meta.parse("using ParallelStencil.FiniteDifferences$(N)D") ) 

    # start ParallelStencil
    global PTArray
    if model.device == :gpu 
        eval(
            :(@init_parallel_stencil(CUDA, $T, $N) )
        )
        Base.eval(
            Main, Meta.parse("using CUDA") 
        ) 
        eval(
            :(PTArray =  CUDA.CuArray{$T, $N})
        )
    else
        @eval begin
            @init_parallel_stencil(Threads, $T, $N)
            PTArray = Array{$T, $N}
        end
    end
    
    # create array structs
    make_velocity_struct!(N) # velocity
    make_symmetrictensor_struct!(N) # (symmetric) tensors
    make_residual_struct!(N) # residuals
    make_stokes_struct!() # Arrays for Stokes solver
    make_PTstokes_struct!()

    @eval begin
        include(joinpath(@__DIR__,"stokes/Stokes.jl"))
        include(joinpath(@__DIR__,"stokes/Elasticity.jl"))
        include(joinpath(@__DIR__,"boundaryconditions/BoundaryConditions.jl"))
        include(joinpath(@__DIR__,"Macros.jl"))
    
        export USE_GPU, PTArray, Velocity, SymmetricTensor, Residual, StokesArrays, PTStokesCoeffs, smooth!, solve!, stress
        export AbstractStokesModel, Viscous, ViscoElastic
        export pureshear_bc!, free_slip_x!, free_slip_y!, free_slip_z!, apply_free_slip!
    end

end

function ps_reset!()
    Base.eval(Main, ParallelStencil.@reset_parallel_stencil)
    Base.eval(@__MODULE__, ParallelStencil.@reset_parallel_stencil)
end
