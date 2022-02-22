struct PS_Setup{B,C}
    device::Symbol

    function PS_Setup(device::Symbol, precission::DataType, nDim::Integer)
        new{precission, nDim}(device)
    end
end

function environment!(model::PS_Setup{T, N}) where {T, N}

    gpu = model.device == :gpu ? true : false

    # environment variable for XPU
    @eval(
        const USE_GPU = haskey(ENV, "USE_GPU") ? parse(Bool, ENV["USE_GPU"]) : $gpu
    )
    
    # call appropriate FD module
    eval(
        Meta.parse("using ParallelStencil.FiniteDifferences$(N)D")
    )

    Base.eval(@__MODULE__, Meta.parse("using ParallelStencil.FiniteDifferences$(N)D") ) 

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
            :( PTArray =  CUDA.CuArray{$T, $N})
        )
    else
        @eval begin
            @init_parallel_stencil(Threads, $T, $N)
            PTArray = Array{$T, $N}
        end
    end
    
    # create array structs
    make_velocity_struct!(N) # velocity
    make_tensor_struct!(N) # (symmetric) tensors
    make_residual_struct!(N) # residuals
    make_stokes_struct!() # Arrays for Stokes solver
    make_PTstokes_struct!()

    @eval begin
        include(joinpath(@__DIR__,"../stokes/Stokes.jl"))
        include(joinpath(@__DIR__,"../boundaryconditions/BoundaryConditions.jl"))
    
        export USE_GPU, PTArray, SymmetricTensor, Residual, StokesArrays, PTStokesCoeffs, pureshear_bc!, smooth!, solve!
    end

end

function ps_reset!()
    Base.eval(Main, ParallelStencil.@reset_parallel_stencil)
    Base.eval(@__MODULE__, ParallelStencil.@reset_parallel_stencil)
end