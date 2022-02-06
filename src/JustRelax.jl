module JustRelax

# include("DiffusionNonlinearSolvers.jl")

using ParallelStencil

# PS.jl exports
# import ParallelStencil: @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand
# export @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand

export PS_Setup, environment!, PS_reset!

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
        USE_GPU = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : $gpu
    )

    # eval(Meta.parse("using ParallelStencil"))
    
    # call appropriate FD module
    eval(
        Meta.parse("using ParallelStencil.FiniteDifferences$(N)D")
    )

    # start ParallelStencil
    if model.device == :gpu 
        eval(
            Meta.parse("@init_parallel_stencil(CUDA, $T, $N)")
        )
    else
        eval(
            Meta.parse("@init_parallel_stencil(Threads, $T, $N)")
        )
    end

end

PS_reset!() = ParallelStencil.@reset_parallel_stencil()

end # module
