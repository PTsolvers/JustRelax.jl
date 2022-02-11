module JustRelax

# include("DiffusionNonlinearSolvers.jl")

using ParallelStencil
using LinearAlgebra
using Printf

# PS.jl exports
# import ParallelStencil: @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand
# export @parallel, @hide_communication, @parallel_indices, @parallel_async, @synchronize, @zeros, @ones, @rand

export PS_Setup, Geometry

export environment!, PS_reset!
struct PS_Setup{B,C}
    device::Symbol

    function PS_Setup(device::Symbol, precission::DataType, nDim::Integer)
        new{precission, nDim}(device)
    end
end

struct Geometry{nDim}
    ni::NTuple{nDim, Integer}
    li::NTuple{nDim, Float64}
    max_li::Float64
    di::NTuple{nDim, Float64}
    xci::NTuple{nDim, StepRangeLen}
    xvi::NTuple{nDim, StepRangeLen}

    function Geometry(ni::NTuple{nDim, Integer}, li::NTuple{nDim, T}) where {nDim, T}
        li isa NTuple{nDim, Float64} == false && (li = Float64.(li))
        di = li./ni
        new{nDim}(
            ni,
            li,
            Float64(max(li...)),
            di,
            Tuple([di[i]/2:di[i]:(li[i]-di[i]/2) for i in 1:nDim]),
            Tuple([0:di[i]:li[i] for i in 1:nDim])
        )
    end

end

function environment!(model::PS_Setup{T, N}) where {T, N}
    gpu = model.device == :gpu ? true : false

    # environment variable for XPU
    @eval(
        const USE_GPU = haskey(ENV, "USE_GPU"    ) ? parse(Bool, ENV["USE_GPU"]    ) : $gpu
    )
    
    # call appropriate FD module
    eval(
        Meta.parse("using ParallelStencil.FiniteDifferences$(N)D")
    )

    # start ParallelStencil
    local PTArray
    if model.device == :gpu 
        eval(
            Meta.parse("@init_parallel_stencil(CUDA, $T, $N)")
        )
        eval(
            :(const PTArray =  CUDA.CuArray{$T, $N})
        )
    else
        eval(
            Meta.parse("@init_parallel_stencil(Threads, $T, $N)")
        )
        eval(
            :(const PTArray = Array{$T, $N})
        )
    end

    # create array structs
    make_velocity_struct!(N) # velocity
    make_tensor_struct!(N) # (symmetric) tensors
    make_residual_struct!(N) # residuals
    make_stokes_struct!() # Arrays for Stokes solver
    make_PTstokes_struct!()

    eval(
        :(include(joinpath(pwd(),"src/stokes/Stokes.jl")))
    )

    eval(
        :(include(joinpath(pwd(),"src/boundaryconditions/BoundaryConditions.jl")))
    )

    eval(
        :(export USE_GPU, PTArray, SymmetricTensor, Residual, StokesArrays, PTStokesCoeffs, pureshear_bc!, smooth!, solve!)
    )

end

function make_velocity_struct!(nDim::Integer)
    dims = ("x", "y", "z")
    str = Meta.parse(
        "struct Velocity{T} \n"*
        join("V$(dims[i])::T\n" for i in 1:nDim)*
        "end"
    )
    eval(str)
end

function make_tensor_struct!(nDim::Integer)
    dims = ("x", "y", "z")
    str = Meta.parse(
        "struct SymmetricTensor{T} \n"*
        join("$(dims[i])$(dims[j])::T\n" for i in 1:nDim, j in 1:nDim if j≥i)*
        "end"
    )
    eval(str)
end

function make_residual_struct!(nDim::Integer)
    dims = ("x", "y", "z")
    str = Meta.parse(
        "struct Residual{T} \n"*
        join("R$(dims[i])::T\n" for i in 1:nDim)*
        "end"
    )
    eval(str)
end

make_stokes_struct!() =
    eval(
        Meta.parse(
        "struct StokesArrays{A,B,C,T, nDim}
            P::T
            V::A
            dV::A
            ∇V::T
            τ::B
            R::C
            
            function StokesArrays(ni::NTuple{2, T}) where T
                P = @zeros(ni...)
                ∇V = @zeros(ni...)
                V = Velocity(
                    @zeros(ni[1]+1, ni[2]),
                    @zeros(ni[1], ni[2]+1)
                )
                τ = SymmetricTensor(
                    @zeros(ni...), # xx
                    @zeros(ni[1]-1, ni[2]-1), # xy
                    @zeros(ni...) # yy
                )
                dV = Velocity(
                    @zeros(ni[1]-1, ni[2]-2),
                    @zeros(ni[1]-2, ni[2]-1)
                )

                R = Residual(
                    @zeros(ni[1]-1, ni[2]-2),
                    @zeros(ni[1]-2, ni[2]-1)
                )

                new{typeof(V),typeof(τ),typeof(R),typeof(P), 2}(P, V, dV, ∇V, τ, R)
            end
        end"
        )
    )

    
make_PTstokes_struct!() =
    eval(
        Meta.parse(
        "struct PTStokesCoeffs{T, nDim}
            CFL::T
            ϵ::T # PT tolerance
            Re::T # Reynolds Number
            r::T # 
            Vpdτ::T
            dτ_Rho::AbstractArray{T, nDim} 
            Gdτ::AbstractArray{T, nDim}
        
            function PTStokesCoeffs(ni::NTuple{nDim, T}, di; 
                ϵ = 1e-8, Re = 5π, CFL = 0.9/√2, r=1e0) where {nDim, T}
            
                Vpdτ = min(di...)*CFL
                Gdτ = @zeros(ni...)
                dτ_Rho = @zeros(ni...)
        
                new{eltype(Gdτ), nDim}(CFL,ϵ,Re,r,Vpdτ,dτ_Rho,Gdτ)
            end
        
        end"
        )
)

PS_reset!() = ParallelStencil.@reset_parallel_stencil()

end # module