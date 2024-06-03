@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, Test

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    CPUBackend
end

@testset "Array conversions" begin
    ni      = 2, 2
    stokes  = StokesArrays(backend, ni)
    thermal = ThermalArrays(backend, ni)
    A1      = Array(stokes.V)
    A2      = Array(stokes.τ)
    A3      = Array(stokes.R)
    A4      = Array(stokes.P)
    A5      = Array(stokes)
    A6      = Array(thermal)

    @test typeof(A1) <: JustRelax.Velocity{<:Array}
    @test typeof(A2) <: JustRelax.SymmetricTensor{<:Array}
    @test typeof(A3) <: JustRelax.Residual{<:Array}
    @test typeof(A4) <: Array
    @test typeof(A5) <: JustRelax.StokesArrays
    @test typeof(A6) <: JustRelax.ThermalArrays{<:Array}
end

# const JR_T = Union{
#     JustRelax.StokesArrays,
#     JustRelax.SymmetricTensor,
#     JustRelax.ThermalArrays,
#     JustRelax.Velocity,
#     JustRelax.Residual
# }

# @inline remove_parameters(::T) where {T} = Base.typename(T).wrapper

# foo(x::T) where {T} = foo(JustRelax.backend(x), x)
# foo(x::Array) = x
# foo(::Nothing) = nothing

# function foo(::Any, x::T) where {T<:JR_T}
#     nfields = fieldcount(T)
#     cpu_fields = ntuple(Val(nfields)) do i
#         Base.@_inline_meta
#         @show i
#         foo(getfield(x, i))
#     end
#     T_clean = remove_parameters(x)
#     return T_clean(cpu_fields...)
# end

# foo(stokes.V)

# pot = 29
