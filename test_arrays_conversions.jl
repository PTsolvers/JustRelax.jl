using JustRelax, Test
model = PS_Setup(:Threads, Float64, 2)
environment!(model)

# @testset "Array conversions" begin
    ni = 10, 10
    stokes  = StokesArrays(ni, ViscoElastic)
    thermal = ThermalArrays(ni)

    @test Array(stokes.V) isa Velocity{Array{T, N}} where {T, N}
    @test Array(stokes.τ) isa SymmetricTensor{Array{T, N}} where {T, N}
    @test Array(stokes.R) isa Residual{Array{T, N}} where {T, N}
    @test Array(stokes.P) isa Array{T, N} where {T, N}
    @test Array(stokes)   isa StokesArrays
    @test Array(thermal)  isa ThermalArrays{Array{T, N}} where {T, N}
# end

# abstract type DeviceTrait end
# struct CPUDeviceTrait <: DeviceTrait end
# struct NonCPUDeviceTrait <: DeviceTrait end

# @inline iscpu(::Array) = CPUDeviceTrait()
# @inline iscpu(::Type{Array}) = CPUDeviceTrait()
# @inline iscpu(::AbstractArray) = NonCPUDeviceTrait()
# @inline iscpu(::T) where T = throw(ArgumentError("Unkown device"))

# @inline iscpu(::Velocity{Array{T, N}}) where {T, N} = CPUDeviceTrait()
# @inline iscpu(::Velocity{AbstractArray{T, N}}) where {T, N} = NonCPUDeviceTrait()

# @inline iscpu(::SymmetricTensor{Array{T, N}}) where {T, N} = CPUDeviceTrait()
# @inline iscpu(::SymmetricTensor{AbstractArray{T, N}}) where {T, N} = NonCPUDeviceTrait()

# @inline iscpu(::Residual{Array{T, N}}) where {T, N} = CPUDeviceTrait()
# @inline iscpu(::Residual{AbstractArray{T, N}}) where {T, N} = NonCPUDeviceTrait()

# @inline iscpu(::ThermalArrays{Array{T, N}}) where {T, N} = CPUDeviceTrait()
# @inline iscpu(::ThermalArrays{AbstractArray{T, N}}) where {T, N} = NonCPUDeviceTrait()

# @inline iscpu(::StokesArrays{M,A,B,C,Array{T, N},nDim}) where {M,A,B,C,T,N,nDim} =CPUDeviceTrait()
# @inline iscpu(::StokesArrays{M,A,B,C,AbstractArray{T, N},nDim}) where {M,A,B,C,T,N,nDim} =NonCPUDeviceTrait()


@test JustRelax.iscpu(stokes.V) isa JustRelax.CPUDeviceTrait
@test JustRelax.iscpu(stokes.τ) isa JustRelax.CPUDeviceTrait
@test JustRelax.iscpu(stokes.R) isa JustRelax.CPUDeviceTrait
@test JustRelax.iscpu(stokes.P) isa JustRelax.CPUDeviceTrait
@test JustRelax.iscpu(stokes)   isa JustRelax.CPUDeviceTrait
@test JustRelax.iscpu(thermal)  isa JustRelax.CPUDeviceTrait
@test_throws ArgumentError("Unkown device") JustRelax.iscpu("potato")

# import Base.Array

# @inline remove_parameters(::T) where T = Base.typename(T).wrapper

# function Array(x::T) where T<:Union{SymmetricTensor, ThermalArrays, Velocity, Residual}
#     Array(iscpu(x), x)
# end

# Array(::CPUDeviceTrait, x) = x

# function Array(::NonCPUDeviceTrait, x::T) where T<:Union{SymmetricTensor, ThermalArrays, Velocity, Residual}
#     nfields = fieldcount(T)
#     cpu_fields = ntuple(Val(nfields)) do i
#         Base.@_inline_meta
#         Array(getfield(x, i))
#     end
#     T_clean = remove_parameters(x)
#     return T_clean(cpu_fields...)
# end

# function Array(::NonCPUDeviceTrait, x::StokesArrays{T,A,B,C,M,nDim}) where {T,A,B,C,M,nDim}
#     nfields = fieldcount(StokesArrays)
#     cpu_fields = ntuple(Val(nfields)) do i
#         Base.@_inline_meta
#         Array(getfield(x, i))
#     end
#     T_clean = remove_parameters(x)
#     return T_clean(cpu_fields...)
# end

@edit Array(stokes)

using Chairmarks

abstract type AbstractTrait end
struct ConcreteTrait <: AbstractTrait end
foo(::Array) = ConcreteTrait()
foo(::Type{Array{T,N}}) where {T,N} = ConcreteTrait()

foo1(::Type{Array{Float64,2}}) = ConcreteTrait()


T=Array{Float64,2}

foo2(::Type{T}) = ConcreteTrait()
@b foo2($T)

foo(::Type{Array{T,N}}) where {T,N} = ConcreteTrait()
@b foo($T)

foo3(::Type{T}) where T<:Array = ConcreteTrait()

x=rand(2,2)
T=Array{Float64,2}

@b foo($T)
@btime foo($T)
@b foo1($T)
@b foo2($T)
@b foo3($T)