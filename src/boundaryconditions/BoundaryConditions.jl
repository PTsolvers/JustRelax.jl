@inline bc_index(x::NTuple{2,T}) where {T} = mapreduce(xi -> max(size(xi)...), max, x)
@inline bc_index(x::T) where {T<:AbstractArray{<:Any,2}} = max(size(x)...)

@inline function bc_index(x::NTuple{3,T}) where {T}
    nx, ny, nz = size(x[1])
    return max((nx, ny), (ny, nz), (nx, nz))
end

@inline function bc_index(x::T) where {T<:AbstractArray{<:Any,3}}
    nx, ny, nz = size(x)
    return max((nx, ny), (ny, nz), (nx, nz))
end

@inline do_bc(bc) = reduce(|, values(bc))

"""
    thermal_bcs!(T, bcs::TemperatureBoundaryConditions)

Apply the prescribed heat boundary conditions `bc` on the `T`
"""
thermal_bcs!(thermal, bcs) = thermal_bcs!(backend(thermal), thermal, bcs)
function thermal_bcs!(
    ::CPUBackendTrait, thermal::JustRelax.ThermalArrays, bcs::TemperatureBoundaryConditions
)
    return thermal_bcs!(thermal.T, bcs)
end

function thermal_bcs!(T::AbstractArray, bcs::TemperatureBoundaryConditions)
    n = bc_index(T)

    # no flux boundary conditions
    do_bc(bcs.no_flux) && (@parallel (@idx n) free_slip!(T, bcs.no_flux))

    return nothing
end

"""
    flow_bcs!(stokes, bcs::FlowBoundaryConditions, di)

Apply the prescribed flow boundary conditions `bc` on the `stokes`
"""
flow_bcs!(stokes, bcs) = flow_bcs!(backend(stokes), stokes, bcs)
function flow_bcs!(::CPUBackendTrait, stokes, bcs)
    return _flow_bcs!(bcs, @velocity(stokes))
end
function flow_bcs!(bcs, V::Vararg{T,N}) where {T,N}
    return _flow_bcs!(bcs, tuple(V...))
end

function _flow_bcs!(bcs, V)
    n = bc_index(V)
    # no slip boundary conditions
    do_bc(bcs.no_slip) && (@parallel (@idx n) no_slip!(V..., bcs.no_slip))
    # free slip boundary conditions
    do_bc(bcs.free_slip) && (@parallel (@idx n) free_slip!(V..., bcs.free_slip))

    return nothing
end

# BOUNDARY CONDITIONS KERNELS
include("free_slip.jl")
include("free_surface.jl")
include("no_slip.jl")
include("pure_shear.jl")
