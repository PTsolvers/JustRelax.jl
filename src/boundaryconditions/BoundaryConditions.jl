abstract type AbstractBoundaryConditions end

struct TemperatureBoundaryConditions{T,nD} <: AbstractBoundaryConditions
    no_flux::T
    periodicity::T

    function TemperatureBoundaryConditions(;
        no_flux::T=(left=true, right=false, top=false, bot=false),
        periodicity::T=(left=false, right=false, top=false, bot=false),
    ) where {T}
        @assert length(no_flux) === length(periodicity)
        nD = length(no_flux) == 4 ? 2 : 3
        return new{T,nD}(no_flux, periodicity)
    end
end

struct FlowBoundaryConditions{T,nD} <: AbstractBoundaryConditions
    no_slip::T
    free_slip::T
    periodicity::T

    function FlowBoundaryConditions(;
        no_slip::T=(left=false, right=false, top=false, bot=false),
        free_slip::T=(left=true, right=true, top=true, bot=true),
        periodicity::T=(left=false, right=false, top=false, bot=false),
    ) where {T}
        @assert length(no_slip) === length(free_slip) === length(periodicity)
        nD = length(no_slip) == 4 ? 2 : 3
        return new{T,nD}(no_slip, free_slip, periodicity)
    end
end

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
function thermal_bcs!(T, bcs::TemperatureBoundaryConditions)
    n = bc_index(T)

    # no flux boundary conditions
    do_bc(bcs.no_flux) && (@parallel (@idx n) free_slip!(T, bcs.no_flux))
    # periodic conditions
    do_bc(bcs.periodicity) && (@parallel (@idx n) periodic_boundaries!(T, bcs.periodicity))

    return nothing
end

"""
    flow_bcs!(stokes, bcs::FlowBoundaryConditions, di) 

Apply the prescribed flow boundary conditions `bc` on the `stokes` 
"""
function _flow_bcs!(bcs::FlowBoundaryConditions, V)
    n = bc_index(V)

    # no slip boundary conditions
    do_bc(bcs.no_slip) && (@parallel (@idx n) no_slip!(V..., bcs.no_slip))
    # free slip boundary conditions
    do_bc(bcs.free_slip) && (@parallel (@idx n) free_slip!(V..., bcs.free_slip))
    # periodic conditions
    do_bc(bcs.periodicity) &&
        (@parallel (@idx n) periodic_boundaries!(V..., bcs.periodicity))

    return nothing
end

flow_bcs!(stokes, bcs::FlowBoundaryConditions) = _flow_bcs!(bcs, @velocity(stokes))
function flow_bcs!(bcs::FlowBoundaryConditions, V::Vararg{T,N}) where {T,N}
    return _flow_bcs!(bcs, tuple(V...))
end

# BOUNDARY CONDITIONS KERNELS

@parallel_indices (i) function no_slip!(Ax, Ay, bc)
    @inbounds begin
        if bc.bot
            (i ≤ size(Ay, 1)) && (Ay[i, 1] = 0.0)
            (1 < i < size(Ax, 1)) && (Ax[i, 2] = Ax[i, 3] / 3)
        end
        if bc.top
            (i ≤ size(Ay, 1)) && (Ay[i, end] = 0.0)
            (1 < i < size(Ax, 1)) && (Ax[i, end - 1] = Ax[i, end - 2] / 3)
        end
        if bc.left
            (i ≤ size(Ax, 2)) && (Ax[1, i] = 0.0)
            (1 < i < size(Ay, 2)) && (Ay[2, i] = Ay[3, i] / 3)
        end
        if bc.right
            (i ≤ size(Ax, 2)) && (Ax[end, i] = 0.0)
            (1 < i < size(Ay, 2)) && (Ay[end - 1, i] = Ay[end - 2, i] / 3)
        end
    end
    return nothing
end

@parallel_indices (i) function free_slip!(Ax, Ay, bc)
    @inbounds begin
        if i ≤ size(Ax, 1)
            bc.bot && (Ax[i, 1] = Ax[i, 2])
            bc.top && (Ax[i, end] = Ax[i, end - 1])
        end
        if i ≤ size(Ay, 2)
            bc.left && (Ay[1, i] = Ay[2, i])
            bc.right && (Ay[end, i] = Ay[end - 1, i])
        end
    end
    return nothing
end

@parallel_indices (i, j) function free_slip!(Ax, Ay, Az, bc)
    @inbounds begin
        # free slip in the front and back XZ planes
        if bc.front
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 3)
                Ax[i, 1, j] = Ax[i, 2, j]
            end
            if i ≤ size(Az, 1) && j ≤ size(Az, 3)
                Az[i, 1, j] = Az[i, 2, j]
            end
        end
        if bc.back
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 3)
                Ax[i, end, j] = Ax[i, end - 1, j]
            end
            if i ≤ size(Az, 1) && j ≤ size(Az, 3)
                Az[i, end, j] = Az[i, end - 1, j]
            end
        end
        # free slip in the front and back XY planes
        if bc.top
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
                Ax[i, j, 1] = Ax[i, j, 2]
            end
            if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
                Ay[i, j, 1] = Ay[i, j, 2]
            end
        end
        if bc.bot
            if i ≤ size(Ax, 1) && j ≤ size(Ax, 2)
                Ax[i, j, end] = Ax[i, j, end - 1]
            end
            if i ≤ size(Ay, 1) && j ≤ size(Ay, 2)
                Ay[i, j, end] = Ay[i, j, end - 1]
            end
        end
        # free slip in the front and back YZ planes
        if bc.left
            if i ≤ size(Ay, 2) && j ≤ size(Ay, 3)
                Ay[1, i, j] = Ay[2, i, j]
            end
            if i ≤ size(Az, 2) && j ≤ size(Az, 3)
                Az[1, i, j] = Az[2, i, j]
            end
        end
        if bc.right
            if i ≤ size(Ay, 2) && j ≤ size(Ay, 3)
                Ay[end, i, j] = Ay[end - 1, i, j]
            end
            if i ≤ size(Az, 2) && j ≤ size(Az, 3)
                Az[end, i, j] = Az[end - 1, i, j]
            end
        end
    end
    return nothing
end

@parallel_indices (i) function free_slip!(T::_T, bc) where {_T<:AbstractArray{<:Any,2}}
    @inbounds begin
        if i ≤ size(T, 1)
            bc.bot && (T[i, 1] = T[i, 2])
            bc.top && (T[i, end] = T[i, end - 1])
        end
        if i ≤ size(T, 2)
            bc.left && (T[1, i] = T[2, i])
            bc.right && (T[end, i] = T[end - 1, i])
        end
    end
    return nothing
end

@parallel_indices (i, j) function free_slip!(T::_T, bc) where {_T<:AbstractArray{<:Any,3}}
    nx, ny, nz = size(T)
    @inbounds begin
        if i ≤ nx && j ≤ ny
            bc.bot && (T[i, j, 1] = T[i, j, 2])
            bc.top && (T[i, j, end] = T[i, j, end - 1])
        end
        if i ≤ ny && j ≤ nz
            bc.left && (T[1, i, j] = T[2, i, j])
            bc.right && (T[end, i, j] = T[end - 1, i, j])
        end
        if i ≤ nx && j ≤ nz
            bc.front && (T[i, 1, j] = T[i, 2, j])
            bc.back && (T[i, end, j] = T[i, end - 1, j])
        end
    end
    return nothing
end

@parallel_indices (i) function periodic_boundaries!(Ax, Ay, bc)
    @inbounds begin
        if i ≤ size(Ax, 1)
            bc.bot && (Ax[i, 1] = Ax[i, end - 1])
            bc.top && (Ax[i, end] = Ax[i, 2])
        end
        if i ≤ size(Ay, 2)
            bc.left && (Ay[1, i] = Ay[end - 1, i])
            bc.right && (Ay[end, i] = Ay[2, i])
        end
    end
    return nothing
end

@parallel_indices (i) function periodic_boundaries!(
    T::_T, bc
) where {_T<:AbstractArray{<:Any,2}}
    @inbounds begin
        if i ≤ size(T, 1)
            bc.bot && (T[i, 1] = T[i, end - 1])
            bc.top && (T[i, end] = T[i, 2])
        end
        if i ≤ size(T, 2)
            bc.left && (T[1, i] = T[end - 1, i])
            bc.right && (T[end, i] = T[2, i])
        end
    end
    return nothing
end

@parallel_indices (i, j) function periodic_boundaries!(
    T::_T, bc
) where {_T<:AbstractArray{<:Any,3}}
    nx, ny, nz = size(T)
    @inbounds begin
        if i ≤ nx && j ≤ ny
            bc.bot && (T[i, j, 1] = T[i, j, end - 1])
            bc.top && (T[i, j, end] = T[i, j, 2])
        end
        if i ≤ ny && j ≤ nz
            bc.left && (T[1, i, j] = T[end - 1, i, j])
            bc.right && (T[end, i, j] = T[2, i, j])
        end
        if i ≤ nx && j ≤ nz
            bc.front && (T[i, 1, j] = T[i, end - 1, j])
            bc.back && (T[i, end, j] = T[i, 2, j])
        end
    end
    return nothing
end

function pureshear_bc!(
    stokes::StokesArrays, xci::NTuple{2,T}, xvi::NTuple{2,T}, εbg
) where {T}
    stokes.V.Vx[:, 2:(end - 1)] .= PTArray([εbg * x for x in xvi[1], y in xci[2]])
    stokes.V.Vy[2:(end - 1), :] .= PTArray([-εbg * y for x in xci[1], y in xvi[2]])

    return nothing
end

@parallel_indices (j) function free_slip_x!(A::AbstractArray{eltype(PTArray),2})
    A[1, j] = A[2, j]
    A[end, j] = A[end - 1, j]
    return nothing
end

@parallel_indices (i) function free_slip_y!(A::AbstractArray{eltype(PTArray),2})
    A[i, 1] = A[i, 2]
    A[i, end] = A[i, end - 1]
    return nothing
end

@inbounds @parallel_indices (i) function _apply_free_slip!(Ax, Ay, freeslip_x, freeslip_y)
    if freeslip_x && i ≤ size(Ax, 1)
        Ax[i, 1] = Ax[i, 2]
        Ax[i, end] = Ax[i, end - 1]
    end
    if freeslip_y && i ≤ size(Ay, 2)
        Ay[1, i] = Ay[2, i]
        Ay[end, i] = Ay[end - 1, i]
    end
    return nothing
end

@parallel_indices (ix) function zero_y!(A::AbstractArray{eltype(PTArray),2})
    A[ix, 1] = 0.0
    A[ix, end] = 0.0
    return nothing
end

function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, Vx, Vy) where {T}
    freeslip_x, freeslip_y = freeslip
    n = max(size(Vx, 1), size(Vy, 2))
    # free slip boundary conditions
    @parallel (1:n) _apply_free_slip!(Vx, Vy, freeslip_x, freeslip_y)

    return nothing
end

function thermal_boundary_conditions!(
    insulation::NamedTuple, T::AbstractArray{_T,2}
) where {_T}
    insulation_x, insulation_y = insulation

    nx, ny = size(T)

    insulation_x && (@parallel (1:ny) free_slip_x!(T))
    insulation_y && (@parallel (1:nx) free_slip_y!(T))

    return nothing
end

# 3D KERNELS

@parallel_indices (j, k) function free_slip_x!(A::AbstractArray{T,3}) where {T}
    A[1, j, k] = A[2, j, k]
    A[end, j, k] = A[end - 1, j, k]
    return nothing
end

@parallel_indices (i, k) function free_slip_y!(A::AbstractArray{T,3}) where {T}
    A[i, 1, k] = A[i, 2, k]
    A[i, end, k] = A[i, end - 1, k]
    return nothing
end

@parallel_indices (i, j) function free_slip_z!(A::AbstractArray{T,3}) where {T}
    A[i, j, 1] = A[i, j, 2]
    A[i, j, end] = A[i, j, end - 1]
    return nothing
end

function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{3,T}}, Vx, Vy, Vz) where {T}
    freeslip_x, freeslip_y, freeslip_z = freeslip
    # free slip boundary conditions
    if freeslip_x
        @parallel (1:size(Vy, 2), 1:size(Vy, 3)) free_slip_x!(Vy)
        @parallel (1:size(Vz, 2), 1:size(Vz, 3)) free_slip_x!(Vz)
    end
    if freeslip_y
        @parallel (1:size(Vx, 1), 1:size(Vx, 3)) free_slip_y!(Vx)
        @parallel (1:size(Vz, 1), 1:size(Vz, 3)) free_slip_y!(Vz)
    end
    if freeslip_z
        @parallel (1:size(Vx, 1), 1:size(Vx, 2)) free_slip_z!(Vx)
        @parallel (1:size(Vy, 1), 1:size(Vy, 2)) free_slip_z!(Vy)
    end
end

function thermal_boundary_conditions!(
    insulation::NamedTuple, T::AbstractArray{_T,3}
) where {_T}
    # vertical boundaries
    frontal, lateral = insulation

    nx, ny, nz = size(T)

    frontal && (@parallel (1:nx, 1:nz) free_slip_y!(T))
    lateral && (@parallel (1:ny, 1:nz) free_slip_x!(T))

    return nothing
end
