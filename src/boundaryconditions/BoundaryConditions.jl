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

# 2D KERNELS

@inline do_bc(bc) = reduce(|, values(bc))

function thermal_bcs!(T, bcs::TemperatureBoundaryConditions{_T,2}) where {_T}
    n = max(size(T)...)

    # no flux boundary conditions
    do_bc(bcs.no_flux) && (@parallel (1:n) free_slip!(T, bcs.no_flux))
    # periodic conditions
    do_bc(bcs.periodicity) && (@parallel (1:n) periodic_boundaries!(T, bcs.periodicity))

    return nothing
end

function flow_bcs!(bcs::FlowBoundaryConditions{T,2}, Vx, Vy, di) where {T}
    n = max(size(Vx)..., size(Vy)...)

    # no slip boundary conditions
    do_bc(bcs.no_slip) && (@parallel (1:n) no_slip!(Vx, Vy, bcs.no_slip, di...))
    # free slip boundary conditions
    do_bc(bcs.free_slip) && (@parallel (1:n) free_slip!(Vx, Vy, bcs.free_slip))
    # periodic conditions
    do_bc(bcs.periodicity) &&
        (@parallel (1:n) periodic_boundaries!(Vx, Vy, bcs.periodicity))

    return nothing
end

@parallel_indices (i) function no_slip!(Ax, Ay, bc, dx, dy)
    if bc.bot
        @inbounds (i ≤ size(Ay, 1)) && (Ay[i, 1] = 0.0)
        @inbounds (1 < i < size(Ax, 1)) && (Ax[i, 1] = Ax[i, 2] * 0.5 / dx)
    end
    if bc.top
        @inbounds (i ≤ size(Ay, 1)) && (Ay[i, end] = 0.0)
        @inbounds (1 < i < size(Ax, 1)) && (Ax[i, end] = Ax[i, end - 1] * 0.5 / dx)
    end
    if bc.left
        @inbounds (i ≤ size(Ax, 2)) && (Ax[1, i] = 0.0)
        @inbounds (1 < i < size(Ay, 2)) && (Ay[1, i] = Ay[2, i] * 0.5 / dy)
    end
    if bc.right
        @inbounds (i ≤ size(Ax, 2)) && (Ax[end, i] = 0.0)
        @inbounds (1 < i < size(Ay, 2)) && (Ay[end, i] = Ay[end - 1, i] * 0.5 / dy)
    end
    return nothing
end

@parallel_indices (i) function free_slip!(Ax, Ay, bc)
    if i ≤ size(Ax, 1)
        @inbounds bc.bot && (Ax[i, 1] = Ax[i, 2])
        @inbounds bc.top && (Ax[i, end] = Ax[i, end - 1])
    end
    if i ≤ size(Ay, 2)
        @inbounds bc.left && (Ay[1, i] = Ay[2, i])
        @inbounds bc.right && (Ay[end, i] = Ay[end - 1, i])
    end
    return nothing
end

@parallel_indices (i) function free_slip!(T, bc)
    if i ≤ size(T, 1)
        @inbounds bc.bot && (T[i, 1] = T[i, 2])
        @inbounds bc.top && (T[i, end] = T[i, end - 1])
    end
    if i ≤ size(T, 2)
        @inbounds bc.left && (T[1, i] = T[2, i])
        @inbounds bc.right && (T[end, i] = T[end - 1, i])
    end
    return nothing
end

@parallel_indices (i) function periodic_boundaries!(Ax, Ay, bc)
    if i ≤ size(Ax, 1)
        @inbounds bc.bot && (Ax[i, 1] = Ax[i, end - 1])
        @inbounds bc.top && (Ax[i, end] = Ax[i, 2])
    end
    if i ≤ size(Ay, 2)
        @inbounds bc.left && (Ay[1, i] = Ay[end - 1, i])
        @inbounds bc.right && (Ay[end, i] = Ay[2, i])
    end
    return nothing
end

@parallel_indices (i) function periodic_boundaries!(T, bc)
    if i ≤ size(T, 1)
        @inbounds bc.bot && (T[i, 1] = T[i, end - 1])
        @inbounds bc.top && (T[i, end] = T[i, 2])
    end
    if i ≤ size(T, 2)
        @inbounds bc.left && (T[1, i] = T[end - 1, i])
        @inbounds bc.right && (T[end, i] = T[2, i])
    end
    return nothing
end

function pureshear_bc!(
    stokes::StokesArrays, xci::NTuple{2,T}, xvi::NTuple{2,T}, εbg
) where {T}
    # unpack
    # Vx, Vy = stokes.V.Vx, stokes.V.Vy
    # dx, dy = di
    # lx, ly = li
    # Velocity pure shear boundary conditions
    # stokes.V.Vx .= PTArray([
    #     -εbg * ((i - 1) * dx - 0.5 * lx) for i in 1:size(Vx, 1), j in 1:size(Vx, 2)
    # ])
    # stokes.V.Vy .= PTArray([
    #     εbg * ((j - 1) * dy - 0.5 * ly) for i in 1:size(Vy, 1), j in 1:size(Vy, 2)
    # ])

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

# function apply_free_slip!(freeslip::NamedTuple{<:Any,NTuple{2,T}}, Vx, Vy) where {T}
#     freeslip_x, freeslip_y = freeslip
#     # free slip boundary conditions
#     freeslip_x && (@parallel (1:size(Vy, 2)) free_slip_x!(Vy))

#     freeslip_y && (@parallel (1:size(Vx, 1)) free_slip_y!(Vx))

#     return nothing
# end

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
