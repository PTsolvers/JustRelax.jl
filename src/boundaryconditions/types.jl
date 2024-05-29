abstract type AbstractBoundaryConditions end

struct TemperatureBoundaryConditions{T,nD} <: AbstractBoundaryConditions
    no_flux::T

    function TemperatureBoundaryConditions(;
        no_flux::T=(left=true, right=false, top=false, bot=false)
    ) where {T}
        nD = length(no_flux) == 4 ? 2 : 3
        return new{T,nD}(no_flux)
    end
end

struct FlowBoundaryConditions{T,nD} <: AbstractBoundaryConditions
    no_slip::T
    free_slip::T
    free_surface::Bool

    function FlowBoundaryConditions(;
        no_slip::T=(left=false, right=false, top=false, bot=false),
        free_slip::T=(left=true, right=true, top=true, bot=true),
        free_surface::Bool=false,
    ) where {T}
        @assert length(no_slip) === length(free_slip)
        check_flow_bcs(no_slip, free_slip)

        nD = length(no_slip) == 4 ? 2 : 3
        return new{T,nD}(no_slip, free_slip, free_surface)
    end
end

function check_flow_bcs(no_slip::T, free_slip::T) where T
    v1 = values(no_slip)
    v2 = values(free_slip)
    k  = keys(no_slip)
    for (v1, v2, k) in zip(v1, v2, k)
        if v1 == v2
            error("Incompatible boundary conditions. The $k boundary condition can't be the same for no_slip and free_slip")
        end
    end
end