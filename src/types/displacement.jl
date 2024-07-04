# Velocity to displacement interpolation
velocity2displacement!(stokes, dt) = velocity2displacement!(backend(stokes), stokes, dt)


function velocity2displacement!(::CPUBackendTrait, stokes::JustRelax.StokesArrays, dt)
    return _velocity2displacement!(stokes, dt)
end

function _velocity2displacement!(stokes::JustRelax.StokesArrays, dt)
    ni = size(stokes.P)
    (; V, U) = stokes
    @parallel (@idx ni .+ 2) _velocity2displacement_kernel!(
        V.Vx, V.Vy, V.Vz, U.Ux, U.Uy, U.Uz, dt
    )
    return nothing
end

@parallel_indices (I...) function _velocity2displacement_kernel!(Vx, Vy, Vz, Ux, Uy, Uz, dt)
    if all(I .≤ size(Ux))
        Ux[I...] = Vx[I...] * dt
    end
    if all(I .≤ size(Uy))
        Uy[I...] = Vy[I...] * dt
    end
    if !isnothing(Vz) && all(I .≤ size(Uz))
        Uz[I...] = Vz[I...] * dt
    end
    return nothing
end

# Displacement to velocity interpolation

displacement2velocity!(stokes, dt) =  displacement2velocity!(backend(stokes), stokes::JustRelax.StokesArrays, dt)

function displacement2velocity!(::CPUBackendTrait, stokes::JustRelax.StokesArrays, dt)
    return _displacement2velocity!(stokes, dt)
end

function _displacement2velocity!(stokes::JustRelax.StokesArrays, dt)
    ni = size(stokes.P)
    (; V, U) = stokes
    @parallel (@idx ni .+ 2) _displacement2velocity_kernel!(U.Ux, U.Uy, U.Uz, V.Vx, V.Vy, V.Vz, 1 / dt)
    return nothing
end

@parallel_indices (I...) function _displacement2velocity_kernel!(Ux, Uy, Uz, Vx, Vy, Vz, _dt)
    if all(I .≤ size(Ux))
        Vx[I...] = Ux[I...] * _dt
    end
    if all(I .≤ size(Uy))
        Vy[I...] = Uy[I...] * _dt
    end
    if !isnothing(Vz) && all(I .≤ size(Uz))
        Vz[I...] = Uz[I...] * _dt
    end
    return nothing
end

function displacement2velocity!(stokes,dt, flow_bcs::AbstractFlowBoundaryConditions)
    if typeof(flow_bcs) <: DisplacementBoundaryConditions
        displacement2velocity!(backend(stokes), stokes, dt)
        return nothing
    elseif typeof(flow_bcs) <: VelocityBoundaryConditions
        return nothing
    end
end
