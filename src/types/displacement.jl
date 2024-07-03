function velocity2displacement!(stokes::JustRelax.StokesArrays, dt)
    velocity2displacement!(stokes, backend(stokes), dt)
    return nothing
end

function velocity2displacement!(stokes::JustRelax.StokesArrays, ::CPUBackendTrait, dt)
    return _velocity2displacement!(stokes, dt)
end

function _velocity2displacement!(stokes::JustRelax.StokesArrays, dt)
    ni = size(stokes.P)
    (; V, U) = stokes
    @parallel (@idx ni .+ 2) _velocity2displacement!(
        V.Vx, V.Vy, V.Vz, U.Ux, U.Uy, U.Uz, dt
    )
    return nothing
end

@parallel_indices (I...) function _velocity2displacement!(Vx, Vy, Vz, Ux, Uy, Uz, dt)
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

function displacement2velocity!(stokes::JustRelax.StokesArrays, dt)
    displacement2velocity!(stokes, backend(stokes), dt)
    return nothing
end

function displacement2velocity!(stokes::JustRelax.StokesArrays, ::CPUBackendTrait, dt)
    return _displacement2velocity!(stokes, dt)
end

function _displacement2velocity!(stokes::JustRelax.StokesArrays, dt)
    ni = size(stokes.P)
    (; V, U) = stokes
    @parallel (@idx ni .+ 2) _displacement2velocity!(U.Ux, U.Uy, U.Uz, V.Vx, V.Vy, V.Vz, 1 / dt)
    return nothing
end

@parallel_indices (I...) function _displacement2velocity!(Ux, Uy, Uz, Vx, Vy, Vz, _dt)
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
