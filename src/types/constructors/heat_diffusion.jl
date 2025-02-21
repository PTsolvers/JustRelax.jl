function ThermalArrays(::Type{CPUBackend}, ni::NTuple{N, Integer}) where {N}
    return ThermalArrays(ni...)
end

function ThermalArrays(::Type{CPUBackend}, ni::Vararg{Integer, N}) where {N}
    return ThermalArrays(ni...)
end

function ThermalArrays(nx::Integer, ny::Integer)
    T = @zeros(nx + 3, ny + 1)
    ΔT = @zeros(nx + 3, ny + 1)
    ΔTc = @zeros(nx, ny)
    adiabatic = @zeros(nx + 1, ny - 1)
    Told = @zeros(nx + 3, ny + 1)
    Tc = @zeros(nx, ny)
    H = @zeros(nx, ny)
    shear_heating = @zeros(nx, ny)
    dT_dt = @zeros(nx + 1, ny - 1)
    qTx = @zeros(nx + 2, ny - 1)
    qTy = @zeros(nx + 1, ny)
    qTx2 = @zeros(nx + 2, ny - 1)
    qTy2 = @zeros(nx + 1, ny)
    ResT = @zeros(nx + 1, ny - 1)
    return JustRelax.ThermalArrays(
        T,
        Tc,
        Told,
        ΔT,
        ΔTc,
        adiabatic,
        dT_dt,
        qTx,
        qTy,
        nothing,
        qTx2,
        qTy2,
        nothing,
        H,
        shear_heating,
        ResT,
    )
end

function ThermalArrays(nx::Integer, ny::Integer, nz::Integer)
    T = @zeros(nx + 1, ny + 1, nz + 1)
    ΔT = @zeros(nx + 1, ny + 1, nz + 1)
    ΔTc = @zeros(nx, ny, ny)
    adiabatic = @zeros(nx - 1, ny - 1, nz - 1)
    Told = @zeros(nx + 1, ny + 1, nz + 1)
    Tc = @zeros(nx, ny, nz)
    H = @zeros(nx, ny, nz)
    shear_heating = @zeros(nx, ny, nz)
    dT_dt = @zeros(nx - 1, ny - 1, nz - 1)
    qTx = @zeros(nx, ny - 1, nz - 1)
    qTy = @zeros(nx - 1, ny, nz - 1)
    qTz = @zeros(nx - 1, ny - 1, nz)
    qTx2 = @zeros(nx, ny - 1, nz - 1)
    qTy2 = @zeros(nx - 1, ny, nz - 1)
    qTz2 = @zeros(nx - 1, ny - 1, nz)
    ResT = @zeros(nx - 1, ny - 1, nz - 1)
    return JustRelax.ThermalArrays(
        T,
        Tc,
        Told,
        ΔT,
        ΔTc,
        adiabatic,
        dT_dt,
        qTx,
        qTy,
        qTz,
        qTx2,
        qTy2,
        qTz2,
        H,
        shear_heating,
        ResT,
    )
end
