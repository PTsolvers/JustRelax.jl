"""
    ThermalArrays(ni::NTuple{N, Integer}) where {N}

Create the thermal arrays for the heat diffusion solver in 2D or 3D with the extents given by ni (`nx x ny` or `nx x ny x nz``).
"""
function ThermalArrays(::Type{CPUBackend}, ni::NTuple{N, Integer}) where {N}
    return ThermalArrays(ni...)
end

"""
    ThermalArrays(::Backend, ni::NTuple{N, Integer}) where {N}

Internal entry point function for the ThermalArrays constructor. This allows for dispatching on the backend type and then calling the main constructor with the dimensions.
"""
function ThermalArrays(::Type{CPUBackend}, ni::Vararg{Integer, N}) where {N}
    return ThermalArrays(ni...)
end

"""
    ThermalArrays(nx::Integer, ny::Integer)

2D constructor for the thermal arrays for the heat diffusion solver with the extents given by `nx` and `ny`.

## Fields
- `T`: Temperature at cell centers with one ghost node on every boundary `(nx + 2, ny + 2)`
- `Tc`: View into the physical temperature cells of `T` `(nx, ny)`
- `Told`: Temperature at previous time step at cell centers with ghost nodes `(nx + 2, ny + 2)`
- `ΔT`: Temperature change at cell centers with ghost nodes `(nx + 2, ny + 2)`
- `ΔTc`: Temperature change at cell centers `(nx, ny)`
- `adiabatic`: Adiabatic term α (u ⋅ ∇P) at cell centers `(nx, ny)`
- `dT_dt`: Time derivative of temperature at cell centers `(nx, ny)`
- `qTx`: Conductive heat flux in x direction on cell faces `(nx + 1, ny)`
- `qTy`: Conductive heat flux in y direction on cell faces `(nx, ny + 1)`
- `qTx2`: Conductive heat flux in x direction on cell faces for second order scheme `(nx + 1, ny)`
- `qTy2`: Conductive heat flux in y direction on cell faces for second order scheme `(nx, ny + 1)`
- `H`: Source terms at cell centers `(nx, ny)`
- `shear_heating`: Shear heating terms at cell centers `(nx, ny)`
- `ResT`: Residual of the temperature equation at cell centers `(nx, ny)`
"""
function ThermalArrays(nx::Integer, ny::Integer)
    T = @zeros(nx + 2, ny + 2)
    ΔT = @zeros(nx + 2, ny + 2)
    ΔTc = @zeros(nx, ny)
    adiabatic = @zeros(nx, ny)
    Told = @zeros(nx + 2, ny + 2)
    Tc = @view T[2:(end - 1), 2:(end - 1)]
    H = @zeros(nx, ny)
    shear_heating = @zeros(nx, ny)
    dT_dt = @zeros(nx, ny)
    qTx = @zeros(nx + 1, ny)
    qTy = @zeros(nx, ny + 1)
    qTx2 = @zeros(nx + 1, ny)
    qTy2 = @zeros(nx, ny + 1)
    ResT = @zeros(nx, ny)
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

"""
    ThermalArrays(nx::Integer, ny::Integer, nz::Integer)

3D constructor for the thermal arrays for the heat diffusion solver with the extents given by `nx`, `ny` and `nz`.
## Fields
- `T`: Temperature at stag. location WITHOUT ghost nodes `(nx + 1, ny + 1, nz + 1)`
- `Tc`: Temperature at cell centers `(nx, ny, nz)`
- `Told`: Temperature at previous time step at stag. location WITHOUT ghost nodes `(nx + 1, ny + 1, nz + 1)`
- `ΔT`: Temperature change at stag. location WITHOUT ghost nodes `(nx + 1, ny + 1, nz + 1)`
- `ΔTc`: Temperature change at cell centers `(nx, ny, nz)`
- `adiabatic`: Adiabatic term α (u ⋅ ∇P) at their staggered location `(nx - 1, ny - 1, nz - 1)`
- `dT_dt`: Time derivative of temperature at their staggered location `(nx - 1, ny - 1, nz - 1)`
- `qTx`: Conductive heat flux in x direction at their staggered location `(nx, ny - 1, nz - 1)`
- `qTy`: Conductive heat flux in y direction at their staggered location `(nx - 1, ny, nz - 1)`
- `qTz`: Conductive heat flux in z direction at their staggered location `(nx - 1, ny - 1, nz)`
- `qTx2`: Conductive heat flux in x direction at their staggered location for second order scheme `(nx, ny - 1, nz - 1)`
- `qTy2`: Conductive heat flux in y direction at their staggered location for second order scheme `(nx - 1, ny, nz - 1)`
- `qTz2`: Conductive heat flux in z direction at their staggered location for second order scheme `(nx - 1, ny - 1, nz)`
- `H`: Source terms at cell centers `(nx, ny, nz)`
- `shear_heating`: Shear heating terms at cell centers `(nx, ny, nz)`
- `ResT`: Residual of the temperature equation at their staggered location `(nx - 1, ny - 1, nz - 1)`
"""
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
