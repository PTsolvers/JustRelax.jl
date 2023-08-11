using JustRelax

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

@parallel_indices (i, j) function init_T!(T, z)
    T[i, j] = if z[j] == maximum(z)
        300.0
    elseif z[j] == minimum(z)
        3500.0
    else
        z[j] * (1900.0 - 1600.0) / minimum(z) + 1600.0
    end
    return nothing
end


@parallel_indices (i, j) function compute_flux!(qTx, qTy, qTx2, qTy2, T, K, θr_dτ, _dx, _dy)

    nx = size(θr_dτ, 1)

    d_xa(A)  = _d_xa(A, i, j, _dx)
    d_ya(A)  = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j+1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5

    @inbounds if all((i, j) .≤ size(qTx))
        qx = qTx2[i, j] = -av_xa(K) * d_xa(T)
        qTx[i, j]  = (qTx[i, j] * av_xa(θr_dτ) + qx) / (1.0 + av_xa(θr_dτ))
    end
    
    @inbounds if all((i, j) .≤ size(qTy))
        qy = qTy2[i, j] = -av_ya(K) * d_ya(T) 
        qTy[i, j]  = (qTy[i, j] * av_ya(θr_dτ) + qy)/ (1.0 + av_ya(θr_dτ))
    end
    return nothing
end

@parallel_indices (i, j) function compute_update!(T, Told, qTx, qTy, ρCp, dτ_ρ, _dt, _dx, _dy)

    nx, = size(ρCp)

    d_xa(A)  = _d_xa(A, i, j, _dx)
    d_ya(A)  = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j+1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av(A)    = A |> av_xa |> av_ya
    
    T[i+1, j+1] += av(dτ_ρ) *
        (
            (-(d_xa(qTx) + d_ya(qTy))) -
            av(ρCp) * (T[i+1, j+1] - Told[i+1, j+1]) * _dt
        )
    return nothing
end

@parallel_indices (i, j) function check_res!(ResT, T, Told, qTx2, qTy2, ρCp, _dt, _dx, _dy)
    nx, = size(ρCp)
    
    d_xa(A)  = _d_xa(A, i, j, _dx)
    d_ya(A)  = _d_ya(A, i, j, _dy)
    av_xa(A) = (A[clamp(i - 1, 1, nx), j+1] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av_ya(A) = (A[clamp(i, 1, nx), j] + A[clamp(i - 1, 1, nx), j]) * 0.5
    av(A)    = A |> av_xa |> av_ya

    ResT[i, j] =
        -av(ρCp) * (T[i+1, j+1] - Told[i+1, j+1]) * _dt - (d_xa(qTx2) + d_ya(qTy2))
    return nothing
end


function diffusion_2D(; nx=32, ny=32, lx=100e3, ly=100e3, ρ0=3.3e3, Cp0=1.2e3, K0=3.0)
    kyr = 1e3 * 3600 * 24 * 365.25
    Myr = 1e3 * kyr
    ttot = 1 * Myr # total simulation time
    dt = 50 * kyr # physical time step

    # Physical domain
    ni = (nx, ny)
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / ni # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=(0, -ly)) # nodes at the center and vertices of the cells

    # Define the thermal parameters with GeoParams
    rheology = SetMaterialParams(;
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.1e3, β=0.0, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=Cp0),
        Conductivity      = ConstantConductivity(; k=K0),
    )
    # fields needed to compute density on the fly
    P = @zeros(ni...)
    args = (; P=P)

    ## Allocate arrays needed for every Thermal Diffusion
    thermal = ThermalArrays(ni)

    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux = (left = true, right = true, top = false, bot = false), 
    )
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2])
    @copy thermal.Told thermal.T
    T1 = deepcopy(thermal.T)

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    # Time loop
    while it < nt
        diffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di
        )

        it += 1
        t += dt
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), thermal, T1
end

# Model size 
lx = ly = L = 100e3 # [m]
# model resolution (number of gridpoints)
nx, ny = 64, 64
ρ0  = 3.3e3
Cp0 = 1.2e3
K0  = 3.0

# start model
geometry, thermal, T = diffusion_2D(;
    nx=nx, ny=ny, lx=L, ly=L, ρ0=3.3e3, Cp0=1.2e3, K0=3.0
);

Yv = [y for x in geometry.xvi[1], y in geometry.xvi[2]][:];

fig, ax, = lines(
    T[2:end-1, :][:],
    Yv,
    color=:black,
)

lines!(
    ax,
    thermal.T[2:end-1, :][:],
    Yv,
    color=:red,
)
fig

