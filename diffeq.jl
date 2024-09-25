using LinearAlgebra
using DiffEqBase
using OrdinaryDiffEq: SplitODEProblem, solve, IMEXEuler
import SciMLBase

# Initial thermal profile
function init_T(z)
    depth = 1000e3 - z

    T = if depth < 0e0
        273e0
    elseif 0e0 ≤ (depth) < 120e3
        dTdZ        = (1273-273)/120e3
        offset      = 273e0
        (depth) * dTdZ + offset
    elseif (depth) ≥ 120e3
        offset      = 273e0
        1000.0 + offset
    end

    return T
end

function diffeq_constant_density(ny, Δt, N_t)
    # kyr  = 1e3 * 3600 * 24 * 365.25
    # Myr  = 1e3 * kyr
    # ttot = 1 * Myr # total simulation time
    # N_t  = 150;                        # number of timesteps to take
    # Δt   = 100e3 * kyr;                       # timestep size
    
    a, b, n = 0, 1000e3, ny         # zmin, zmax, number of cells
    n̂_min, n̂_max = -1, 1             # Outward facing unit vectors
    K = 5.0
    Cp = 1250.0
    ρ = 4000.0
    α = K / Cp / ρ;                  # thermal diffusivity, larger means more stiff
    # β, γ = 10000, π;                 # source term coefficients
    FT = Float64;                    # float type
    Δz = FT(b - a) / FT(n)
    Δz² = Δz^2;
    ∇²_op = [1 / Δz², -2 / Δz², 1 / Δz²]; # interior Laplacian operator
    ∇T_bottom = 0;                       # Temperature gradient at the top
    T_top = 273;                            # Temperature at the bottom
    # S(z) = β * sin(γ * z)                 # source term, (sin for easy integration)
    S(z) = 0                              # source term, (sin for easy integration)
    zf = range(a, b, length = n + 1);     # coordinates on cell faces

    # discrete diffusion operator
    # Initialize interior and boundary stencils:
    ∇² = Tridiagonal(ones(FT, n) .* ∇²_op[1],
        ones(FT, n + 1) .* ∇²_op[2],
        ones(FT, n) .* ∇²_op[3]);

    # Modify boundary stencil to account for BCs
    ∇².d[1] = -2 / Δz²
    ∇².du[1] = +2 / Δz²

    # Modify boundary stencil to account for BCs
    ∇².du[n] = 0  # modified stencil
    ∇².d[n + 1] = 0 # to ensure `∂_t T = 0` at `z=zmax`
    ∇².dl[n] = 0  # to ensure `∂_t T = 0` at `z=zmax`
    D = α .* ∇²

    # Boundary source
    AT_b = zeros(FT, n + 1);
    AT_b[1] = α * 2 / Δz * ∇T_bottom * n̂_min;
    AT_b[end - 1] = α * T_top / Δz²;

    T = init_T.(zf);
    T[n + 1] = T_top; # set top BC
    lines(T, collect(zf))

    function rhs!(dT, T, params, t)
        n = params.n
        i = 1:n # interior domain
        dT[i] .= S.(zf[i]) .+ AT_b[i]
        return dT
    end;

    params = (; n)

    tspan = (FT(0), N_t * FT(Δt))

    prob = SplitODEProblem(
        SciMLBase.DiffEqArrayOperator(D),
        rhs!,
        T,
        tspan,
        params
    )

    alg = IMEXEuler()
    println("Solving...")
    sol = solve(
        prob,
        alg,
        dt = Δt,
        saveat = range(FT(0), N_t * FT(Δt), length = 5),
        progress = true,
        progress_message = (dt, u, p, t) -> t
    );
    return sol
end

density(ρ0, α, T) = ρ0 * (1 - α * (T - 273))


function diffeq_Tdep_density(ny, Δt, N_t)
    # N_t = 150;                        # number of timesteps to take
    # Δt = 1000e3 * 365 * 3600 * 24;                       # timestep size
    
    a, b, n = 0, 1000e3, ny         # zmin, zmax, number of cells
    n̂_min, n̂_max = -1, 1             # Outward facing unit vectors
     # β, γ = 10000, π;                 # source term coefficients
    FT = Float64;                    # float type
    Δz = FT(b - a) / FT(n)
    Δz² = Δz^2;
    ∇²_op = [1 / Δz², -2 / Δz², 1 / Δz²]; # interior Laplacian operator
    ∇T_bottom = 0;                       # Temperature gradient at the top
    T_top = 273;                            # Temperature at the bottom
    # S(z) = β * sin(γ * z)                 # source term, (sin for easy integration)
    S(z) = 0                              # source term, (sin for easy integration)
    zf = range(a, b, length = n + 1);     # coordinates on cell faces

    T = init_T.(zf);
    T[n + 1] = T_top; # set top BC
    
    K = 5.0
    Cp = 1250.0
    ρ0 = 4e3
    ρ = density.(ρ0, 2.5e-5, T)
    α = @. K / Cp / ρ;                  # thermal diffusivity, larger means more stiff
  

    # discrete diffusion operator
    # Initialize interior and boundary stencils:
    ∇² = Tridiagonal(ones(FT, n) .* ∇²_op[1],
        ones(FT, n + 1) .* ∇²_op[2],
        ones(FT, n) .* ∇²_op[3]);

    # Modify boundary stencil to account for BCs
    ∇².d[1] = -2 / Δz²
    ∇².du[1] = +2 / Δz²

    # Modify boundary stencil to account for BCs
    ∇².du[n] = 0  # modified stencil
    ∇².d[n + 1] = 0 # to ensure `∂_t T = 0` at `z=zmax`
    ∇².dl[n] = 0  # to ensure `∂_t T = 0` at `z=zmax`
    D = α .* ∇²

    # Boundary source
    AT_b = zeros(FT, n + 1);
    AT_b[1] = α[1] * 2 / Δz * ∇T_bottom * n̂_min;
    AT_b[end - 1] = α[end - 1] * T_top / Δz²;

    function rhs!(dT, T, params, t)
        n = params.n
        i = 1:n # interior domain
        dT[i] .= S.(zf[i]) .+ AT_b[i]
        return dT
    end;

    params = (; n)

    tspan = (FT(0), N_t * FT(Δt))

    prob = SplitODEProblem(
        SciMLBase.DiffEqArrayOperator(D),
        rhs!,
        T,
        tspan,
        params
    )

    alg = IMEXEuler()
    println("Solving...")
    sol = solve(
        prob,
        alg,
        dt = Δt,
        saveat = range(FT(0), N_t * FT(Δt), length = 5),
        progress = true,
        progress_message = (dt, u, p, t) -> t
    );
    return sol
end
