import JustRelax.compute_ρCp

function subgrid_diffusion!(pT, pT0, pPhases, rheology, stokes, particles, di, dt)
    ni = size(pT)
    @parallel (@idx ni) subgrid_diffusion!(pT, pT0, pPhases, rheology, stokes.P, particles.index, di, dt)
end

@parallel_indices (I...) function subgrid_diffusion!(pT, pT0, pPhases, rheology, P, index, di, dt)

    P_cell = P[I...]

    for ip in JustRelax.cellaxes(pT)
        # early escape if there is no particle in this memory locaitons
        doskip(index, ip, I...) && continue

        pT0ᵢ = @cell pT0[ip, I...]
        pTᵢ = @cell pT[ip, I...]
        phase = Int(@cell(pPhases[ip, I...]))
        argsᵢ = (; T = pTᵢ, P = P_cell)
        # dimensionless numerical diffusion coefficient (0 ≤ d ≤ 1)
        d = 0.5
        # Compute the characteristic timescale `dt₀` of the local cell
        ρCp = compute_ρCp(rheology, phase, argsᵢ)
        K = compute_conductivity(rheology, phase, argsᵢ)
        sum_dxi = mapreduce(x-> inv(x)^2, +, di)
        dt₀ = ρCp / (2 * K * sum_dxi)
        # subgrid diffusion of the i-th particle
        @cell pT[ip, I...] = pT0ᵢ - (pT0ᵢ - pTᵢ) * exp(-d * dt / dt₀)
    end

    return nothing
end
