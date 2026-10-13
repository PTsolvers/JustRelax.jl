using Test, LinearAlgebra, SparseArrays

const gmg_backend_name = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")
@static if gmg_backend_name == "CUDA"
    import CUDA
    CUDA.allowscalar(false)
elseif gmg_backend_name == "AMDGPU"
    using AMDGPU
end

using JustRelax, JustRelax.JustRelax2D

# Test-only Stage 1 coarse-space screen: assembled 2D MAC operators, transfers
# assembled from kernel basis responses, exact coarse solves. Free-slip box.
const JR = JustRelax
const gmg_backend_type = gmg_backend_name == "CUDA" ? JR.CUDABackend :
    gmg_backend_name == "AMDGPU" ? JR.AMDGPUBackend : JR.CPUBackend
const K = gmg_backend_name == "CUDA" ? Base.get_extension(JR, :JustRelaxCUDAExt).JustRelax2D :
    gmg_backend_name == "AMDGPU" ? Base.get_extension(JR, :JustRelaxAMDGPUExt).JustRelax2D : JR.JustRelax2D

# Numbering of retained velocity DOFs: interior normal faces, all transverse cells.
function velocity_ids(level)
    n = level.grid.ni
    ids = [zeros(Int, size(level.correction.v[C])) for C in 1:2]
    k = 0
    for C in 1:2, I in CartesianIndices(ids[C])
        t = 3 - C
        if 2 <= I[C] <= n[C] && 2 <= I[t] <= n[t] + 1
            ids[C][I] = (k += 1)
        end
    end
    return ids, k
end

# Density-form A_γ = -Δ + γ Mv⁻¹ Dᵀ Mc D, divergence D, and both mass matrices.
function operators(level, γ)
    ids, nv = velocity_ids(level)
    n = level.grid.ni
    cw = map(Array, level.measures.cell)
    fw = map(Array, level.measures.face)
    rows, cols, vals = Int[], Int[], Float64[]
    add!(r, c, v) = c > 0 && (push!(rows, r); push!(cols, c); push!(vals, v))
    for C in 1:2, I in CartesianIndices(ids[C])
        r = ids[C][I]
        r == 0 && continue
        t = 3 - C
        i, j = I[C], I[t]
        id(a, b) = C == 1 ? ids[C][a, b] : ids[C][b, a]
        hl, hr, hd = cw[C][i - 1], cw[C][i], fw[C][i]
        add!(r, r, 1 / (hl * hd) + 1 / (hr * hd))
        add!(r, id(i - 1, j), -1 / (hl * hd))
        add!(r, id(i + 1, j), -1 / (hr * hd))
        h = cw[t][j - 1]
        for jj in (j - 1, j + 1)
            # Free slip: the mirrored ghost cancels the wall flux.
            2 <= jj <= n[t] + 1 || continue
            dist = (h + cw[t][jj - 1]) / 2
            add!(r, r, 1 / (dist * h))
            add!(r, id(i, jj), -1 / (dist * h))
        end
    end
    A = sparse(rows, cols, vals, nv, nv)
    cell = LinearIndices(n)
    rows, cols, vals = Int[], Int[], Float64[]
    for C in 1:2, I in CartesianIndices(ids[C])
        c = ids[C][I]
        c == 0 && continue
        i, j = I[C], I[3 - C] - 1
        below, above = C == 1 ? (cell[i - 1, j], cell[i, j]) : (cell[j, i - 1], cell[j, i])
        append!(rows, (below, above)); append!(cols, (c, c)); append!(vals, (1 / cw[C][i - 1], -1 / cw[C][i]))
    end
    D = sparse(rows, cols, vals, prod(n), nv)
    Mc = Diagonal(vec([cw[1][I[1]] * cw[2][I[2]] for I in CartesianIndices(n)]))
    Mv = Diagonal(zeros(nv))
    for C in 1:2, I in CartesianIndices(ids[C])
        ids[C][I] > 0 && (Mv.diag[ids[C][I]] = fw[C][I[C]] * cw[3 - C][I[3 - C] - 1])
    end
    return Matrix(A + γ * (Mv \ (D' * Mc * D))), Matrix(D), Mv, Mc
end

# Kernel basis responses, built on the host and copied whole to the backend.
function transfer_matrices(fine, coarse, transfer)
    fids, nfv = velocity_ids(fine)
    cids, ncv = velocity_ids(coarse)
    nfp, ncp = length(fine.correction.p), length(coarse.correction.p)
    function load!(fields, ids, k, nv)
        copyto!(fields.p, [Float64(k == nv + m) for m in 1:length(fields.p)])
        return foreach((v, id) -> copyto!(v, Float64.(id .== k)), fields.v, ids)
    end
    function flat_v(fields, ids, nv)
        x = zeros(nv)
        for C in 1:2
            A = Array(fields.v[C])
            for I in CartesianIndices(A)
                ids[C][I] > 0 && (x[ids[C][I]] = A[I])
            end
        end
        return [x; vec(Array(fields.p))]
    end
    P = zeros(nfv + nfp, ncv + ncp)
    for k in 1:(ncv + ncp)
        load!(coarse.correction, cids, k, ncv)
        K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 0)
        P[:, k] = flat_v(fine.correction, fids, nfv)
    end
    R = zeros(ncv + ncp, nfv + nfp)
    for k in 1:(nfv + nfp)
        load!(fine.residual, fids, k, nfv)
        K.gmg_restrict!(coarse.rhs, fine.residual, transfer)
        R[:, k] = flat_v(coarse.rhs, cids, ncv)
    end
    return P, R, nfv, ncv
end

function levels(n, velocity; aspect = 1.0, coarsening = (2, 2))
    x, y = collect(range(0.0, 1.0; length = n + 1)), collect(range(0.0, 1 / aspect; length = n + 1))
    options = JR.GMGOptions(Val(2); coarsening, coarsest_cells = 2, max_levels = 2)
    hierarchy = JR.GMGHierarchy(gmg_backend_type, (x, y), JR.GMGPartition((n, n), (1:n, 1:n)); options)
    fine, coarse = hierarchy.levels
    return fine, coarse, JR.GMGTransfer(fine, coarse, hierarchy.transitions[1]; velocity)
end

jacobi(A) = I - 4 / (3 * maximum(real, eigvals(Diagonal(1 ./ diag(A)) * A))) * Diagonal(1 ./ diag(A)) * A

# Velocity block: ρ of the 2+2 Jacobi two-grid and the divergence commutator.
function velocity_screen(n, velocity; γ = 0.0, kwargs...)
    fine, coarse, transfer = levels(n, velocity; kwargs...)
    Af, Df, _, _ = operators(fine, γ)
    Ac, Dc, _, _ = operators(coarse, γ)
    P, R, nfv, ncv = transfer_matrices(fine, coarse, transfer)
    Pv, Rv, Pp = P[1:nfv, 1:ncv], R[1:ncv, 1:nfv], P[(nfv + 1):end, (ncv + 1):end]
    S = jacobi(Af)
    ρ = maximum(abs, eigvals(S^2 * (I - Pv * (Ac \ (Rv * Af))) * S^2))
    commutator = opnorm(Df * Pv - Pp * Dc) / opnorm(Dc)
    return ρ, commutator
end

# Coupled [A_γ G; D 0] with a DR/Uzawa sweep (pressure first, then damped
# Jacobi velocity); the constant-pressure gauge is projected out.
function stokes_screen(n, velocity; γ = 0.0, θ = 0.5)
    fine, coarse, transfer = levels(n, velocity)
    coupled(level) = begin
        A, D, Mv, Mc = operators(level, γ)
        [A -(Mv \ (D' * Mc)); D zeros(size(D, 1), size(D, 1))], size(A, 1), Mc
    end
    Kf, nvf, Mcf = coupled(fine)
    Kc, nvc, Mcc = coupled(coarse)
    P, R, _, _ = transfer_matrices(fine, coarse, transfer)
    Avv = Kf[1:nvf, 1:nvf]
    Dinv = Diagonal(1 ./ diag(Avv))
    ω = 4 / (3 * maximum(real, eigvals(Dinv * Avv)))
    sweep(x) = begin
        v, p = x[1:nvf], x[(nvf + 1):end]
        p = p - θ * (1 + γ) * (Kf[(nvf + 1):end, 1:nvf] * v)
        v = v - ω * (Dinv * (Avv * v + Kf[1:nvf, (nvf + 1):end] * p))
        [v; p]
    end
    S = reduce(hcat, sweep.(eachcol(Matrix(1.0I, size(Kf)...))))
    gauge(Mc, nv) = I - [zeros(nv); ones(size(Mc, 1))] * [zeros(nv); diag(Mc)]' / sum(diag(Mc))
    Qf, Qc = gauge(Mcf, nvf), gauge(Mcc, nvc)
    E = Qf * S^2 * (I - P * Qc * pinv(Kc) * Qc' * R * Kf) * S^2 * Qf
    return maximum(abs, eigvals(E)), maximum(abs, eigvals(Qf * S^4 * Qf))
end

@testset "GMG two-grid coarse-space screen" begin
    conservative, smooth = JR.GMGConservativeVelocity(), JR.GMGSmoothVelocity()

    @testset "Velocity block" begin
        for velocity in (conservative, smooth)
            ρ8, c8 = velocity_screen(8, velocity)
            ρ16, c16 = velocity_screen(16, velocity)
            @test ρ8 < 0.25 && ρ16 < 0.25
            @test abs(ρ16 - ρ8) < 0.03
            if velocity === conservative
                @test c8 < 1.0e-12 && c16 < 1.0e-12
            else
                # The prototype rule does not commute with divergence.
                @test c8 > 0.5 && c16 > 0.5
            end
        end
    end

    @testset "Cell aspect needs semicoarsening" begin
        ρ_full, _ = velocity_screen(16, conservative; aspect = 4.0)
        ρ_semi, _ = velocity_screen(16, conservative; aspect = 4.0, coarsening = (1, 2))
        @test ρ_full > 0.8
        @test ρ_semi < 0.1
    end

    @testset "Coupled Stokes" begin
        for velocity in (conservative, smooth)
            ρ8, smoother8 = stokes_screen(8, velocity)
            ρ16, smoother16 = stokes_screen(16, velocity)
            @test ρ8 < 0.45 && ρ16 < 0.45
            @test abs(ρ16 - ρ8) < 0.03
            @test smoother16 > 0.98
        end
        # Strong augmentation is smoother-limited for both rules, not transfer-limited.
        ρc, _ = stokes_screen(8, conservative; γ = 10.0)
        ρs, _ = stokes_screen(8, smooth; γ = 10.0)
        @test ρc > 0.8 && abs(ρc - ρs) < 0.02
    end
end
