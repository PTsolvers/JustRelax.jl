using Test, LinearAlgebra, SparseArrays, Random

const gmg_backend_name = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")
@static if gmg_backend_name == "CUDA"
    import CUDA
    CUDA.allowscalar(false)
elseif gmg_backend_name == "AMDGPU"
    using AMDGPU
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.JustRelax3D

const JR = JustRelax
const gmg_backend_type = gmg_backend_name == "CUDA" ? JR.CUDABackend :
    gmg_backend_name == "AMDGPU" ? JR.AMDGPUBackend : JR.CPUBackend

function gmg_kernel_module(N)
    extension = gmg_backend_name == "CUDA" ? :JustRelaxCUDAExt :
        gmg_backend_name == "AMDGPU" ? :JustRelaxAMDGPUExt : nothing
    parent = isnothing(extension) ? JR : Base.get_extension(JR, extension)
    return N == 2 ? parent.JustRelax2D : parent.JustRelax3D
end

stretched(n, a, b, s) = (ξ = range(0.0, 1.0; length = n + 1); collect(a .+ (b - a) .* (ξ .+ s .* ξ .* (1 .- ξ))))

function hierarchy(ni; s = 0.3, max_levels = 1, coarsening = ntuple(_ -> 2, length(ni)))
    N = length(ni)
    xvi = ntuple(d -> stretched(ni[d], 0.0, 1.0 + 0.5d, d == 1 ? s : -s / 2), N)
    options = JR.GMGOptions(Val(N); coarsening, coarsest_cells = 2, max_levels)
    return JR.GMGHierarchy(gmg_backend_type, xvi, JR.GMGPartition(ni, ntuple(d -> 1:ni[d], N)); options)
end

# DOF masks in the screen ordering: velocity components (column-major), then pressure.
dof_masks(op) = (map(m -> Array(m) .> 0, op.velocity_mask)..., Array(op.pressure_mask) .> 0)

host(f) = (map(Array, f.v)..., Array(f.p))

function load!(f, masks, x)
    k = 0
    arrays = map(m -> zeros(size(m)), masks)
    for (A, m) in zip(arrays, masks), I in findall(m)
        A[I] = x[k += 1]
    end
    foreach((dst, src) -> copyto!(dst, src), f.v, arrays[1:(end - 1)])
    copyto!(f.p, arrays[end])
    return f
end

gather(arrays, masks) = reduce(vcat, [A[m] for (A, m) in zip(arrays, masks)])

function apply(K, op, e, masks, x)
    load!(e, masks, x)
    JR.gmg_prepare!(op, e)
    K.gmg_apply_operator!(op, e)
    return gather(host(op.out), masks)
end

function assemble(K, op, e, masks)
    n = sum(count, masks)
    return reduce(hcat, [apply(K, op, e, masks, Float64.(1:n .== k)) for k in 1:n])
end

# Independent 2D reference for constant η = 1 (free slip): -Δ, divergence, masses.
function reference(level)
    n = level.grid.ni
    cw, fw = map(Array, level.measures.cell), map(Array, level.measures.face)
    ids = [zeros(Int, size(level.correction.v[C])) for C in 1:2]
    k = 0
    for C in 1:2, I in CartesianIndices(ids[C])
        t = 3 - C
        2 <= I[C] <= n[C] && 2 <= I[t] <= n[t] + 1 && (ids[C][I] = (k += 1))
    end
    A = zeros(k, k)
    D = zeros(prod(n), k)
    Mv = zeros(k)
    cell = LinearIndices(n)
    for C in 1:2, I in CartesianIndices(ids[C])
        r = ids[C][I]
        r == 0 && continue
        t = 3 - C
        i, j = I[C], I[t]
        id(a, b) = C == 1 ? ids[C][a, b] : ids[C][b, a]
        hl, hr, hd = cw[C][i - 1], cw[C][i], fw[C][i]
        A[r, r] += 1 / (hl * hd) + 1 / (hr * hd)
        id(i - 1, j) > 0 && (A[r, id(i - 1, j)] -= 1 / (hl * hd))
        id(i + 1, j) > 0 && (A[r, id(i + 1, j)] -= 1 / (hr * hd))
        h = cw[t][j - 1]
        for jj in (j - 1, j + 1)
            2 <= jj <= n[t] + 1 || continue
            dist = (h + cw[t][jj - 1]) / 2
            A[r, r] += 1 / (dist * h)
            A[r, id(i, jj)] -= 1 / (dist * h)
        end
        below, above = C == 1 ? (cell[i - 1, j - 1], cell[i, j - 1]) : (cell[j - 1, i - 1], cell[j - 1, i])
        D[below, r] += 1 / cw[C][i - 1]
        D[above, r] -= 1 / cw[C][i]
        Mv[r] = hd * h
    end
    Mc = vec([cw[1][I[1]] * cw[2][I[2]] for I in CartesianIndices(n)])
    return A, D, Diagonal(Mv), Diagonal(Mc)
end

function viscosity(level, contrast)
    xc = map(Array, level.grid.xci)
    N = length(xc)
    η = [exp(log(contrast) * prod(sin(2.3 * d * xc[d][I[d]] + 0.4d) for d in 1:N)^2) for I in CartesianIndices(level.grid.ni)]
    return copyto!(similar(level.correction.p), η)
end

@testset "GMG matrix-free operator" begin
    rng = Xoshiro(7)

    @testset "2D constant viscosity matches assembled reference" begin
        K = gmg_kernel_module(2)
        level = only(hierarchy((6, 8)).levels)
        for (γ, c) in ((0.0, 0.0), (2.0, 0.0), (0.5, 0.3))
            op = JR.GMGLevelOperator(level)
            K.gmg_set_coefficients!(op, 1.0; compliance = c, γ)
            masks = dof_masks(op)
            Kmat = assemble(K, op, level.correction, masks)
            A, D, Mv, Mc = reference(level)
            G = -(Mv \ (D' * Mc))
            nv = size(A, 1)
            expected = [A + (1 / 3 + γ) * (Mv \ (D' * Mc * D)) (1 - γ * c) * G; D c * I]
            @test size(Kmat) == size(expected)
            @test Kmat ≈ expected atol = 1.0e-10 * maximum(abs, expected)
        end
    end

    for N in (2, 3)
        K = gmg_kernel_module(N)
        ni = N == 2 ? (6, 8) : (4, 4, 6)
        level = only(hierarchy(ni).levels)
        @testset "$(N)D variable viscosity, no slip = $no_slip" for no_slip in (false, true)
            bcs = ntuple(d -> (no_slip, false), N)
            op = JR.GMGLevelOperator(level; no_slip = bcs)
            K.gmg_set_coefficients!(op, viscosity(level, 100.0); γ = 1.0)
            masks = dof_masks(op)
            n = sum(count, masks)
            x, y = randn(rng, n), randn(rng, n)
            @test all(iszero, apply(K, op, level.correction, masks, zeros(n)))
            @test apply(K, op, level.correction, masks, 2x .- 3y) ≈ 2 .* apply(K, op, level.correction, masks, x) .- 3 .* apply(K, op, level.correction, masks, y)
            Kmat = assemble(K, op, level.correction, masks)
            nv = n - count(masks[end])
            # Velocity block is symmetric positive definite in the dual-volume inner product.
            cw, fw = map(Array, level.measures.cell), map(Array, level.measures.face)
            Mv = Diagonal(reduce(vcat, [[prod(ntuple(d -> d == C ? fw[d][I[d]] : cw[d][I[d] - 1], N)) for I in findall(masks[C])] for C in 1:N]))
            A = Kmat[1:nv, 1:nv]
            @test Mv * A ≈ (Mv * A)' rtol = 1.0e-10
            @test minimum(eigvals(Symmetric(Mv * A))) > 0
            # Probing recovers the exact diagonal and a valid Gershgorin bound.
            λ = K.gmg_refresh_bounds!(op, level.partition)
            inv_diagonal = gather((map(Array, op.inv_diagonal)..., zeros(size(masks[end]))), masks)[1:nv]
            @test inv_diagonal ≈ 1 ./ diag(A) rtol = 1.0e-10
            gershgorin = maximum(sum(abs, A; dims = 2) ./ diag(A))
            @test λ ≈ gershgorin rtol = 1.0e-10
            @test λ >= maximum(real, eigvals(Diagonal(1 ./ diag(A)) * A)) * (1 - 1.0e-10)
            @test @inferred(K.gmg_apply_operator!(op, level.correction)) === nothing

            # One sweep equals the explicit recurrence with assembled matrices.
            smoother = JR.GMGSmoother(; substeps = 3, spectral_range = 8, pressure_step = 0.7)
            e0, r = randn(rng, n), randn(rng, n)
            load!(level.rhs, masks, r)
            load!(level.correction, masks, e0)
            K.gmg_dr_smooth!(op, level.correction, level.rhs, smoother)
            got = gather(host(level.correction), masks)
            Dinv = Diagonal(1 ./ diag(A))
            a, b = λ / 8, λ
            θ, δ = (a + b) / 2, (b - a) / 2
            σ = θ / δ
            v, p = e0[1:nv], e0[(nv + 1):end]
            Kvp = Kmat[1:nv, (nv + 1):end]
            ρ = 1 / σ
            d = Dinv * (r[1:nv] - A * v - Kvp * p) / θ
            v += d
            for _ in 2:3
                ρn = 1 / (2σ - ρ)
                d = ρn * ρ * d + 2ρn / δ * Dinv * (r[1:nv] - A * v - Kvp * p)
                v += d
                ρ = ρn
            end
            p += 0.7 * (r[(nv + 1):end] - Kmat[(nv + 1):end, :] * [v; p])
            @test got ≈ [v; p] rtol = 1.0e-10
        end
    end

    @testset "Coefficient restriction" begin
        for N in (2, 3)
            K = gmg_kernel_module(N)
            h = hierarchy(N == 2 ? (8, 8) : (4, 4, 4); max_levels = 2)
            fine, coarse = h.levels
            transfer = JR.GMGTransfer(fine, coarse, h.transitions[1])
            fop, cop = JR.GMGLevelOperator(fine), JR.GMGLevelOperator(coarse)
            K.gmg_set_coefficients!(fop, 3.0; compliance = 0.25, γ = 2.0)
            K.gmg_restrict_coefficients!(cop, fop, transfer)
            @test all(≈(3.0), Array(cop.coefficients.η))
            @test all(≈(3.0), Array(cop.coefficients.η_edge[1]))
            @test all(≈(0.25), Array(cop.coefficients.compliance))
            @test all(≈(2.0), Array(cop.coefficients.γ))
            # Two-valued field: harmonic volume average per parent.
            ηf = viscosity(fine, 1.0e3)
            K.gmg_set_coefficients!(fop, ηf)
            K.gmg_restrict_coefficients!(cop, fop, transfer)
            vol(level) = [prod(ntuple(d -> Array(level.measures.cell[d])[I[d]], N)) for I in CartesianIndices(level.grid.ni)]
            vf, vc = vol(fine), vol(coarse)
            ηh = Array(ηf)
            expected = [vc[J] / sum(vf[I] / ηh[I] for I in CartesianIndices(ηh) if all(Tuple(I) .÷ 2 .+ Tuple(I) .% 2 .== Tuple(J))) for J in CartesianIndices(vc)]
            @test Array(cop.coefficients.η) ≈ expected rtol = 1.0e-12
        end
    end

    @testset "Matrix-free two-grid with exact coarse solve" begin
        K = gmg_kernel_module(2)
        smoother = JR.GMGSmoother(; substeps = 5, spectral_range = 8, pressure_step = 1.0)
        factors = map((8, 16)) do n
            h = hierarchy((n, n); s = 0.0, max_levels = 2)
            fine, coarse = h.levels
            transfer = JR.GMGTransfer(fine, coarse, h.transitions[1])
            fop, cop = JR.GMGLevelOperator(fine), JR.GMGLevelOperator(coarse)
            K.gmg_set_coefficients!(fop, 1.0; γ = 1.0)
            K.gmg_restrict_coefficients!(cop, fop, transfer)
            K.gmg_refresh_bounds!(fop, fine.partition)
            fmasks, cmasks = dof_masks(fop), dof_masks(cop)
            Kc = assemble(K, cop, coarse.correction, cmasks)
            nf = sum(count, fmasks)
            pf = count(fmasks[end])
            gauge = I - [zeros(nf - pf); ones(pf)] * [zeros(nf - pf); fill(1 / pf, pf)]'
            function cycle(x)
                load!(fine.correction, fmasks, x)
                foreach(f -> fill!(f, 0), (fine.rhs.p, fine.rhs.v...))
                K.gmg_dr_smooth!(fop, fine.correction, fine.rhs, smoother)
                K.gmg_apply_operator!(fop, fine.correction)
                foreach((r, o) -> r .= .-o, (fine.residual.v..., fine.residual.p), (fop.out.v..., fop.out.p))
                K.gmg_restrict!(coarse.rhs, fine.residual, transfer)
                ec = pinv(Kc) * gather(host(coarse.rhs), cmasks)
                load!(coarse.correction, cmasks, ec)
                K.gmg_prolong!(fine.correction, coarse.correction, transfer)
                K.gmg_dr_smooth!(fop, fine.correction, fine.rhs, smoother)
                return gather(host(fine.correction), fmasks)
            end
            E = gauge * reduce(hcat, [cycle(gauge[:, k]) for k in 1:nf]) * gauge
            maximum(abs, eigvals(E))
        end
        @test all(<(0.5), factors)
        @test abs(factors[2] - factors[1]) < 0.05
    end
end
