using Test, LinearAlgebra, Random

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

# Host masks of owned, retained DOFs and their geometric measures.
function host_dofs(level::JR.GMGLevel{G, D, P}) where {G, D, P}
    part = level.partition
    N = length(part.global_ni)
    cell = map(Array, level.measures.cell)
    face = map(Array, level.measures.face)
    tensor(ws) = [prod(ntuple(d -> ws[d][I[d]], N)) for I in CartesianIndices(map(length, ws))]
    pmask = falses(size(level.correction.p))
    pmask[part.owned_local...] .= true
    p = (; mask = pmask, measure = tensor(cell))
    v = ntuple(N) do C
        owned = JR.gmg_owned_velocity(part, Val(C))
        mask = falses(size(level.correction.v[C]))
        mask[owned...] .= true
        if !part.periodic[C]
            # Nonperiodic normal boundary faces carry a prescribed zero correction.
            n = size(mask, C)
            first(part.owned[C]) == 1 && (selectdim(mask, C, 1) .= false)
            last(part.owned[C]) == part.global_ni[C] && (selectdim(mask, C, n) .= false)
        end
        widths = ntuple(d -> d == C ? face[d] : [zero(eltype(face[d])); cell[d]; zero(eltype(face[d]))], N)
        (; mask, measure = tensor(widths))
    end
    return (p, v...)
end

host_fields(f::JR.GMGCoupledFields) = (Array(f.p), map(Array, f.v)...)

function set_fields!(f::JR.GMGCoupledFields, values)
    copyto!(f.p, values[1])
    foreach((dst, src) -> copyto!(dst, src), f.v, values[2:end])
    return f
end

dot_m(a, b, dofs) = sum(sum(x .* y .* d.measure .* d.mask) for (x, y, d) in zip(a, b, dofs))

function stretched(T, n, a, b, s)
    ξ = range(zero(T), one(T); length = n + 1)
    return collect(a .+ (b - a) .* (ξ .+ T(s) .* ξ .* (one(T) .- ξ)))
end

function build(T, N, ni, coarsening; periodic = ntuple(_ -> false, N), velocity, no_slip = ntuple(_ -> (false, false), N), s = 0.3)
    xvi = ntuple(d -> stretched(T, ni[d], T(-d), T(d), d == 1 ? s : -s / 2), N)
    partition = JR.GMGPartition(ni, ntuple(d -> 1:ni[d], N); periodic)
    options = JR.GMGOptions(Val(N); coarsening, coarsest_cells = 2, max_levels = 2)
    hierarchy = JR.GMGHierarchy(gmg_backend_type, xvi, partition; options)
    fine, coarse = hierarchy.levels
    transfer = JR.GMGTransfer(fine, coarse, hierarchy.transitions[1]; velocity, no_slip)
    return fine, coarse, transfer
end

randomize(level, rng) = map(x -> rand(rng, eltype(x), size(x)) .- eltype(x)(0.5), host_fields(level.correction))

@testset "GMG transfers" begin
    rng = Xoshiro(1234)
    for N in (2, 3), T in (Float64, Float32)
        K = gmg_kernel_module(N)
        tol = T == Float64 ? 1.0e-12 : 1.0e-5
        ni = N == 2 ? (8, 12) : (4, 6, 4)
        ratios = N == 2 ? ((2, 2), (1, 2), (2, 1)) : ((2, 2, 2), (2, 1, 2), (1, 2, 1))
        rules = (JR.GMGConservativeVelocity(), JR.GMGSmoothVelocity())
        for coarsening in ratios, velocity in rules, periodic in (ntuple(_ -> false, N), ntuple(d -> d == 1, N))
            no_slip = ntuple(d -> (d == N, false), N)
            fine, coarse, transfer = build(T, N, ni, coarsening; periodic, velocity, no_slip)
            @test fine.domain.ni == ni
            @test coarse.domain.ni == ntuple(d -> ni[d] ÷ coarsening[d], N)
            fdofs, cdofs = host_dofs(fine), host_dofs(coarse)

            # Weighted adjoint identity on owned retained DOFs.
            c, f = randomize(coarse, rng), randomize(fine, rng)
            set_fields!(coarse.correction, c)
            set_fields!(fine.residual, f)
            fill!(fine.correction.p, T(NaN))
            foreach(v -> fill!(v, zero(T)), fine.correction.v)
            K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 0)
            K.gmg_restrict!(coarse.rhs, fine.residual, transfer)
            Pc, Rf = host_fields(fine.correction), host_fields(coarse.rhs)
            @test all(isfinite, Pc[1])
            lhs, rhs = dot_m(Pc, f, fdofs), dot_m(c, Rf, cdofs)
            @test abs(lhs - rhs) <= tol * (abs(lhs) + abs(rhs) + 1)
            # Restriction writes nothing outside owned retained coarse DOFs.
            @test all(iszero(x[.!d.mask]) for (x, d) in zip(Rf, cdofs))

            # Constant pressure gauge: prolongated and restricted exactly.
            fill!(coarse.correction.p, one(T))
            fill!(fine.residual.p, one(T))
            K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 0)
            K.gmg_restrict!(coarse.rhs, fine.residual, transfer)
            @test all(x -> abs(x - 1) <= tol, Array(fine.correction.p)[fdofs[1].mask])
            @test all(x -> abs(x - 1) <= tol, Array(coarse.rhs.p)[cdofs[1].mask])

            # Integrated continuity residual is conserved.
            fp = rand(rng, T, size(fine.residual.p))
            copyto!(fine.residual.p, fp)
            K.gmg_restrict!(coarse.rhs, fine.residual, transfer)
            @test dot_m((fp,), (one.(fp),), fdofs[1:1]) ≈ dot_m((Array(coarse.rhs.p),), (one.(Array(coarse.rhs.p)),), cdofs[1:1]) rtol = tol * 10

            # Normal-direction affine reproduction away from eliminated walls;
            # an affine field is not periodic, and no slip damps the first cell.
            for C in 1:N
                periodic[C] && continue
                xf, xc = fine.domain.xvi[C], coarse.domain.xvi[C]
                cv = [xc[I[C]] for I in CartesianIndices(size(coarse.correction.v[C]))] .* cdofs[C + 1].mask
                copyto!(coarse.correction.v[C], cv)
                K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 0)
                fv = Array(fine.correction.v[C])
                # Fine faces whose coarse support excludes an eliminated wall face.
                wall(I) = any(d -> d != C && no_slip[d][1] && I[d] == 2, 1:N)
                interior = [fdofs[C + 1].mask[I] && xc[2] <= xf[I[C]] <= xc[end - 1] && !wall(I) for I in CartesianIndices(fv)]
                @test all(abs(fv[I] - xf[I[C]]) <= tol * 10 for I in findall(interior))
            end

            # Conservative transfer commutes with divergence (unmasked, nonperiodic).
            if velocity isa JR.GMGConservativeVelocity && !any(periodic)
                v = map(d -> d.mask .* (rand(rng, T, size(d.mask)) .- T(0.5)), cdofs[2:end])
                foreach((dst, src) -> copyto!(dst, src), coarse.correction.v, v)
                K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 0)
                div(level, vs) = begin
                    cw = map(Array, level.measures.cell)
                    out = zeros(T, level.grid.ni)
                    for I in CartesianIndices(out), d in 1:N
                        hi = ntuple(k -> I[k] + 1, N)
                        lo = ntuple(k -> I[k] + (k == d ? 0 : 1), N)
                        out[I] += (vs[d][hi...] - vs[d][lo...]) / cw[d][I[d]]
                    end
                    out
                end
                divc = div(coarse, v)
                divf = div(fine, map(Array, fine.correction.v))
                parent = [divc[ntuple(d -> fld(I[d] - 1, coarsening[d]) + 1, N)...] for I in CartesianIndices(divf)]
                @test maximum(abs, divf .- parent) <= tol * 100 * maximum(abs, divc)
            end
        end
    end

    @testset "Inference" begin
        for N in (2, 3), velocity in (JR.GMGConservativeVelocity(), JR.GMGSmoothVelocity())
            K = gmg_kernel_module(N)
            fine, coarse, transfer = build(Float64, N, ntuple(_ -> 4, N), ntuple(_ -> 2, N); velocity)
            hierarchy_transition = JR.GMGTransition(ntuple(_ -> 2, N))
            @test @inferred(JR.GMGTransfer(fine, coarse, hierarchy_transition; velocity)) isa typeof(transfer)
            @test isconcretetype(typeof(transfer))
            @test @inferred(K.gmg_prolong!(fine.correction, coarse.correction, transfer)) === nothing
            @test @inferred(K.gmg_restrict!(coarse.rhs, fine.residual, transfer)) === nothing
        end
    end

    @testset "Assembled 2D oracle" begin
        K = gmg_kernel_module(2)
        for velocity in (JR.GMGConservativeVelocity(), JR.GMGSmoothVelocity())
            fine, coarse, transfer = build(Float64, 2, (4, 4), (2, 2); velocity, no_slip = ((false, false), (true, false)))
            fdofs, cdofs = host_dofs(fine), host_dofs(coarse)
            flat(x, dofs) = reduce(vcat, [a[d.mask] for (a, d) in zip(x, dofs)])
            function basis(dofs, k)
                fields = map(d -> zeros(size(d.mask)), dofs)
                count = 0
                for (a, d) in zip(fields, dofs), I in findall(d.mask)
                    count += 1
                    count == k && (a[I] = 1.0)
                end
                return fields
            end
            nf, nc = sum(d -> count(d.mask), fdofs), sum(d -> count(d.mask), cdofs)
            # Retained counts: pressure cells plus interior normal faces.
            @test nc == 4 + 2 * (1 * 2)
            @test nf == 16 + 2 * (3 * 4)
            P = zeros(nf, nc)
            for k in 1:nc
                set_fields!(coarse.correction, basis(cdofs, k))
                K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 0)
                P[:, k] = flat(host_fields(fine.correction), fdofs)
            end
            R = zeros(nc, nf)
            for k in 1:nf
                set_fields!(fine.residual, basis(fdofs, k))
                K.gmg_restrict!(coarse.rhs, fine.residual, transfer)
                R[:, k] = flat(host_fields(coarse.rhs), cdofs)
            end
            Mf = Diagonal(flat(map(d -> d.measure, fdofs), fdofs))
            Mc = Diagonal(flat(map(d -> d.measure, cdofs), cdofs))
            @test R ≈ Mc \ (P' * Mf) atol = 1.0e-13
            @test all(sum(P[1:16, 1:4]; dims = 2) .≈ 1)
            # β accumulates instead of overwriting.
            set_fields!(coarse.correction, basis(cdofs, 1))
            K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 0)
            K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 1)
            @test flat(host_fields(fine.correction), fdofs) ≈ 2 .* P[:, 1]
        end
    end
end
