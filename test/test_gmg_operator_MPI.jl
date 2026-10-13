using Test, Random

const gmg_mpi_backend_name = get(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")
@static if gmg_mpi_backend_name == "CUDA"
    import CUDA
    CUDA.allowscalar(false)
elseif gmg_mpi_backend_name == "AMDGPU"
    using AMDGPU
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.JustRelax3D

const JR = JustRelax
const MPI = JR.MPI
const gmg_backend_type = gmg_mpi_backend_name == "CUDA" ? JR.CUDABackend :
    gmg_mpi_backend_name == "AMDGPU" ? JR.AMDGPUBackend : JR.CPUBackend

function gmg_kernel_module(N)
    extension = gmg_mpi_backend_name == "CUDA" ? :JustRelaxCUDAExt :
        gmg_mpi_backend_name == "AMDGPU" ? :JustRelaxAMDGPUExt : nothing
    parent = isnothing(extension) ? JR : Base.get_extension(JR, extension)
    return N == 2 ? parent.JustRelax2D : parent.JustRelax3D
end

# Per-direction storage rule and ghost offset of each coupled field.
locations(N) = (
    ntuple(_ -> (JR.GMGInjection(), false), N),
    ntuple(C -> ntuple(d -> d == C ? (JR.GMGLinearFaces(), false) : (JR.GMGInjection(), true), N), N)...,
)

# Wrapped global index of a storage position, or nothing for velocity ghosts.
function position_global(part, location, I)
    N = length(I)
    g = ntuple(N) do d
        rule, ghosted = location[d]
        valid = JR.gmg_valid_positions(rule, part, d, ghosted)
        I[d] in valid || return 0
        return JR.gmg_wrap(JR.gmg_storage_global(part, d, I[d], ghosted), part.global_ni[d], part.periodic[d])
    end
    return any(iszero, g) ? nothing : g
end

is_owned(part, location, I) = all(d -> I[d] in JR.gmg_owned_positions(location[d][1], part, d, location[d][2]), 1:length(I))

host_fields(f) = (Array(f.p), map(Array, f.v)...)

function set_fields!(f, values)
    copyto!(f.p, values[1])
    foreach((dst, src) -> copyto!(dst, src), f.v, values[2:end])
    return f
end

# Host field whose owned entries come from a global array; halos are NaN.
function from_global(level, globals, T; owned_only = true)
    part = level.partition
    shapes = (size(level.correction.p), map(size, level.correction.v)...)
    return map(shapes, locations(length(part.global_ni)), globals) do shape, location, G
        A = fill(T(NaN), shape)
        for I in CartesianIndices(A)
            g = position_global(part, location, Tuple(I))
            isnothing(g) && continue
            (!owned_only || is_owned(part, location, Tuple(I))) && (A[I] = G[g...])
        end
        A
    end
end

block(ni, dims, coords) = ntuple(d -> ((ni[d] ÷ dims[d]) * coords[d] + 1):((ni[d] ÷ dims[d]) * (coords[d] + 1)), length(ni))

function decompositions(N, nprocs)
    candidates = N == 2 ? [(nprocs, 1), (1, nprocs), (2, nprocs ÷ 2)] : [(nprocs, 1, 1), (1, 1, nprocs), (2, nprocs ÷ 2, 1)]
    return unique(filter(d -> prod(d) == nprocs, candidates))
end

# Owned output of a distributed run against the serial owned output, by global index.
function owned_mismatch(dlevel, slevel, dfields, sfields, N, T)
    worst = zero(T)
    for (A, S, location) in zip(host_fields(dfields), host_fields(sfields), locations(N))
        reference = Dict{NTuple{N, Int}, T}()
        for I in CartesianIndices(S)
            g = position_global(slevel.partition, location, Tuple(I))
            !isnothing(g) && is_owned(slevel.partition, location, Tuple(I)) && (reference[g] = S[I])
        end
        for I in CartesianIndices(A)
            g = position_global(dlevel.partition, location, Tuple(I))
            (isnothing(g) || !is_owned(dlevel.partition, location, Tuple(I))) && continue
            worst = max(worst, abs(A[I] - reference[g]))
        end
    end
    return worst
end

# Center field over all stored cells, halos included, from a global array.
function center_field(level, G)
    part = level.partition
    A = [G[ntuple(d -> JR.gmg_wrap(JR.gmg_storage_global(part, d, I[d], false), part.global_ni[d], part.periodic[d]), length(I))...] for I in Tuple.(CartesianIndices(size(level.correction.p)))]
    return copyto!(similar(level.correction.p), A)
end

center_values(level, f) = [f(Tuple(I)) for I in CartesianIndices(level.domain.ni)]

@testset "GMG distributed operator and DR sweep" begin
    MPI.Initialized() || MPI.Init()
    world = MPI.COMM_WORLD
    nprocs, rank = MPI.Comm_size(world), MPI.Comm_rank(world)
    for N in (2, 3), dims in decompositions(N, nprocs), periodic in (ntuple(_ -> false, N), ntuple(d -> d == 1, N))
        K = gmg_kernel_module(N)
        T = Float64
        # Periodic x needs a multiple of 3 for probing; every split keeps two owned cells.
        ni = N == 2 ? (12, 12) : (12, 8, 8)
        coords = Tuple(CartesianIndices(dims)[rank + 1]) .- 1
        xvi = ntuple(d -> collect(range(T(0), T(d); length = ni[d] + 1)) .+ T(0.02) .* sin.(range(0, 3; length = ni[d] + 1)), N)
        options = JR.GMGOptions(Val(N); coarsest_cells = 2, max_levels = 1)
        dlevel = only(JR.GMGHierarchy(gmg_backend_type, xvi, JR.GMGPartition(ni, block(ni, dims, coords); periodic, comm = world); options).levels)
        slevel = only(JR.GMGHierarchy(gmg_backend_type, xvi, JR.GMGPartition(ni, ntuple(d -> 1:ni[d], N); periodic); options).levels)
        no_slip = ntuple(d -> (d == N, false), N)
        dop, sop = JR.GMGLevelOperator(dlevel; no_slip), JR.GMGLevelOperator(slevel; no_slip)
        η = center_values(dlevel, I -> exp(sum(sin.(1.3 .* I))))
        C = center_values(dlevel, I -> 0.1 * (1 + cos(sum(I))))
        for (level, op) in ((dlevel, dop), (slevel, sop))
            K.gmg_set_coefficients!(op, center_field(level, η); compliance = center_field(level, C), γ = 0.5)
        end
        λd = K.gmg_refresh_bounds!(dop, dlevel.partition)
        λs = K.gmg_refresh_bounds!(sop, slevel.partition)
        @test λd ≈ λs rtol = 1.0e-12

        rng = Xoshiro(3)
        globals = map(locations(N)) do location
            randn(rng, T, ntuple(d -> ni[d] + (location[d][1] isa JR.GMGLinearFaces), N))
        end
        rhs = map(x -> 0.5 .* x, globals)
        set_fields!(slevel.correction, from_global(slevel, globals, T; owned_only = false))
        set_fields!(dlevel.correction, from_global(dlevel, globals, T))
        foreach((level, op) -> (JR.gmg_prepare!(op, level.correction); K.gmg_apply_operator!(op, level.correction)), (slevel, dlevel), (sop, dop))
        @test owned_mismatch(dlevel, slevel, dop.out, sop.out, N, T) <= 1.0e-11
        # Face-measure inner product: owned weights partition the serial one.
        ds = K.gmg_velocity_dot(sop, slevel, slevel.correction.v, sop.out.v)
        dd = K.gmg_velocity_dot(dop, dlevel, dlevel.correction.v, dop.out.v)
        @test dd ≈ ds rtol = 1.0e-11

        smoother = JR.GMGSmoother(; substeps = 3, spectral_range = 8, pressure_step = 0.7)
        set_fields!(slevel.rhs, from_global(slevel, rhs, T; owned_only = false))
        set_fields!(dlevel.rhs, from_global(dlevel, rhs, T))
        K.gmg_dr_smooth!(sop, slevel.correction, slevel.rhs, smoother)
        K.gmg_dr_smooth!(dop, dlevel.correction, dlevel.rhs, smoother)
        @test owned_mismatch(dlevel, slevel, dlevel.correction, slevel.correction, N, T) <= 1.0e-11
        @test all(isfinite, Array(dlevel.correction.p)[dlevel.partition.owned_local...])
    end
end
