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

@testset "GMG level exchange and distributed transfers" begin
    MPI.Initialized() || MPI.Init()
    world = MPI.COMM_WORLD
    nprocs, rank = MPI.Comm_size(world), MPI.Comm_rank(world)
    for N in (2, 3), dims in decompositions(N, nprocs), periodic in (ntuple(_ -> false, N), ntuple(d -> d == 1, N))
        K = gmg_kernel_module(N)
        T = Float64
        ni = N == 2 ? (16, 8) : (8, 8, 8)
        coords = Tuple(CartesianIndices(dims)[rank + 1]) .- 1
        xvi = ntuple(d -> collect(range(T(0), T(d); length = ni[d] + 1)) .+ T(0.01) .* sin.(range(0, 3; length = ni[d] + 1)), N)
        options = JR.GMGOptions(Val(N); coarsest_cells = 2, max_levels = 2)
        distributed = JR.GMGHierarchy(gmg_backend_type, xvi, JR.GMGPartition(ni, block(ni, dims, coords); periodic, comm = world); options)
        # Rank-local cell counts may stop a direction; the serial reference follows the same ratios.
        serial_options = JR.GMGOptions(Val(N); coarsening = distributed.transitions[1].ratio, coarsest_cells = 2, max_levels = 2)
        serial = JR.GMGHierarchy(gmg_backend_type, xvi, JR.GMGPartition(ni, ntuple(d -> 1:ni[d], N); periodic); options = serial_options)
        @test length(distributed.levels) == length(serial.levels) == 2

        # Halos hold the owning rank's values, including edges and corners.
        for level in (distributed.levels..., serial.levels...)
            part = level.partition
            exchange = @inferred JR.GMGExchange(level)
            code = map(locations(N)) do location
                shape = ntuple(d -> part.global_ni[d] + (location[d][1] isa JR.GMGLinearFaces), N)
                [T(sum(I[d] * 100^(d - 1) for d in 1:N)) for I in Tuple.(CartesianIndices(shape))]
            end
            set_fields!(level.correction, from_global(level, code, T))
            @test @inferred(JR.exchange_level!(exchange, level.correction)) === nothing
            filled = host_fields(level.correction)
            expected = from_global(level, code, T; owned_only = false)
            for (A, B) in zip(filled, expected)
                both = .!isnan.(B)
                @test A[both] == B[both]
            end
        end

        # Transfers on exchanged halos reproduce the serial owned action.
        fine, coarse = distributed.levels
        sfine, scoarse = serial.levels
        transfer = JR.GMGTransfer(fine, coarse, distributed.transitions[1]; velocity = JR.GMGSmoothVelocity(), no_slip = ntuple(d -> (d == N, false), N))
        stransfer = JR.GMGTransfer(sfine, scoarse, serial.transitions[1]; velocity = JR.GMGSmoothVelocity(), no_slip = ntuple(d -> (d == N, false), N))
        fexchange, cexchange = JR.GMGExchange(fine), JR.GMGExchange(coarse)
        rng = Xoshiro(42)
        global_fields(level) = map(locations(N)) do location
            rand(rng, T, ntuple(d -> level.domain.ni[d] + (location[d][1] isa JR.GMGLinearFaces), N))
        end
        coarse_values, fine_values = global_fields(coarse), global_fields(fine)
        set_fields!(scoarse.correction, from_global(scoarse, coarse_values, T; owned_only = false))
        set_fields!(sfine.residual, from_global(sfine, fine_values, T; owned_only = false))
        foreach(f -> fill!(f, zero(T)), (sfine.correction.p, sfine.correction.v...))
        K.gmg_prolong!(sfine.correction, scoarse.correction, stransfer; β = 0)
        K.gmg_restrict!(scoarse.rhs, sfine.residual, stransfer)

        set_fields!(coarse.correction, from_global(coarse, coarse_values, T))
        set_fields!(fine.residual, from_global(fine, fine_values, T))
        foreach(f -> fill!(f, zero(T)), (fine.correction.p, fine.correction.v...))
        JR.exchange_level!(cexchange, coarse.correction)
        JR.exchange_level!(fexchange, fine.residual)
        K.gmg_prolong!(fine.correction, coarse.correction, transfer; β = 0)
        K.gmg_restrict!(coarse.rhs, fine.residual, transfer)

        for (dlevel, slevel, out, sout) in ((fine, sfine, fine.correction, sfine.correction), (coarse, scoarse, coarse.rhs, scoarse.rhs))
            dpart, spart = dlevel.partition, slevel.partition
            for (A, S, location) in zip(host_fields(out), host_fields(sout), locations(N))
                reference = Dict{NTuple{N, Int}, T}()
                for I in CartesianIndices(S)
                    g = position_global(spart, location, Tuple(I))
                    !isnothing(g) && is_owned(spart, location, Tuple(I)) && (reference[g] = S[I])
                end
                worst = zero(T)
                for I in CartesianIndices(A)
                    g = position_global(dpart, location, Tuple(I))
                    (isnothing(g) || !is_owned(dpart, location, Tuple(I))) && continue
                    worst = max(worst, abs(A[I] - reference[g]))
                end
                @test worst <= 1.0e-13
            end
        end
    end
end
