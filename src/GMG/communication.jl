# Only scalar, already locally owned contributions enter these reductions.
# Coarse halo transport and rank agglomeration are separate transfer-stage work.
@inline gmg_sum(value::Number, ::Nothing) = value
@inline gmg_max(value::Number, ::Nothing) = value
@inline gmg_all(value::Bool, ::Nothing) = value

@inline gmg_sum(value::T, comm::MPI.Comm) where {T <: Number} =
    MPI.Allreduce(value, MPI.SUM, comm)::T
@inline gmg_max(value::T, comm::MPI.Comm) where {T <: Number} =
    MPI.Allreduce(value, MPI.MAX, comm)::T
@inline gmg_all(value::Bool, comm::MPI.Comm) =
    MPI.Allreduce(Int(value), MPI.MIN, comm) == 1

@inline level_sum(level::GMGLevel, value::Number) = gmg_sum(value, level.partition.comm)
@inline level_max(level::GMGLevel, value::Number) = gmg_max(value, level.partition.comm)

function gmg_require(condition::Bool, message, comm)
    gmg_all(condition, comm) || throw(ArgumentError(message))
    return nothing
end

@inline gmg_collective_setup(domain, partition::GMGPartition{N, Nothing}, options) where {N} = nothing

# Validate loop-controlling setup before any rank can enter a different sequence
# of level collectives. Only host coordinate metadata is broadcast, never fields.
function gmg_collective_setup(domain::GMGDomain{N, T}, partition::GMGPartition{N, MPI.Comm}, options) where {N, T}
    comm = partition.comm
    specification = Int[
        domain.ni..., options.coarsening..., options.coarsest_cells...,
        options.max_levels, sizeof(T), partition.periodic..., partition.halo...,
    ]
    lower = MPI.Allreduce(specification, MPI.MIN, comm)
    upper = MPI.Allreduce(specification, MPI.MAX, comm)
    lower == upper || throw(ArgumentError("GMG hierarchy configuration must agree across its communicator"))
    for d in 1:N
        reference = copy(domain.xvi[d])
        MPI.Bcast!(reference, comm; root = 0)
        gmg_require(reference == domain.xvi[d], "GMG global coordinates must agree across their communicator", comm)
    end
    return nothing
end
