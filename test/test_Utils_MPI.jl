push!(LOAD_PATH, "..")

using Test
using JustRelax, JustRelax.JustRelax2D
import JustRelax.JustRelax2D: norm_mpi, masked_norm_mpi, sum_mpi
using MPI

igg = IGG(init_global_grid(4, 4, 1; init_MPI = true, select_device = false)...)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
me = igg.me

@testset "MPI reductions (norm_mpi / masked_norm_mpi / sum_mpi)" begin
    nx, ny = 4, 4

    # every rank holds the same local array, so the true multi-rank Allreduce must scale
    # the single-rank result by nranks — a bug that silently reduced only locally (or
    # dropped the Allreduce) would instead return the single-rank value here.
    A = fill(2.0, nx, ny)
    B = fill(3.0, nx, ny)

    @test norm_mpi(A) ≈ sqrt(nranks * sum(abs2, A))
    @test norm_mpi(A, B) ≈ sqrt(nranks * sum(abs2, A .* B))

    mask = trues(nx, ny)
    mask[1, 1] = false
    @test masked_norm_mpi(mask, A) ≈ sqrt(nranks * (sum(abs2, A) - abs2(A[1, 1])))
    @test masked_norm_mpi(mask, A, B) ≈ sqrt(nranks * (sum(abs2, A .* B) - abs2(A[1, 1] * B[1, 1])))

    @test sum_mpi(A) ≈ nranks * sum(A)
    @test sum_mpi((a, b) -> a * b, A, B) ≈ nranks * sum(A .* B)
    @test sum_mpi(abs2, A) ≈ nranks * sum(abs2, A)

    # rank-dependent data: each rank r holds (r+1) everywhere, so the reduction must
    # actually cross ranks rather than just replicate one rank's contribution.
    C = fill(Float64(me + 1), nx, ny)
    expected = sum(r -> (r + 1) * nx * ny, 0:(nranks - 1))
    @test sum_mpi(C) ≈ expected

    # norm_mpi(A) computes sqrt(Allreduce(sum(abs2, A))) without materializing A .^ 2; it must
    # agree with the direct sum(A .^ 2) formula under a real multi-rank Allreduce, using the
    # same rank-dependent C as above.
    direct_formula_norm = sqrt(MPI.Allreduce(sum(C .^ 2), MPI.SUM, MPI.COMM_WORLD))
    @test norm_mpi(C) ≈ direct_formula_norm
end

finalize_global_grid(; finalize_MPI = false)
