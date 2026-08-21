@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    using CUDA
end

using Test
using CellArrays, StaticArrays
using JustRelax, JustPIC
import JustRelax.JustRelax2D as JR2
import JustRelax.JustRelax3D as JR3

const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDABackend
else
    JustPIC.CPU
end

@testset "JustPIC interface" begin
    @testset "@index" begin
        @test Symbol("@index") in names(JR2)
        @test Symbol("@index") in names(JR3)
        # `JustPIC.@index` is KernelAbstractions' kernel-index macro; the cell accessor
        # JustRelax kernels use is CellArraysIndexing's.
        @test nameof(parentmodule(JR2.var"@index")) === :CellArraysIndexing
        @test JR2.var"@index" === JR3.var"@index"

        A = CPUCellArray{SVector{2, Float64}}(undef, 2, 2)
        JR2.@index A[1, 2, 2] = 3.0
        @test (JR2.@index A[1, 2, 2]) == 3.0
    end

    @testset "names taken from JustPIC" begin
        for name in (:PhaseRatios, :update_phase_ratios!, :nphases, :numphases, :cell_index)
            @test name in names(JustPIC)
            @test getfield(JR2, name) === getfield(JustPIC, name)
            @test getfield(JR3, name) === getfield(JustPIC, name)
        end
        # unexported upstream helpers, spelled `JustPIC.f` in src/phases
        for name in (
                :compute_dx, :face_offset, :distance, :interp1D_inner, :interp1D_extremas,
            )
            @test isdefined(JustPIC, name)
        end
    end

    @testset "backend tags" begin
        # JustPIC containers carry the KernelAbstractions backend as their first type
        # parameter, which is what the JustRelax GPU extensions dispatch on.
        phase_ratios = JustPIC.PhaseRatios(backend_JP, 2, (2, 2))
        @test phase_ratios isa JustPIC.PhaseRatios{backend_JP}
        # JustRelax has backend types of its own; they are unrelated to JustPIC's.
        @test !(CPUBackend <: JustPIC.CPU)
    end
end
