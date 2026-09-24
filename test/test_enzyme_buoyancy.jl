get!(ENV, "JULIA_JUSTRELAX_BACKEND", "CPU")

using Test, GeoParams, JustPIC, ParallelStencil
using JustRelax, JustRelax.JustRelax2D

@init_parallel_stencil(Threads, Float64, 2)

@parallel_indices (i, j) function _buoyancy_test_phases!(phase)
    @index phase[1, i, j] = i == 1 ? 1.0 : 0.3
    @index phase[2, i, j] = i == 1 ? 0.0 : 0.7
    return nothing
end

@testset "Free-surface stabilization adjoint" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    ρg = (@zeros(ni...), @zeros(ni...))
    ρg[2] .= reshape(collect(1.0:prod(ni)), ni)
    Ry_seed = reshape(cos.(1:length(adjoint.R.Ry)), size(adjoint.R.Ry))
    adjoint.R.Ry .= Ry_seed
    free_surface_dt = 0.3

    JustRelax2D.enzyme_compute_PH_residual_V!(
        stokes, adjoint, ρg, grid._di, ni; free_surface_dt
    )

    expected = zero(adjoint.V.Vy)
    for i in axes(adjoint.R.Ry, 1), j in axes(adjoint.R.Ry, 2)
        dρgdy = (ρg[2][i, j + 1] - ρg[2][i, j]) * grid._di.center[2]
        expected[i + 1, j + 1] = Ry_seed[i, j] * dρgdy * free_surface_dt
    end
    @test adjoint.V.Vy ≈ expected
end

# Reverse the actual momentum residual, including the original buoyancy evaluation,
# and compare its pressure column with finite differences at every cell.
@testset "Pressure-density-buoyancy adjoint" begin
    ni = (3, 2)
    grid = Geometry(ni, (3.0, 2.0))
    phases = PhaseRatios(JustPIC.CPU, 2, ni)
    @parallel (@idx ni) _buoyancy_test_phases!(phases.center)
    stokes = StokesArrays(CPUBackend, ni)
    adjoint = AdjointStokesArrays(CPUBackend, ni)
    ρg = (@zeros(ni...), @zeros(ni...))
    stokes.P .= reshape(collect(1.0:prod(ni)), ni) ./ 10
    T = reshape(collect(1.0:20.0), ni .+ 2) ./ 10
    seedx = reshape(sin.(1:length(stokes.R.Rx)), size(stokes.R.Rx))
    seedy = reshape(cos.(1:length(stokes.R.Ry)), size(stokes.R.Ry))
    P_initial = copy(stokes.P)
    models = (
        PT_Density(; ρ0 = 2.0, α = 0.1, β = 0.3, T0 = 0.2, P0 = 0.1),
        Compressible_Density(; ρ0 = 3.0, β = 0.2, P0 = 0.1),
    )
    for gravity in (ConstantGravity(; g = 1.4), DippingGravity(40.0, 0.0, 1.4)),
            pressure_dependent in (true, false)
        rheology = ntuple(2) do p
            SetMaterialParams(;
                Phase = p,
                Density = pressure_dependent ? models[p] : ConstantDensity(; ρ = 2.0 + p),
                Gravity = gravity,
            )
        end
        args = (; T, P = stokes.P)
        objective = function ()
            compute_ρg!(ρg, phases, rheology, args)
            @parallel (@idx ni) JustRelax2D.compute_PH_residual_V!(
                stokes.R.Rx, stokes.R.Ry, stokes.P, stokes.ΔPψ,
                stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, ρg...,
                grid._di.center, grid._di.vertex,
            )
            return sum(seedx .* stokes.R.Rx) + sum(seedy .* stokes.R.Ry) + 0.37sum(stokes.P)
        end
        objective()
        # Repeated application must clear buoyancy scratch and preserve objective seeds.
        previous = nothing
        for _ in 1:2
            adjoint.P .= 0.37
            adjoint.R.Rx .= seedx
            adjoint.R.Ry .= seedy
            adjoint.dρgx .= 123.0
            adjoint.ρ .= 123.0
            JustRelax2D.enzyme_compute_PH_residual_V!(
                stokes, adjoint, ρg, grid._di, ni, rheology, phases, args
            )
            isnothing(previous) || @test adjoint.P ≈ previous
            previous = copy(adjoint.P)
            @test stokes.P == P_initial
        end
        for I in CartesianIndices(stokes.P)
            h = 1.0e-6
            stokes.P[I] = P_initial[I] + h
            plus = objective()
            stokes.P[I] = P_initial[I] - h
            minus = objective()
            stokes.P[I] = P_initial[I]
            @test adjoint.P[I] ≈ (plus - minus) / (2h) rtol = 1.0e-6 atol = 1.0e-8
        end
        # Confirm that this case would fail with the old, fixed-buoyancy adjoint.
        adjoint.P .= 0.37
        adjoint.R.Rx .= seedx
        adjoint.R.Ry .= seedy
        JustRelax2D.enzyme_compute_PH_residual_V!(stokes, adjoint, ρg, grid._di, ni)
        if pressure_dependent
            @test maximum(abs, adjoint.P .- previous) > 1.0e-3
        else
            @test adjoint.P ≈ previous
        end
    end
end
