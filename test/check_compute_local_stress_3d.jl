using Test
using JustRelax, JustRelax.JustRelax2D
using GeoParams
using StaticArrays

@testset "compute_local_stress 3D" begin
    eta = 2.0
    G = 10.0
    Kb = 20.0
    dt = 0.25

    elastic = ConstantElasticity(; G = G, Kb = Kb)
    viscous = LinearViscous(; η = eta)
    rheology = (
        SetMaterialParams(;
            Phase = 1,
            CompositeRheology = CompositeRheology((viscous, elastic)),
            Elasticity = elastic,
        ),
    )

    eps_ij = (0.10, -0.03, -0.07, 0.02, 0.03, -0.01) # xx, yy, zz, yz, xz, xy
    tau_old = ntuple(_ -> 0.0, Val(6))
    phase_ratio = SVector(1.0)

    out = JustRelax2D.compute_local_stress(
        eps_ij,
        tau_old,
        eta,
        0.0, # P
        0.0, # lambda
        1.0, # lambda relaxation
        rheology,
        phase_ratio,
        dt,
        0.0, # EII
    )

    @test length(out) == 17

    tau = out[1:6]
    eps_pl = out[7:12]
    tauII, lambda, deltaP, eta_vep, eps_vol_pl = out[13:17]

    eta_ve = eta * G * dt / (eta + G * dt)
    expected_tau = 2 .* eta_ve .* eps_ij

    @test all(isapprox.(tau, expected_tau))
    @test all(iszero, eps_pl)
    @test tauII > 0
    @test lambda == 0.0
    @test deltaP == 0.0
    @test eta_vep ≈ eta_ve
    @test eps_vol_pl == 0.0
end
