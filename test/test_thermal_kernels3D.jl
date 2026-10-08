push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax3D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 3)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 3)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 3)
    JustRelax.CPUBackend
end

using JustPIC
const backend_JP = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    AMDGPU.ROCBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    CUDA.CUDABackend
else
    JustPIC.CPU
end

const JR3K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax3D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax3D
else
    JustRelax.JustRelax3D
end

function to_device(A::AbstractArray{T}) where {T}
    B = @zeros(size(A)..., eltype = T)
    copyto!(B, A)
    return B
end

@parallel_indices (I...) function _set_ratios!(ratios, vals)
    for p in 1:size(vals, 1)
        @index ratios[p, I...] = vals[p, I...]
    end
    return nothing
end

function set_ratios!(ratios, host_vals)
    vals = to_device(host_vals)
    @parallel (@idx size(ratios)) _set_ratios!(ratios, vals)
    return nothing
end

fT(x, y, z) = 1.0 + 0.3 * sin(2.1 * x) + 0.5 * y^2 + 0.2 * x * z - 0.4 * cos(3z)
fK(i, j, k) = 1.5 + 0.4 * sin(0.9 * i + 0.3 * j - 0.6 * k)
fθ(i, j, k) = 0.7 + 0.25 * cos(0.5 * i - 0.8 * j + 0.4 * k)

# Host reference: q = -K ∇T on the faces, K averaged from the two adjacent cells, with the
# pseudo-transient damping q ← (q_old θ + q) / (1 + θ).
function reference_flux(T, K, θ, q0, di, bc)
    ni = size(K)
    q = copy.(q0)
    q2 = ntuple(d -> zeros(size(q0[d])), 3)
    faces = ((:left, :right), (:front, :back), (:bot, :top))
    for d in 1:3
        e = ntuple(==(d), 3)
        for I in CartesianIndices(q0[d])
            Iv = Tuple(I)
            IL = ntuple(n -> n == d ? clamp(Iv[n] - 1, 1, ni[n]) : Iv[n], 3)
            IR = ntuple(n -> n == d ? clamp(Iv[n], 1, ni[n]) : Iv[n], 3)
            Kf = 0.5 * (K[IL...] + K[IR...])
            θf = 0.5 * (θ[IL...] + θ[IR...])
            Tp = T[(Iv .+ 1)...]
            Tm = T[(Iv .+ 1 .- e)...]
            q2[d][I] = -Kf * (Tp - Tm) / di[d]
            q[d][I] = (q0[d][I] * θf + q2[d][I]) / (1 + θf)
        end
        lo, hi = getfield(bc, faces[d][1]), getfield(bc, faces[d][2])
        lo isa Bool || (selectdim(q[d], d, 1) .= lo)
        hi isa Bool || (selectdim(q[d], d, size(q[d], d)) .= hi)
    end
    return q, q2
end

function reference_residual(T, Told, q, H, SH, ρCp, dt, di)
    R = zeros(size(H))
    for I in CartesianIndices(H)
        i, j, k = Tuple(I)
        divq = (q[1][i + 1, j, k] - q[1][i, j, k]) / di[1] +
            (q[2][i, j + 1, k] - q[2][i, j, k]) / di[2] +
            (q[3][i, j, k + 1] - q[3][i, j, k]) / di[3]
        R[I] = -ρCp[I] * (T[i + 1, j + 1, k + 1] - Told[i + 1, j + 1, k + 1]) / dt - divq + H[I] + SH[I]
    end
    return R
end

interior(A) = A[2:(end - 1), 2:(end - 1), 2:(end - 1)]

const no_flux_bc = (left = false, right = false, front = false, back = false, bot = false, top = false)

@testset "Thermal diffusion kernels 3D" begin
    nx, ny, nz = 5, 4, 3
    ni = nx, ny, nz
    di = 0.3, 0.2, 0.15
    grid = Geometry(ni, ni .* di; origin = (0.0, 0.0, 0.0))
    _di = grid._di
    _di_c, _di_v = _di.center, _di.vertex
    xg = ntuple(d -> [(i - 1.5) * di[d] for i in 1:(ni[d] + 2)], 3)

    Th = [fT(x, y, z) for x in xg[1], y in xg[2], z in xg[3]]
    Kh = [fK(i, j, k) for i in 1:nx, j in 1:ny, k in 1:nz]
    θh = [fθ(i, j, k) for i in 1:nx, j in 1:ny, k in 1:nz]
    q0 = (
        [0.1 * sin(i + 2j - k) for i in 1:(nx + 1), j in 1:ny, k in 1:nz],
        [0.1 * cos(2i - j + k) for i in 1:nx, j in 1:(ny + 1), k in 1:nz],
        [0.1 * sin(i * j + k) for i in 1:nx, j in 1:ny, k in 1:(nz + 1)],
    )
    function thermal_with(; T = Th, Told = Th, q = q0, q2 = nothing)
        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, T)
        copyto!(thermal.Told, Told)
        foreach(copyto!, @qT(thermal), q)
        isnothing(q2) || foreach(copyto!, @qT2(thermal), q2)
        return thermal
    end

    @testset "linear profile in a uniform conductor carries a uniform flux" begin
        g = (0.7, -1.3, 0.4)
        k0 = 2.5
        Tlin = [2.0 + g[1] * x + g[2] * y + g[3] * z for x in xg[1], y in xg[2], z in xg[3]]
        thermal = thermal_with(; T = Tlin, Told = Tlin)
        @parallel (@idx ni .+ 1) JR3K.compute_flux!(
            @qT(thermal)..., @qT2(thermal)..., thermal.T, to_device(fill(k0, ni)),
            to_device(fill(0.6, ni)), _di, no_flux_bc,
        )
        for d in 1:3
            @test all(Array(@qT2(thermal)[d]) .≈ -k0 * g[d])
        end
        zero_src = @zeros(ni...)
        @parallel (@idx ni) JR3K.check_res!(
            thermal.ResT, thermal.T, thermal.Told, @qT2(thermal)..., zero_src, zero_src,
            to_device(fill(3.0, ni)), JustRelax.DirichletBoundaryCondition(), 0.5, _di_v,
        )
        @test maximum(abs, Array(thermal.ResT)) < 1.0e-12
    end

    @testset "flux on non-uniform T, K and θ matches a hand-written stencil" begin
        for bc in (
                no_flux_bc,
                (left = 1.25, right = false, front = 0.3, back = false, bot = false, top = -0.5),
                (left = false, right = -0.75, front = false, back = 0.2, bot = 0.4, top = false),
            )
            thermal = thermal_with()
            @parallel (@idx ni .+ 1) JR3K.compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, to_device(Kh), to_device(θh), _di, bc
            )
            q, _ = reference_flux(Th, Kh, θh, q0, di, bc)
            for d in 1:3
                @test Array(@qT(thermal)[d]) ≈ q[d]
            end
        end
    end

    Hh = [0.2 + 0.1 * sin(i * j + k) for i in 1:nx, j in 1:ny, k in 1:nz]
    SHh = [0.05 * (1 + cos(i - j + k)) for i in 1:nx, j in 1:ny, k in 1:nz]
    ρCph = [2.0 + 0.3 * sin(i + j - k) for i in 1:nx, j in 1:ny, k in 1:nz]
    dτh = [0.05 + 0.01 * i + 0.02 * j + 0.015 * k for i in 1:nx, j in 1:ny, k in 1:nz]
    Toldh = Th .- [0.05 * sin(x + 2y - z) for x in xg[1], y in xg[2], z in xg[3]]
    dt = 0.5

    @testset "residual of the energy equation" begin
        Dv = zeros(ni .+ 2)
        Dv[3, 2, 3] = 4.0
        for dirichlet in (JustRelax.DirichletBoundaryCondition(), JustRelax.DirichletBoundaryCondition(to_device(Dv)))
            thermal = thermal_with(; Told = Toldh, q2 = q0)
            @parallel (@idx ni) JR3K.check_res!(
                thermal.ResT, thermal.T, thermal.Told, @qT2(thermal)..., to_device(Hh), to_device(SHh),
                to_device(ρCph), dirichlet, inv(dt), _di_v,
            )
            R = reference_residual(Th, Toldh, q0, Hh, SHh, ρCph, dt, di)
            isnothing(dirichlet.mask) || (R[2, 1, 2] = 0.0)
            @test Array(thermal.ResT) ≈ R
        end
    end

    @testset "pseudo-transient update is implicit in the storage term" begin
        Dv = zeros(ni .+ 2)
        Dv[3, 2, 3] = 4.0
        for dirichlet in (JustRelax.DirichletBoundaryCondition(), JustRelax.DirichletBoundaryCondition(to_device(Dv)))
            thermal = thermal_with(; Told = Toldh)
            @parallel (@idx ni) JR3K.update_T!(
                thermal.T, thermal.Told, @qT(thermal)..., to_device(Hh), to_device(SHh), to_device(ρCph),
                to_device(dτh), dirichlet, inv(dt), _di_c,
            )
            Tnew = Array(thermal.T)
            R = reference_residual(Tnew, Toldh, q0, Hh, SHh, ρCph, dt, di)
            ΔT_expected = dτh .* R
            if !isnothing(dirichlet.mask)
                @test Tnew[3, 2, 3] == 4.0
                ΔT_expected[2, 1, 2] = 4.0 - Th[3, 2, 3]
            end
            @test interior(Tnew) - interior(Th) ≈ ΔT_expected
        end
    end

    ρ0, α, β = (2.0, 2.6), (0.01, 0.03), (0.001, 0.002)
    Cp, k, Hr = (1.5, 0.9), (2.5, 0.7), (0.3, 0.1)
    rheology = ntuple(
        p -> SetMaterialParams(;
            Phase = p,
            Density = PT_Density(; ρ0 = ρ0[p], α = α[p], β = β[p], T0 = 0.0, P0 = 0.0),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp[p]),
            Conductivity = ConstantConductivity(; k = k[p]),
            RadioactiveHeat = ConstantRadioactiveHeat(; H_r = Hr[p]),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0),)),
        ), Val(2)
    )
    ρ(p, T, P) = ρ0[p] * (1 - α[p] * T + β[p] * P)

    φ1 = [0.5 + 0.5 * sin(0.8 * i + 1.1 * j - 0.7 * k) for i in 1:nx, j in 1:ny, k in 1:nz]
    φ1[1, 1, 1], φ1[end, end, end] = 1.0, 0.0
    φh = permutedims(cat(φ1, 1 .- φ1; dims = 4), (4, 1, 2, 3))
    phase_ratios = PhaseRatios(backend_JP, 2, ni)
    set_ratios!(phase_ratios.center, φh)
    Ph = [1.0 + 0.5 * i + 0.25 * j^2 - 0.1 * k for i in 1:nx, j in 1:ny, k in 1:nz]
    args = (; P = to_device(Ph))
    Ah = [0.02 * sin(i + 3j - k) for i in 1:nx, j in 1:ny, k in 1:nz]
    Kmix = φ1 .* k[1] .+ (1 .- φ1) .* k[2]
    Hr_mix = φ1 .* Hr[1] .+ (1 .- φ1) .* Hr[2]
    ρCp_mix(Tcur) = [
        φ1[I] * ρ(1, Tcur[(Tuple(I) .+ 1)...], Ph[I]) * Cp[1] +
            (1 - φ1[I]) * ρ(2, Tcur[(Tuple(I) .+ 1)...], Ph[I]) * Cp[2] for I in CartesianIndices(φ1)
    ]

    @testset "phase-ratio flux equals the flux of the phase-averaged conductivity" begin
        for bc in (no_flux_bc, (left = 0.3, right = -0.2, front = 0.1, back = -0.1, bot = 0.05, top = 0.15))
            thermal = thermal_with()
            pc = phase_ratios.center
            @parallel (@idx ni .+ 1) JR3K.compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, rheology, pc, pc, pc,
                to_device(θh), _di, args, bc,
            )
            q, _ = reference_flux(Th, Kmix, θh, q0, di, bc)
            for d in 1:3
                @test Array(@qT(thermal)[d]) ≈ q[d]
            end
        end
    end

    @testset "face-sized phase ratios (PhaseRatios.Vx/Vy/Vz) are read at the face" begin
        # A sharp material interface on face i0 (x), j0 (y) and k0 (z): phase 1 covers
        # every face before the interface, phase 2 from the interface on. Averaging the
        # ratio from the two *cell-center* neighbors of a face (as if `.Vx`/`.Vy`/`.Vz`
        # were center-sized) would blend the two conductivities at the interface instead
        # of reading the sharp jump directly at that face.
        i0, j0, k0 = 3, 2, 2
        ratio1_x = [i < i0 ? 1.0 : 0.0 for i in 1:(nx + 1), j in 1:ny, k in 1:nz]
        ratio1_y = [j < j0 ? 1.0 : 0.0 for i in 1:nx, j in 1:(ny + 1), k in 1:nz]
        ratio1_z = [k < k0 ? 1.0 : 0.0 for i in 1:nx, j in 1:ny, k in 1:(nz + 1)]
        set_ratios!(phase_ratios.Vx, permutedims(cat(ratio1_x, 1 .- ratio1_x; dims = 4), (4, 1, 2, 3)))
        set_ratios!(phase_ratios.Vy, permutedims(cat(ratio1_y, 1 .- ratio1_y; dims = 4), (4, 1, 2, 3)))
        set_ratios!(phase_ratios.Vz, permutedims(cat(ratio1_z, 1 .- ratio1_z; dims = 4), (4, 1, 2, 3)))

        Kx_ref = ratio1_x .* k[1] .+ (1 .- ratio1_x) .* k[2]
        Ky_ref = ratio1_y .* k[1] .+ (1 .- ratio1_y) .* k[2]
        Kz_ref = ratio1_z .* k[1] .+ (1 .- ratio1_z) .* k[2]
        qx_ref = [-Kx_ref[i, j, k] * (Th[i + 1, j + 1, k + 1] - Th[i, j + 1, k + 1]) / di[1] for i in 1:(nx + 1), j in 1:ny, k in 1:nz]
        qy_ref = [-Ky_ref[i, j, k] * (Th[i + 1, j + 1, k + 1] - Th[i + 1, j, k + 1]) / di[2] for i in 1:nx, j in 1:(ny + 1), k in 1:nz]
        qz_ref = [-Kz_ref[i, j, k] * (Th[i + 1, j + 1, k + 1] - Th[i + 1, j + 1, k]) / di[3] for i in 1:nx, j in 1:ny, k in 1:(nz + 1)]

        thermal = thermal_with()
        @parallel (@idx ni .+ 1) JR3K.compute_flux!(
            @qT(thermal)..., @qT2(thermal)..., thermal.T, rheology, phase_ratios.Vx, phase_ratios.Vy, phase_ratios.Vz,
            to_device(θh), _di, args, no_flux_bc,
        )
        @test Array(thermal.qTx2) ≈ qx_ref
        @test Array(thermal.qTy2) ≈ qy_ref
        @test Array(thermal.qTz2) ≈ qz_ref
        # the interface faces themselves carry the pure phase-2 conductivity, not a
        # blend of phase 1 and phase 2
        @test all(==(k[2]), Kx_ref[i0, :, :])
        @test all(==(k[2]), Ky_ref[:, j0, :])
        @test all(==(k[2]), Kz_ref[:, :, k0])
    end

    @testset "phase-ratio update and residual" begin
        ρCp_ref = ρCp_mix(Th)
        Hsrc = Hh .+ Hr_mix .+ Ah .* interior(Th)

        thermal = thermal_with(; Told = Toldh, q2 = q0)
        copyto!(thermal.adiabatic, Ah)
        @parallel (@idx ni) JR3K.check_res!(
            thermal.ResT, thermal.T, thermal.Told, @qT2(thermal)..., to_device(Hh), to_device(SHh),
            thermal.adiabatic, rheology, phase_ratios.center, JustRelax.DirichletBoundaryCondition(),
            inv(dt), _di_v, args,
        )
        @test Array(thermal.ResT) ≈ reference_residual(Th, Toldh, q0, Hsrc, SHh, ρCp_ref, dt, di)

        @parallel (@idx ni) JR3K.update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., to_device(Hh), to_device(SHh), thermal.adiabatic,
            rheology, phase_ratios.center, to_device(dτh), JustRelax.DirichletBoundaryCondition(),
            inv(dt), _di_v, args,
        )
        expected = similar(Hh)
        for I in CartesianIndices(Hh)
            i, j, kk = Tuple(I)
            divq = (q0[1][i + 1, j, kk] - q0[1][i, j, kk]) / di[1] +
                (q0[2][i, j + 1, kk] - q0[2][i, j, kk]) / di[2] +
                (q0[3][i, j, kk + 1] - q0[3][i, j, kk]) / di[3]
            src = -divq + Toldh[i + 1, j + 1, kk + 1] * ρCp_ref[I] / dt + Hsrc[I] + SHh[I]
            expected[I] = (dτh[I] * src + Th[i + 1, j + 1, kk + 1]) / (1 + dτh[I] * ρCp_ref[I] / dt)
        end
        @test interior(Array(thermal.T)) ≈ expected
    end

    @testset "adiabatic heating term α (P - P0) / Δt" begin
        thermal = ThermalArrays(backend, ni)
        P0h = Ph .- [0.1 * cos(i * j - k) for i in 1:nx, j in 1:ny, k in 1:nz]
        JR3K.adiabatic_heating!(thermal, (; P = to_device(Ph), P0 = to_device(P0h)), rheology, phase_ratios.center, inv(dt))
        @test Array(thermal.adiabatic) ≈ (Ph .- P0h) .* (φ1 .* α[1] .+ (1 .- φ1) .* α[2]) ./ dt
    end

    @testset "shear heating Χ τij (εij - εij_el)" begin
        G, Χ = (2.0, 5.0), (0.8, 0.5)
        rheo_sh = ntuple(
            p -> SetMaterialParams(;
                Phase = p,
                CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0), ConstantElasticity(; G = G[p]))),
                ShearHeat = ConstantShearheating(Χ[p]),
            ), Val(2)
        )
        dt_sh = 0.25
        stokes = StokesArrays(backend, ni)
        c = CartesianIndices(ni)
        τ = ntuple(n -> [sin(0.7i + 0.2j * n - 0.3k) / n for (i, j, k) in Tuple.(c)], 6)
        τo = ntuple(n -> 0.8 .* τ[n] .+ 0.05 * (-1)^n, 6)
        εn = ntuple(n -> [0.6 * sin(0.3i * n + 0.9j - 0.2k) for (i, j, k) in Tuple.(c)], 3)
        # edge-staggered shear strain rates: yz (nx, ny+1, nz+1), xz (nx+1, ny, nz+1), xy (nx+1, ny+1, nz)
        ε_yz = [0.4 * cos(0.7i + 0.35j - k) for i in 1:nx, j in 1:(ny + 1), k in 1:(nz + 1)]
        ε_xz = [0.3 * sin(0.5i - 0.6j + 0.9k) for i in 1:(nx + 1), j in 1:ny, k in 1:(nz + 1)]
        ε_xy = [0.35 * cos(0.2i + j + 0.4k) for i in 1:(nx + 1), j in 1:(ny + 1), k in 1:nz]
        foreach(copyto!, (stokes.τ.xx, stokes.τ.yy, stokes.τ.zz, stokes.τ.yz_c, stokes.τ.xz_c, stokes.τ.xy_c), τ)
        foreach(copyto!, (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.zz, stokes.τ_o.yz_c, stokes.τ_o.xz_c, stokes.τ_o.xy_c), τo)
        foreach(copyto!, (stokes.ε.xx, stokes.ε.yy, stokes.ε.zz, stokes.ε.yz, stokes.ε.xz, stokes.ε.xy), (εn..., ε_yz, ε_xz, ε_xy))

        # each edge component is averaged over the four edges bounding the cell
        ε_c = (
            εn...,
            [0.25 * (ε_yz[i, j, k] + ε_yz[i, j + 1, k] + ε_yz[i, j, k + 1] + ε_yz[i, j + 1, k + 1]) for (i, j, k) in Tuple.(c)],
            [0.25 * (ε_xz[i, j, k] + ε_xz[i + 1, j, k] + ε_xz[i, j, k + 1] + ε_xz[i + 1, j, k + 1]) for (i, j, k) in Tuple.(c)],
            [0.25 * (ε_xy[i, j, k] + ε_xy[i + 1, j, k] + ε_xy[i, j + 1, k] + ε_xy[i + 1, j + 1, k]) for (i, j, k) in Tuple.(c)],
        )
        # τ:ε with the off-diagonal components counted twice
        work(Gc, I) = sum((n > 3 ? 2 : 1) * τ[n][I] * (ε_c[n][I] - 0.5 * (τ[n][I] - τo[n][I]) / (Gc * dt_sh)) for n in 1:6)

        thermal = ThermalArrays(backend, ni)
        compute_shear_heating!(thermal, stokes, rheo_sh[1], dt_sh)
        expected = [max(0.0, Χ[1] * work(G[1], I)) for I in c]
        @test any(iszero, expected) && any(>(0), expected)
        @test Array(thermal.shear_heating) ≈ expected

        compute_shear_heating!(thermal, stokes, phase_ratios, rheo_sh, dt_sh)
        Gmix = φ1 .* G[1] .+ (1 .- φ1) .* G[2]
        Χmix = φ1 .* Χ[1] .+ (1 .- φ1) .* Χ[2]
        @test Array(thermal.shear_heating) ≈ [max(0.0, Χmix[I] * work(Gmix[I], I)) for I in c]
    end

    @testset "flux on a non-uniform grid uses the local center spacing" begin
        xv = [0.0, 0.2, 0.5, 0.6, 1.0, 1.3]
        yv = [0.0, 0.1, 0.35, 0.45, 0.7]
        zv = [0.0, 0.3, 0.4, 0.8]
        grid_nu = Geometry(PTArray(backend), xv, yv, zv)
        vertices = xv, yv, zv
        xc = map(v -> 0.5 .* (v[1:(end - 1)] .+ v[2:end]), vertices)
        xgn = map((v, c) -> [2v[1] - c[1]; c; 2v[end] - c[end]], vertices, xc)
        g, k0 = (0.7, -1.3, 0.4), 2.5
        thermal = thermal_with(; T = [1.0 + g[1] * x + g[2] * y + g[3] * z for x in xgn[1], y in xgn[2], z in xgn[3]])
        @parallel (@idx ni .+ 1) JR3K.compute_flux!(
            @qT(thermal)..., @qT2(thermal)..., thermal.T, to_device(fill(k0, ni)),
            to_device(fill(0.5, ni)), grid_nu._di, no_flux_bc,
        )
        # Boundary ghost temperatures are reflected about the physical boundary, as in
        # thermal_bcs!, so their distance to the adjacent center is the boundary cell width.
        @test all(Array(thermal.qTx2) .≈ -k0 * g[1])
        @test all(Array(thermal.qTy2) .≈ -k0 * g[2])
        @test all(Array(thermal.qTz2) .≈ -k0 * g[3])
    end

    @testset "update_T! and check_res! divide by the cell (vertex-to-vertex) spacing on a non-uniform grid" begin
        xv = [0.0, 0.2, 0.5, 0.6, 1.0, 1.3]
        yv = [0.0, 0.1, 0.35, 0.45, 0.7]
        zv = [0.0, 0.3, 0.4, 0.8]
        grid_nu = Geometry(PTArray(backend), xv, yv, zv)
        dv = Array.(grid_nu.di.vertex) # width of each thermal cell along each axis

        qh = (
            [0.05 * sin(i + 2j - k) for i in 1:(nx + 1), j in 1:ny, k in 1:nz],
            [0.05 * cos(2i - j + k) for i in 1:nx, j in 1:(ny + 1), k in 1:nz],
            [0.05 * sin(i * j - k) for i in 1:nx, j in 1:ny, k in 1:(nz + 1)],
        )
        function divq(i, j, k)
            return (qh[1][i + 1, j, k] - qh[1][i, j, k]) / dv[1][i] +
                (qh[2][i, j + 1, k] - qh[2][i, j, k]) / dv[2][j] +
                (qh[3][i, j, k + 1] - qh[3][i, j, k]) / dv[3][k]
        end

        thermal = thermal_with(; Told = Toldh, q = qh)
        H, SH, ρCp, dτρ = to_device(Hh), to_device(SHh), to_device(ρCph), to_device(dτh)
        @parallel (@idx ni) JR3K.update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., H, SH, ρCp, dτρ,
            JustRelax.DirichletBoundaryCondition(), inv(dt), grid_nu._di.vertex,
        )
        Tnew = Array(thermal.T)
        expected = similar(Hh)
        for I in CartesianIndices(Hh)
            i, j, k = Tuple(I)
            num = dτh[I] * (-divq(i, j, k) + Toldh[i + 1, j + 1, k + 1] * ρCph[I] / dt + Hh[I] + SHh[I]) + Th[i + 1, j + 1, k + 1]
            expected[I] = num / (1 + dτh[I] * ρCph[I] / dt)
        end
        @test interior(Tnew) ≈ expected

        thermal2 = thermal_with(; Told = Toldh, q2 = qh)
        @parallel (@idx ni) JR3K.check_res!(
            thermal2.ResT, thermal2.T, thermal2.Told, @qT2(thermal2)..., H, SH, ρCp,
            JustRelax.DirichletBoundaryCondition(), inv(dt), grid_nu._di.vertex,
        )
        R_expected = similar(Hh)
        for I in CartesianIndices(Hh)
            i, j, k = Tuple(I)
            R_expected[I] = -ρCph[I] * (Th[i + 1, j + 1, k + 1] - Toldh[i + 1, j + 1, k + 1]) / dt - divq(i, j, k) + Hh[I] + SHh[I]
        end
        @test Array(thermal2.ResT) ≈ R_expected
    end
end
