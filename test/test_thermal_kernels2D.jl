push!(LOAD_PATH, "..")

@static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    using AMDGPU
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    import CUDA
end

using Test
using GeoParams
using JustRelax, JustRelax.JustRelax2D
using ParallelStencil

const backend = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    @init_parallel_stencil(AMDGPU, Float64, 2)
    JustRelax.AMDGPUBackend
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    @init_parallel_stencil(CUDA, Float64, 2)
    JustRelax.CUDABackend
else
    @init_parallel_stencil(Threads, Float64, 2)
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

const JR2K = @static if ENV["JULIA_JUSTRELAX_BACKEND"] === "AMDGPU"
    Base.get_extension(JustRelax, :JustRelaxAMDGPUExt).JustRelax2D
elseif ENV["JULIA_JUSTRELAX_BACKEND"] === "CUDA"
    Base.get_extension(JustRelax, :JustRelaxCUDAExt).JustRelax2D
else
    JustRelax.JustRelax2D
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

# Smooth, non-polynomial fields so that every stencil entry is distinct.
fT(x, y) = 1.0 + 0.3 * sin(2.1 * x) + 0.5 * y^2 + 0.2 * x * y
fK(i, j) = 1.5 + 0.4 * sin(0.9 * i + 0.3 * j)
fθ(i, j) = 0.7 + 0.25 * cos(0.5 * i - 0.8 * j)

# Host reference: q = -K ∂T/∂x on the faces, with the conductivity averaged from the two
# adjacent cells and the pseudo-transient damping q ← (q_old θ + q) / (1 + θ).
function reference_flux(T, K, θ, qx0, qy0, dx, dy, bc)
    nx, ny = size(K)
    qx, qy = copy(qx0), copy(qy0)
    qx2, qy2 = zeros(nx + 1, ny), zeros(nx, ny + 1)
    for j in 1:ny, i in 1:(nx + 1)
        iL, iR = clamp(i - 1, 1, nx), clamp(i, 1, nx)
        Kf = 0.5 * (K[iL, j] + K[iR, j])
        θf = 0.5 * (θ[iL, j] + θ[iR, j])
        qx2[i, j] = -Kf * (T[i + 1, j + 1] - T[i, j + 1]) / dx
        qx[i, j] = (qx0[i, j] * θf + qx2[i, j]) / (1 + θf)
    end
    for j in 1:(ny + 1), i in 1:nx
        jB, jT = clamp(j - 1, 1, ny), clamp(j, 1, ny)
        Kf = 0.5 * (K[i, jB] + K[i, jT])
        θf = 0.5 * (θ[i, jB] + θ[i, jT])
        qy2[i, j] = -Kf * (T[i + 1, j + 1] - T[i + 1, j]) / dy
        qy[i, j] = (qy0[i, j] * θf + qy2[i, j]) / (1 + θf)
    end
    bc.left isa Bool || (qx[1, :] .= bc.left)
    bc.right isa Bool || (qx[end, :] .= bc.right)
    bc.bot isa Bool || (qy[:, 1] .= bc.bot)
    bc.top isa Bool || (qy[:, end] .= bc.top)
    return qx, qy, qx2, qy2
end

# Energy residual R = -ρCp (T - T_old)/Δt - ∇·q + H + H_shear on the interior nodes.
function reference_residual(T, Told, qx, qy, H, SH, ρCp, dt, dx, dy)
    nx, ny = size(H)
    R = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        divq = (qx[i + 1, j] - qx[i, j]) / dx + (qy[i, j + 1] - qy[i, j]) / dy
        R[i, j] = -ρCp[i, j] * (T[i + 1, j + 1] - Told[i + 1, j + 1]) / dt - divq + H[i, j] + SH[i, j]
    end
    return R
end

const no_flux_bc = (left = false, right = false, bot = false, top = false)

@testset "Thermal diffusion kernels 2D" begin
    nx, ny = 7, 6
    ni = nx, ny
    dx, dy = 0.3, 0.15
    grid = Geometry(ni, (nx * dx, ny * dy); origin = (0.0, 0.0))
    _di = grid._di
    _di_c, _di_v = _di.center, _di.vertex
    # cell-center coordinates extended by one ghost node on each side
    xg = [(i - 1.5) * dx for i in 1:(nx + 2)]
    yg = [(j - 1.5) * dy for j in 1:(ny + 2)]

    Th = [fT(x, y) for x in xg, y in yg]
    Kh = [fK(i, j) for i in 1:nx, j in 1:ny]
    θh = [fθ(i, j) for i in 1:nx, j in 1:ny]
    qx0 = [0.1 * sin(i + 2j) for i in 1:(nx + 1), j in 1:ny]
    qy0 = [0.1 * cos(2i - j) for i in 1:nx, j in 1:(ny + 1)]

    @testset "linear profile in a uniform conductor carries a uniform flux" begin
        thermal = ThermalArrays(backend, ni)
        a, b, c, k0 = 2.0, 0.7, -1.3, 2.5
        copyto!(thermal.T, [a + b * x + c * y for x in xg, y in yg])
        copyto!(thermal.Told, Array(thermal.T))
        K = to_device(fill(k0, ni))
        θ = to_device(fill(0.6, ni))
        @parallel (@idx ni .+ 1) JR2K.compute_flux!(
            @qT(thermal)..., @qT2(thermal)..., thermal.T, K, θ, _di, no_flux_bc
        )
        @test all(Array(thermal.qTx2) .≈ -k0 * b)
        @test all(Array(thermal.qTy2) .≈ -k0 * c)
        # a uniform flux has zero divergence, so a steady state with T = T_old has no residual
        ρCp = to_device(fill(3.0, ni))
        zero_src = @zeros(ni...)
        @parallel (@idx ni) JR2K.check_res!(
            thermal.ResT, thermal.T, thermal.Told, @qT2(thermal)..., zero_src, zero_src,
            ρCp, JustRelax.DirichletBoundaryCondition(), 0.5, _di_v,
        )
        @test maximum(abs, Array(thermal.ResT)) < 1.0e-12
        # and the steady state is a fixed point of the pseudo-transient update
        T_before = Array(thermal.T)
        @parallel (@idx ni) JR2K.update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., zero_src, zero_src, ρCp,
            to_device(fill(0.4, ni)), JustRelax.DirichletBoundaryCondition(), 2.0, _di_c,
        )
        @test Array(thermal.T) ≈ T_before
    end

    @testset "flux on non-uniform T, K and θ matches a hand-written stencil" begin
        for bc in (no_flux_bc, (left = 1.25, right = false, bot = false, top = -0.5), (left = false, right = -0.75, bot = 0.4, top = false))
            thermal = ThermalArrays(backend, ni)
            copyto!(thermal.T, Th)
            copyto!(thermal.qTx, qx0)
            copyto!(thermal.qTy, qy0)
            @parallel (@idx ni .+ 1) JR2K.compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, to_device(Kh), to_device(θh), _di, bc
            )
            qx, qy, qx2, qy2 = reference_flux(Th, Kh, θh, qx0, qy0, dx, dy, bc)
            @test Array(thermal.qTx) ≈ qx
            @test Array(thermal.qTy) ≈ qy
            # the undamped flux is independent of the prescribed boundary flux
            @test Array(thermal.qTx2)[2:(end - 1), :] ≈ qx2[2:(end - 1), :]
            @test Array(thermal.qTy2)[:, 2:(end - 1)] ≈ qy2[:, 2:(end - 1)]
        end
    end

    Hh = [0.2 + 0.1 * sin(i * j) for i in 1:nx, j in 1:ny]
    SHh = [0.05 * (1 + cos(i - j)) for i in 1:nx, j in 1:ny]
    ρCph = [2.0 + 0.3 * sin(i + j) for i in 1:nx, j in 1:ny]
    dτh = [0.05 + 0.01 * i + 0.02 * j for i in 1:nx, j in 1:ny]
    Toldh = Th .- [0.05 * sin(x + 2y) for x in xg, y in yg]
    dt = 0.5

    @testset "residual of the energy equation" begin
        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, Th)
        copyto!(thermal.Told, Toldh)
        copyto!(thermal.qTx2, qx0)
        copyto!(thermal.qTy2, qy0)
        @parallel (@idx ni) JR2K.check_res!(
            thermal.ResT, thermal.T, thermal.Told, @qT2(thermal)..., to_device(Hh), to_device(SHh),
            to_device(ρCph), JustRelax.DirichletBoundaryCondition(), inv(dt), _di_v,
        )
        @test Array(thermal.ResT) ≈ reference_residual(Th, Toldh, qx0, qy0, Hh, SHh, ρCph, dt, dx, dy)
    end

    @testset "pseudo-transient update is implicit in the storage term" begin
        # T_new - T = dτ_ρ R(T_new), with R evaluated at the updated temperature
        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, Th)
        copyto!(thermal.Told, Toldh)
        copyto!(thermal.qTx, qx0)
        copyto!(thermal.qTy, qy0)
        @parallel (@idx ni) JR2K.update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., to_device(Hh), to_device(SHh), to_device(ρCph),
            to_device(dτh), JustRelax.DirichletBoundaryCondition(), inv(dt), _di_c,
        )
        Tnew = Array(thermal.T)
        R = reference_residual(Tnew, Toldh, qx0, qy0, Hh, SHh, ρCph, dt, dx, dy)
        @test Tnew[2:(end - 1), 2:(end - 1)] - Th[2:(end - 1), 2:(end - 1)] ≈ dτh .* R
        # ghost nodes are left to the boundary conditions
        @test Tnew[1, :] == Th[1, :]
        @test Tnew[:, end] == Th[:, end]

        # the launch wrapper runs the same update over the whole interior
        thermal2 = ThermalArrays(backend, ni)
        copyto!(thermal2.T, Th)
        copyto!(thermal2.Told, Toldh)
        copyto!(thermal2.qTx, qx0)
        copyto!(thermal2.qTy, qy0)
        copyto!(thermal2.H, Hh)
        copyto!(thermal2.shear_heating, SHh)
        JR2K.update_T(
            nothing, nothing, thermal2, to_device(ρCph), (; dτ_ρ = to_device(dτh)),
            JustRelax.DirichletBoundaryCondition(), inv(dt), _di_c, ni,
        )
        @test Array(thermal2.T) ≈ Tnew
    end

    @testset "Dirichlet nodes are pinned and carry no residual" begin
        Dv = zeros(nx + 2, ny + 2)
        Dv[4, 3] = 7.5
        dirichlet = JustRelax.DirichletBoundaryCondition(to_device(Dv))
        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, Th)
        copyto!(thermal.Told, Toldh)
        copyto!(thermal.qTx, qx0)
        copyto!(thermal.qTy, qy0)
        H, SH, ρCp = to_device(Hh), to_device(SHh), to_device(ρCph)
        @parallel (@idx ni) JR2K.update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., H, SH, ρCp, to_device(dτh), dirichlet, inv(dt), _di_c
        )
        Tnew = Array(thermal.T)
        @test Tnew[4, 3] == 7.5
        free = trues(nx + 2, ny + 2)
        free[4, 3] = false
        free[[1, end], :] .= false
        free[:, [1, end]] .= false
        R = reference_residual(Tnew, Toldh, qx0, qy0, Hh, SHh, ρCph, dt, dx, dy)
        @test (Tnew - Th)[free] ≈ (dτh .* R)[free[2:(end - 1), 2:(end - 1)]]

        copyto!(thermal.qTx2, qx0)
        copyto!(thermal.qTy2, qy0)
        @parallel (@idx ni) JR2K.check_res!(
            thermal.ResT, thermal.T, thermal.Told, @qT2(thermal)..., H, SH, ρCp, dirichlet, inv(dt), _di_v
        )
        ResT = Array(thermal.ResT)
        @test ResT[3, 2] == 0
        @test ResT[free[2:(end - 1), 2:(end - 1)]] ≈ R[free[2:(end - 1), 2:(end - 1)]]
    end

    @testset "update_ΔT!" begin
        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, Th)
        copyto!(thermal.Told, Toldh)
        @parallel (@idx size(thermal.T)) JR2K.update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)
        @test Array(thermal.ΔT) ≈ Th - Toldh
    end

    # Two materials with distinct conductivity, heat capacity, radiogenic heat, and a
    # temperature- and pressure-dependent density.
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

    φ1 = [0.5 + 0.5 * sin(0.8 * i + 1.1 * j) for i in 1:nx, j in 1:ny]
    φ1[1, 1], φ1[end, end] = 1.0, 0.0
    φh = permutedims(cat(φ1, 1 .- φ1; dims = 3), (3, 1, 2))
    phase_ratios = PhaseRatios(backend_JP, 2, ni)
    set_ratios!(phase_ratios.center, φh)
    phase_id = [isodd(i + j) ? 1 : 2 for i in 1:nx, j in 1:ny]
    phase_id_dev = to_device(phase_id)
    Ph = [1.0 + 0.5 * i + 0.25 * j^2 for i in 1:nx, j in 1:ny]
    args = (; P = to_device(Ph))
    Ah = [0.02 * sin(i + 3j) for i in 1:nx, j in 1:ny]

    # (phase field, conductivity, ρCp(T, P), radiogenic heat) for each way of passing phases
    phase_cases = (
        (
            "phase ratios", phase_ratios.center, φ1 .* k[1] .+ (1 .- φ1) .* k[2],
            (T, P, i, j) -> φ1[i, j] * ρ(1, T, P) * Cp[1] + (1 - φ1[i, j]) * ρ(2, T, P) * Cp[2],
            φ1 .* Hr[1] .+ (1 .- φ1) .* Hr[2],
        ),
        (
            "phase indices", phase_id_dev, [k[p] for p in phase_id],
            (T, P, i, j) -> ρ(phase_id[i, j], T, P) * Cp[phase_id[i, j]], [Hr[p] for p in phase_id],
        ),
    )

    @testset "rheology-based flux with $name" for (name, phase, Kref, _, _) in phase_cases
        for bc in (no_flux_bc, (left = 0.3, right = -0.2, bot = 0.1, top = false))
            thermal = ThermalArrays(backend, ni)
            copyto!(thermal.T, Th)
            copyto!(thermal.qTx, qx0)
            copyto!(thermal.qTy, qy0)
            @parallel (@idx ni .+ 1) JR2K.compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, rheology, phase, phase,
                to_device(θh), _di, args, bc,
            )
            qx, qy, _, _ = reference_flux(Th, Kref, θh, qx0, qy0, dx, dy, bc)
            @test Array(thermal.qTx) ≈ qx
            @test Array(thermal.qTy) ≈ qy
        end
    end

    @testset "phase ratios sized neither per cell nor per face are rejected" begin
        @test_throws DimensionMismatch JustRelax2D.phase_at_face(ones(Int, 7, 4), 4, 1, (2, 2), (1, 2), (2, 2))
    end

    @testset "face-sized phase ratios (PhaseRatios.Vx/Vy) are read at the face" begin
        # A sharp material interface sitting exactly on face i0 (x) and j0 (y): phase 1
        # covers every face before the interface, phase 2 from the interface on.
        # Averaging the ratio from the two *cell-center* neighbors of a face (as if
        # `.Vx`/`.Vy` were center-sized, i.e. nx/ny entries instead of nx+1/ny+1) would
        # blend the two conductivities at the interface instead of reading the sharp
        # jump directly at that face.
        i0, j0 = 4, 3
        ratio1_x = [i < i0 ? 1.0 : 0.0 for i in 1:(nx + 1), j in 1:ny]
        ratio1_y = [j < j0 ? 1.0 : 0.0 for i in 1:nx, j in 1:(ny + 1)]
        set_ratios!(phase_ratios.Vx, permutedims(cat(ratio1_x, 1 .- ratio1_x; dims = 3), (3, 1, 2)))
        set_ratios!(phase_ratios.Vy, permutedims(cat(ratio1_y, 1 .- ratio1_y; dims = 3), (3, 1, 2)))

        Kx_ref = ratio1_x .* k[1] .+ (1 .- ratio1_x) .* k[2]
        Ky_ref = ratio1_y .* k[1] .+ (1 .- ratio1_y) .* k[2]
        qx_ref = [-Kx_ref[i, j] * (Th[i + 1, j + 1] - Th[i, j + 1]) / dx for i in 1:(nx + 1), j in 1:ny]
        qy_ref = [-Ky_ref[i, j] * (Th[i + 1, j + 1] - Th[i + 1, j]) / dy for i in 1:nx, j in 1:(ny + 1)]

        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, Th)
        @parallel (@idx ni .+ 1) JR2K.compute_flux!(
            @qT(thermal)..., @qT2(thermal)..., thermal.T, rheology, phase_ratios.Vx, phase_ratios.Vy,
            to_device(θh), _di, args, no_flux_bc,
        )
        @test Array(thermal.qTx2) ≈ qx_ref
        @test Array(thermal.qTy2) ≈ qy_ref
        # the interface face itself carries the pure phase-2 conductivity, not a blend
        # of phase 1 and phase 2
        @test all(==(k[2]), Kx_ref[i0, :])
        @test all(==(k[2]), Ky_ref[:, j0])
    end

    @testset "single-rheology flux equals the pure-phase flux" begin
        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, Th)
        @parallel (@idx ni .+ 1) JR2K.compute_flux!(
            @qT(thermal)..., @qT2(thermal)..., thermal.T, rheology[1], nothing, nothing,
            to_device(θh), _di, args, no_flux_bc,
        )
        _, _, qx2, qy2 = reference_flux(Th, fill(k[1], ni), θh, qx0, qy0, dx, dy, no_flux_bc)
        @test Array(thermal.qTx2) ≈ qx2
        @test Array(thermal.qTy2) ≈ qy2
    end

    # Reference for the rheology-based kernels: the array-based kernels, fed with ρCp
    # evaluated at the current temperature and the combined source H + H_r + A·T.
    function rheology_reference_inputs(Tcur, ρCp_fn, Hr_field)
        ρCp = [ρCp_fn(Tcur[i + 1, j + 1], Ph[i, j], i, j) for i in 1:nx, j in 1:ny]
        Hsrc = Hh .+ Hr_field .+ Ah .* Tcur[2:(end - 1), 2:(end - 1)]
        return ρCp, Hsrc
    end

    all_cases = (
        phase_cases...,
        ("single rheology", nothing, nothing, (T, P, i, j) -> ρ(1, T, P) * Cp[1], fill(Hr[1], ni)),
    )
    @testset "rheology-based update and residual with $name" for (name, phase, _, ρCp_fn, Hr_field) in all_cases
        rheo = isnothing(phase) ? rheology[1] : rheology
        ρCp_ref, Hsrc = rheology_reference_inputs(Th, ρCp_fn, Hr_field)

        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, Th)
        copyto!(thermal.Told, Toldh)
        copyto!(thermal.qTx, qx0)
        copyto!(thermal.qTy, qy0)
        copyto!(thermal.adiabatic, Ah)
        @parallel (@idx ni) JR2K.update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., to_device(Hh), to_device(SHh), thermal.adiabatic,
            rheo, phase, to_device(dτh), JustRelax.DirichletBoundaryCondition(), inv(dt), _di_v, args,
        )
        Tnew = Array(thermal.T)
        expected = similar(Th, nx, ny)
        for j in 1:ny, i in 1:nx
            divq = (qx0[i + 1, j] - qx0[i, j]) / dx + (qy0[i, j + 1] - qy0[i, j]) / dy
            num = dτh[i, j] * (-divq + Toldh[i + 1, j + 1] * ρCp_ref[i, j] / dt + Hsrc[i, j] + SHh[i, j]) + Th[i + 1, j + 1]
            expected[i, j] = num / (1 + dτh[i, j] * ρCp_ref[i, j] / dt)
        end
        @test Tnew[2:(end - 1), 2:(end - 1)] ≈ expected

        thermal2 = ThermalArrays(backend, ni)
        copyto!(thermal2.T, Th)
        copyto!(thermal2.Told, Toldh)
        copyto!(thermal2.qTx, qx0)
        copyto!(thermal2.qTy, qy0)
        copyto!(thermal2.H, Hh)
        copyto!(thermal2.shear_heating, SHh)
        copyto!(thermal2.adiabatic, Ah)
        JR2K.update_T(
            nothing, nothing, thermal2, rheo, phase, (; dτ_ρ = to_device(dτh)),
            JustRelax.DirichletBoundaryCondition(), inv(dt), _di_v, ni, args,
        )
        @test Array(thermal2.T) ≈ Tnew

        copyto!(thermal.T, Th)
        copyto!(thermal.qTx2, qx0)
        copyto!(thermal.qTy2, qy0)
        @parallel (@idx ni) JR2K.check_res!(
            thermal.ResT, thermal.T, thermal.Told, @qT2(thermal)..., to_device(Hh), to_device(SHh),
            thermal.adiabatic, rheo, phase, JustRelax.DirichletBoundaryCondition(), inv(dt), _di_v, args,
        )
        @test Array(thermal.ResT) ≈ reference_residual(Th, Toldh, qx0, qy0, Hsrc, SHh, ρCp_ref, dt, dx, dy)
    end

    @testset "pseudo-transient coefficients" begin
        # Vpdτ = CFL min(Δx), Re = π + √(π² + ρCp L² / (K Δt)), θr_dτ = L / (Vpdτ Re), dτ_ρ = Vpdτ L / (K Re)
        CFL, L, dtc = 0.5, max(nx * dx, ny * dy), 0.7
        Vpdτ = CFL * min(dx, dy)
        Re(ρCp, K) = π + sqrt(π^2 + ρCp * L^2 / (K * dtc))
        θr(ρCp, K) = L / (Vpdτ * Re(ρCp, K))
        dτρ(ρCp, K) = Vpdτ * L / (K * Re(ρCp, K))
        args_T = (; T = to_device(Th), P = args.P)
        # ρCp at the cell temperature, read past the ghost ring of T
        ρCp_mix = [φ1[I] * ρ(1, Th[I + CartesianIndex(1, 1)], Ph[I]) * Cp[1] + (1 - φ1[I]) * ρ(2, Th[I + CartesianIndex(1, 1)], Ph[I]) * Cp[2] for I in CartesianIndices(ni)]
        ρCp_1 = [ρ(1, Th[I + CartesianIndex(1, 1)], Ph[I]) * Cp[1] for I in CartesianIndices(ni)]
        Kmix = φ1 .* k[1] .+ (1 .- φ1) .* k[2]

        pt = PTThermalCoeffs(backend, rheology, phase_ratios, args_T, dtc, ni, (dx, dy), (nx * dx, ny * dy); CFL)
        @test pt.Vpdτ ≈ Vpdτ
        @test Array(pt.θr_dτ) ≈ θr.(ρCp_mix, Kmix)
        @test Array(pt.dτ_ρ) ≈ dτρ.(ρCp_mix, Kmix)

        pt1 = PTThermalCoeffs(backend, rheology[1], args_T, dtc, ni, (dx, dy), (nx * dx, ny * dy); CFL)
        @test Array(pt1.θr_dτ) ≈ θr.(ρCp_1, k[1])
        @test Array(pt1.dτ_ρ) ≈ dτρ.(ρCp_1, k[1])

        K_arr, ρCp_arr = to_device(Kh), to_device(ρCph)
        pt_arr = PTThermalCoeffs(backend, K_arr, ρCp_arr, dtc, (dx, dy), (nx * dx, ny * dy); CFL)
        @test Array(pt_arr.θr_dτ) ≈ θr.(ρCph, Kh)
        @test Array(pt_arr.dτ_ρ) ≈ dτρ.(ρCph, Kh)

        # refreshing at a new time step reproduces a fresh construction
        dt2 = 0.2
        Re2(ρCp, K) = π + sqrt(π^2 + ρCp * L^2 / (K * dt2))
        JR2K.update_thermal_coeffs!(pt, rheology, phase_ratios, args_T, dt2)
        @test Array(pt.dτ_ρ) ≈ Vpdτ * L ./ (Kmix .* Re2.(ρCp_mix, Kmix))
        JR2K.update_thermal_coeffs!(pt1, rheology[1], args_T, dt2)
        @test Array(pt1.θr_dτ) ≈ L ./ (Vpdτ .* Re2.(ρCp_1, k[1]))
        JR2K.update_thermal_coeffs!(pt1, rheology[1], args_T, dtc)
        JR2K.update_thermal_coeffs!(pt1, rheology[1], nothing, args_T, dt2)
        @test Array(pt1.θr_dτ) ≈ L ./ (Vpdτ .* Re2.(ρCp_1, k[1]))
    end

    @testset "adiabatic heating term α (P - P0) / Δt" begin
        thermal = ThermalArrays(backend, ni)
        P0h = Ph .- [0.1 * cos(i * j) for i in 1:nx, j in 1:ny]
        stokes_P = (; P = to_device(Ph), P0 = to_device(P0h))
        JR2K.adiabatic_heating!(thermal, stokes_P, rheology, phase_ratios.center, inv(dt))
        α_mix = φ1 .* α[1] .+ (1 .- φ1) .* α[2]
        @test Array(thermal.adiabatic) ≈ (Ph .- P0h) .* α_mix ./ dt
        # without Stokes arrays there is no pressure change to heat from
        copyto!(thermal.adiabatic, Ah)
        JR2K.adiabatic_heating!(thermal, nothing, rheology, phase_ratios.center, inv(dt))
        @test Array(thermal.adiabatic) == Ah
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
        τxx = [sin(0.7i + 0.2j) for i in 1:nx, j in 1:ny]
        τyy = [cos(0.4i - 0.5j) for i in 1:nx, j in 1:ny]
        τxy = [0.5 * sin(i - 0.6j) for i in 1:nx, j in 1:ny]
        τo = (0.8 .* τxx .+ 0.1, 0.6 .* τyy, 0.9 .* τxy .- 0.05)
        εxx = [0.6 * sin(0.3i + 0.9j) for i in 1:nx, j in 1:ny]
        εyy = -εxx .+ 0.1
        εxy_v = [0.4 * cos(0.7i + 0.35j) for i in 1:(nx + 1), j in 1:(ny + 1)]
        foreach(copyto!, (stokes.τ.xx, stokes.τ.yy, stokes.τ.xy_c), (τxx, τyy, τxy))
        foreach(copyto!, (stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy_c), τo)
        foreach(copyto!, (stokes.ε.xx, stokes.ε.yy, stokes.ε.xy), (εxx, εyy, εxy_v))

        # the vertex shear strain rate is averaged onto the cell centers
        εxy_c = [0.25 * (εxy_v[i, j] + εxy_v[i + 1, j] + εxy_v[i, j + 1] + εxy_v[i + 1, j + 1]) for i in 1:nx, j in 1:ny]
        function work(Gc, i, j)
            el(τn, τ0) = 0.5 * (τn - τ0) / (Gc * dt_sh)
            return τxx[i, j] * (εxx[i, j] - el(τxx[i, j], τo[1][i, j])) +
                τyy[i, j] * (εyy[i, j] - el(τyy[i, j], τo[2][i, j])) +
                2 * τxy[i, j] * (εxy_c[i, j] - el(τxy[i, j], τo[3][i, j]))
        end

        thermal = ThermalArrays(backend, ni)
        compute_shear_heating!(thermal, stokes, rheo_sh[1], dt_sh)
        expected = [max(0.0, Χ[1] * work(G[1], i, j)) for i in 1:nx, j in 1:ny]
        @test any(iszero, expected) && any(>(0), expected)
        @test Array(thermal.shear_heating) ≈ expected

        compute_shear_heating!(thermal, stokes, phase_ratios, rheo_sh, dt_sh)
        expected_mix = [
            begin
                    w = work(φ1[i, j] * G[1] + (1 - φ1[i, j]) * G[2], i, j)
                    max(0.0, (φ1[i, j] * Χ[1] + (1 - φ1[i, j]) * Χ[2]) * w)
                end for i in 1:nx, j in 1:ny
        ]
        @test Array(thermal.shear_heating) ≈ expected_mix
    end

    @testset "flux on a non-uniform grid uses the local center spacing" begin
        # T linear in x and a uniform conductor: the exact face flux is -k0 * b everywhere.
        xv = [0.0, 0.2, 0.5, 0.6, 1.0, 1.3, 1.45, 1.8]
        yv = [0.0, 0.1, 0.35, 0.45, 0.7, 0.8, 1.0]
        grid_nu = Geometry(PTArray(backend), xv, yv)
        xc = 0.5 .* (xv[1:(end - 1)] .+ xv[2:end])
        yc = 0.5 .* (yv[1:(end - 1)] .+ yv[2:end])
        xgn = [2xv[1] - xc[1]; xc; 2xv[end] - xc[end]]
        ygn = [2yv[1] - yc[1]; yc; 2yv[end] - yc[end]]
        b, c, k0 = 0.7, -1.3, 2.5
        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, [1.0 + b * x + c * y for x in xgn, y in ygn])
        @parallel (@idx ni .+ 1) JR2K.compute_flux!(
            @qT(thermal)..., @qT2(thermal)..., thermal.T, to_device(fill(k0, ni)),
            to_device(fill(0.5, ni)), grid_nu._di, no_flux_bc,
        )
        # Boundary ghost temperatures are reflected about the physical boundary, as in
        # thermal_bcs!, so their distance to the adjacent center is the boundary cell width.
        @test all(Array(thermal.qTx2) .≈ -k0 * b)
        @test all(Array(thermal.qTy2) .≈ -k0 * c)
    end

    @testset "update_T! and check_res! divide by the cell (vertex-to-vertex) spacing on a non-uniform grid" begin
        xv = [0.0, 0.2, 0.5, 0.6, 1.0, 1.3, 1.45, 1.8]
        yv = [0.0, 0.1, 0.35, 0.45, 0.7, 0.8, 1.0]
        grid_nu = Geometry(PTArray(backend), xv, yv)
        dxv = Array(grid_nu.di.vertex[1]) # width of each thermal cell, one entry per cell
        dyv = Array(grid_nu.di.vertex[2])

        qxh = [0.05 * sin(i + 2j) for i in 1:(nx + 1), j in 1:ny]
        qyh = [0.05 * cos(2i - j) for i in 1:nx, j in 1:(ny + 1)]
        divq(i, j) = (qxh[i + 1, j] - qxh[i, j]) / dxv[i] + (qyh[i, j + 1] - qyh[i, j]) / dyv[j]

        thermal = ThermalArrays(backend, ni)
        copyto!(thermal.T, Th)
        copyto!(thermal.Told, Toldh)
        copyto!(thermal.qTx, qxh)
        copyto!(thermal.qTy, qyh)
        H, SH, ρCp, dτρ = to_device(Hh), to_device(SHh), to_device(ρCph), to_device(dτh)
        @parallel (@idx ni) JR2K.update_T!(
            thermal.T, thermal.Told, @qT(thermal)..., H, SH, ρCp, dτρ,
            JustRelax.DirichletBoundaryCondition(), inv(dt), grid_nu._di.vertex,
        )
        Tnew = Array(thermal.T)
        expected = similar(Th, nx, ny)
        for j in 1:ny, i in 1:nx
            num = dτh[i, j] * (-divq(i, j) + Toldh[i + 1, j + 1] * ρCph[i, j] / dt + Hh[i, j] + SHh[i, j]) + Th[i + 1, j + 1]
            expected[i, j] = num / (1 + dτh[i, j] * ρCph[i, j] / dt)
        end
        @test Tnew[2:(end - 1), 2:(end - 1)] ≈ expected

        copyto!(thermal.qTx2, qxh)
        copyto!(thermal.qTy2, qyh)
        @parallel (@idx ni) JR2K.check_res!(
            thermal.ResT, thermal.T, thermal.Told, @qT2(thermal)..., H, SH, ρCp,
            JustRelax.DirichletBoundaryCondition(), inv(dt), grid_nu._di.vertex,
        )
        R_expected = similar(Th, nx, ny)
        for j in 1:ny, i in 1:nx
            R_expected[i, j] = -ρCph[i, j] * (Tnew[i + 1, j + 1] - Toldh[i + 1, j + 1]) / dt - divq(i, j) + Hh[i, j] + SHh[i, j]
        end
        @test Array(thermal.ResT) ≈ R_expected
    end
end
