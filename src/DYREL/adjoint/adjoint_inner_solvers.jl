# Inner velocity solvers of the adjoint Powell–Hestenes passes.
#
# Each pass fixes λP and solves the Schur-corrected momentum system M λV = f for λV, where
# M = A_vvᵀ + A_pvᵀ γ A_vpᵀ is applied by one reverse sweep. Two solvers share one interface:
#
#   inner_solve!(solver, problem, err, ϵ_vel, iter) -> (err, iter)
#   after_pressure_update!(solver)
#
# `problem` holds what both need: `residual!`, `momentum_error`, the unknowns `λVi`, the residual
# `R` (interior rows), the diagonal `D`, the budgets `iterMax`, `total_iterMax`, and `nout`,
# `verbose`. `iter` counts reverse sweeps.

"""
    adjoint_inner_solver(name, deflation, stokes_ad, dyrel, maskV, maskRi, periodic,
                         constitutive_scratch, ϕ, ni)

Validate `inner_solver` and `deflation` and build the inner solver: [`AdjointBiCGStab`](@ref)
for `:bicgstab`, [`AdjointDR`](@ref) for `:dr`.
"""
function adjoint_inner_solver(
        name, deflation, stokes_ad, dyrel, maskV, maskRi, periodic, constitutive_scratch, ϕ, ni,
    )
    deflation isa Integer && deflation ≥ 0 || throw(
        ArgumentError("deflation must be a nonnegative integer, the number of recycled vectors")
    )
    if name === :bicgstab
        return AdjointBiCGStab(adjoint_bicgstab_workspace(stokes_ad, (dyrel.Dx, dyrel.Dy), deflation))
    elseif name === :dr
        iszero(deflation) || throw(ArgumentError("deflation requires inner_solver = :bicgstab"))
        return AdjointDR(stokes_ad, dyrel, maskV, maskRi, periodic, constitutive_scratch, ϕ, ni)
    end
    throw(ArgumentError("inner_solver must be :dr or :bicgstab"))
end

# The Schur-complement (penalty) term of the adjoint momentum residual: r_V += A_pvᵀ γ r_P, with
# A_pv = −ϕ div the derivative of the continuity residual, so A_pvᵀ is the ϕ-weighted gradient.
# It is the transpose of the forward penalty; rows eliminated by `ϕ` are set to zero.
@parallel_indices (i, j) function adjoint_schur_correction!(
        Rx, Ry, P, γ_eff, Dx, Dy, ϕ::JustRelax.RockRatio, _di_center,
    )
    @inbounds begin
        if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
            if isvalid_vx(ϕ, i + 1, j)
                iE = wrap_next(i, size(γ_eff, 1))
                Rx[i + 1, j + 1] -= (
                    ϕ.center[i, j] * γ_eff[i, j] * P[i, j] -
                        ϕ.center[iE, j] * γ_eff[iE, j] * P[iE, j]
                ) * @dx(_di_center, i)
            else
                Rx[i + 1, j + 1] = 0.0
            end
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            if isvalid_vy(ϕ, i, j + 1)
                jN = wrap_next(j, size(γ_eff, 2))
                Ry[i + 1, j + 1] -= (
                    ϕ.center[i, j] * γ_eff[i, j] * P[i, j] -
                        ϕ.center[i, jN] * γ_eff[i, jN] * P[i, jN]
                ) * @dy(_di_center, j)
            else
                Ry[i + 1, j + 1] = 0.0
            end
        end
    end
    return nothing
end

#### Damped dynamic relaxation

"""
    AdjointDR

Damped dynamic relaxation of the adjoint velocity, the iteration of the forward solver on the
adjoint operator. Two differences from the forward: the damping is re-estimated from the adjoint
operator (`update_adjoint_damping!`), and for nonlinear viscosity the pseudo-time step is
shortened by the amplification `max(1, |1/(1 − F)|)` of the implicit viscosity tangent.
"""
struct AdjointDR{V, R, D, M, A, P, N}
    λV::V                       # adjoint velocity arrays, ghost rows included
    R::R                        # adjoint momentum residual arrays, ghost rows included
    dyrel::D                    # the adjoint's own DYREL state
    masks::M                    # momentum rows of the damping estimate
    stored::Base.RefValue{Int}  # iteration of the stored step, 0 if none
    amplification::A            # tangent amplification at centers, `nothing` for linear viscosity
    ϕ::P
    ni::N
end

function AdjointDR(stokes_ad, dyrel, maskV, maskRi, periodic, constitutive_scratch, ϕ, ni)
    # the norm rows, on arrays of the full momentum-row size
    masks = map(maskV, maskRi) do mask, trimmed
        rows = fill!(similar(mask), false)
        norm_rows(rows, periodic) .= trimmed
        rows
    end
    amplification = if isnothing(constitutive_scratch)
        nothing
    else
        F = constitutive_scratch.feedback
        @. max(one(eltype(F)), abs(inv(1 - F)))
    end
    dr = AdjointDR(
        (stokes_ad.λV.Vx, stokes_ad.λV.Vy), (stokes_ad.V.Vx, stokes_ad.V.Vy), dyrel, masks,
        Ref(0), amplification, ϕ, ni,
    )
    isnothing(amplification) || update_steps!(dr)
    return dr
end

function update_steps!(dr::AdjointDR)
    if isnothing(dr.amplification)
        update_dτV_α_β!(dr.dyrel)
    else
        update_variational_adjoint_dτV_α_β!(dr.dyrel, dr.amplification)
    end
    return nothing
end

function inner_solve!(dr::AdjointDR, problem, err, ϵ_vel, iter)
    (; λV, R, dyrel, ϕ, ni) = dr
    itPT = 0
    while err > ϵ_vel && itPT < problem.iterMax && iter < problem.total_iterMax
        itPT += 1
        iter += 1
        # with the seed: a pressure objective enters through the Schur term and the λP update
        problem.residual!()
        @parallel (@idx ni) adjoint_damped_update!(
            λV..., dyrel.dVxdτ, dyrel.dVydτ, R..., dyrel.Dx, dyrel.Dy,
            dyrel.αVx, dyrel.αVy, dyrel.βVx, dyrel.βVy, dyrel.dτVx, dyrel.dτVy, ϕ,
        )
        update_adjoint_damping!(dr, problem.R, iter, problem.nout)
        if iszero(iter % problem.nout)
            err = problem.momentum_error()
            isnan(err) && error("NaN detected in inner loop")
            problem.verbose && @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
        end
    end
    return err, iter
end

# a step stored before the pressure update would mix the pressure change into the λmin estimate
after_pressure_update!(dr::AdjointDR) = (dr.stored[] = 0; nothing)

# One damped pseudo-time step of the adjoint velocity from the Schur-corrected residual. Rows
# eliminated by `ϕ` are held at zero, as in the forward `compute_DR_residual_update_V!`.
@parallel_indices (i, j) function adjoint_damped_update!(
        Vx, Vy, dVxdτ, dVydτ, Rx, Ry, Dx, Dy, αVx, αVy, βVx, βVy, dτVx, dτVy,
        ϕ::JustRelax.RockRatio,
    )
    @inbounds begin
        if i ≤ size(Dx, 1) && j ≤ size(Dx, 2)
            if isvalid_vx(ϕ, i + 1, j)
                dVx_new, ΔVx = damped_update_V(dVxdτ[i, j], Rx[i + 1, j + 1] / Dx[i, j], αVx[i, j], βVx[i, j], dτVx[i, j])
                dVxdτ[i, j] = dVx_new
                Vx[i + 1, j + 1] += ΔVx
            else
                dVxdτ[i, j] = 0.0
                Vx[i + 1, j + 1] = 0.0
            end
        end
        if i ≤ size(Dy, 1) && j ≤ size(Dy, 2)
            if isvalid_vy(ϕ, i, j + 1)
                dVy_new, ΔVy = damped_update_V(dVydτ[i, j], Ry[i + 1, j + 1] / Dy[i, j], αVy[i, j], βVy[i, j], dτVy[i, j])
                dVydτ[i, j] = dVy_new
                Vy[i + 1, j + 1] += ΔVy
            else
                dVydτ[i, j] = 0.0
                Vy[i + 1, j + 1] = 0.0
            end
        end
    end
    return nothing
end

@inline preconditioned_residual(R, D) = iszero(D) ? zero(R) : R / D

# The adjoint iterates on the consistent tangent of the converged forward state, whose smallest
# eigenvalue can lie far from the one the forward damping was tuned to (a yielding plastic
# tangent is much softer). Re-estimate the damping every `nout` iterations as the forward solver
# does: λmin is the Rayleigh quotient of the last velocity step against the change of the
# preconditioned residual it caused. The residual of a step only exists in the next iteration,
# so the step and its residual are stored one iteration before each estimate. `residuals` are
# the adjoint momentum residuals of the current iterate.
function update_adjoint_damping!(dr::AdjointDR, residuals, iter, nout)
    (; dyrel, masks, stored, ni) = dr
    if iszero((iter + 1) % nout)
        @parallel (@idx ni) compute_dV!(
            (dyrel.dVx, dyrel.dVy), (dyrel.dVxdτ, dyrel.dVydτ),
            (dyrel.βVx, dyrel.βVy), (dyrel.dτVx, dyrel.dτVy),
        )
        @. dyrel.Rx0 = preconditioned_residual(residuals[1], dyrel.Dx)
        @. dyrel.Ry0 = preconditioned_residual(residuals[2], dyrel.Dy)
        stored[] = iter
    elseif iszero(iter % nout) && stored[] == iter - 1
        λminV = adjoint_λminV(
            (dyrel.dVx, dyrel.dVy), residuals, (dyrel.Dx, dyrel.Dy), (dyrel.Rx0, dyrel.Ry0), masks,
        )
        @parallel (@idx ni) update_cV!((dyrel.cVx, dyrel.cVy), 2 * √(λminV) * dyrel.c_fact)
        update_steps!(dr)
    end
    return nothing
end

function adjoint_λminV(dV, residuals, D, residuals0, masks)
    numerator = sum(
        ntuple(Val(2)) do d
            sum_mpi(
                (m, dv, r, Dd, r0) -> m ? dv * (preconditioned_residual(r, Dd) - r0) : zero(dv),
                masks[d], dV[d], residuals[d], D[d], residuals0[d],
            )
        end
    )
    denominator = sum(
        ntuple(d -> sum_mpi((m, dv) -> m ? abs2(dv) : zero(abs2(dv)), masks[d], dV[d]), Val(2))
    )
    return rayleigh_quotient(numerator, denominator)
end

@inline function set_adjoint_dyrel_parameters!(
        dτV, βV, αV, cV, λmaxV, amplification, CFL, i, j,
    )
    dτV_ij = dτV[i, j] = 2 / sqrt(λmaxV[i, j] * amplification) * CFL
    cdt = cV[i, j] * dτV_ij
    inv_den = inv(2 + cdt)
    βV[i, j] = 2 * dτV_ij * inv_den
    αV[i, j] = (2 - cdt) * inv_den
    return nothing
end

# The variational stress kernel obtains vertex viscosity by harmonic interpolation from centers.
# Use every center contributing to the two vertex stresses in a momentum row when choosing its
# local tangent amplification.
function update_variational_adjoint_dτV_α_β!(dyrel, amplification_center)
    @parallel (@idx size(amplification_center)) _update_variational_adjoint_dτV_α_β!(
        dyrel.dτVx, dyrel.dτVy, dyrel.βVx, dyrel.βVy, dyrel.αVx, dyrel.αVy,
        dyrel.cVx, dyrel.cVy, dyrel.λmaxVx, dyrel.λmaxVy,
        amplification_center, dyrel.CFL,
    )
    return nothing
end

@parallel_indices (i, j) function _update_variational_adjoint_dτV_α_β!(
        dτVx, dτVy, βVx, βVy, αVx, αVy, cVx, cVy, λmaxVx, λmaxVy,
        amplification_center, CFL,
    )
    @inbounds begin
        nx, ny = size(amplification_center)
        periodic_x = size(dτVx, 1) == nx
        periodic_y = size(dτVy, 2) == ny
        iW = i == 1 ? (periodic_x ? nx : 1) : i - 1
        iE = i == nx ? (periodic_x ? 1 : nx) : i + 1
        jS = j == 1 ? (periodic_y ? ny : 1) : j - 1
        jN = j == ny ? (periodic_y ? 1 : ny) : j + 1

        if i ≤ size(dτVx, 1) && j ≤ size(dτVx, 2)
            amplification = max(
                amplification_center[i, jS], amplification_center[iE, jS],
                amplification_center[i, j], amplification_center[iE, j],
                amplification_center[i, jN], amplification_center[iE, jN],
            )
            set_adjoint_dyrel_parameters!(
                dτVx, βVx, αVx, cVx, λmaxVx, amplification, CFL, i, j,
            )
        end
        if i ≤ size(dτVy, 1) && j ≤ size(dτVy, 2)
            amplification = max(
                amplification_center[iW, j], amplification_center[i, j],
                amplification_center[iE, j], amplification_center[iW, jN],
                amplification_center[i, jN], amplification_center[iE, jN],
            )
            set_adjoint_dyrel_parameters!(
                dτVy, βVy, αVy, cVy, λmaxVy, amplification, CFL, i, j,
            )
        end
    end
    return nothing
end

#### BiCGStab with optional recycled deflation

"""
    AdjointBiCGStab

BiCGStab on the symmetrically scaled adjoint velocity system `D^-½ M D^-½`, optionally deflated
by recycled solution increments (`deflation = k`). The adjoint is linear, so it needs no damping
or eigenvalue estimates, and BiCGStab does not need `M` to be symmetric or to have real
eigenvalues (non-associated plasticity, the tangent of a nonlinear viscosity).
"""
struct AdjointBiCGStab{W}
    work::W
end

function inner_solve!(solver::AdjointBiCGStab, problem, err, ϵ_vel, iter)
    (; residual!, momentum_error, λVi, R, D) = problem
    # BiCGStab starts from the residual of the current λ
    residual!()
    err, iter = adjoint_bicgstab!(
        residual!, momentum_error, solver.work, λVi, R, D, err, ϵ_vel, iter,
        problem.iterMax, problem.total_iterMax; problem.verbose, problem.nout,
    )
    # the last sweep was of a search direction: refresh the residual of λ, whose continuity part
    # the λP update reads
    residual!()
    recycle_increment!(solver.work.deflation, λVi, R, solver.work.isqrtD)
    return err, iter
end

after_pressure_update!(::AdjointBiCGStab) = nothing

adjoint_dot(a, b) = sum(map((a, b) -> sum_mpi((x, y) -> x * y, a, b), a, b))

# Recycled deflation of the Krylov inner solver. A few very slow modes of the velocity system,
# e.g. the rigid motions of a stiff body enclosed by a weak layer, cost the diagonally
# preconditioned iteration most of its sweeps. What an inexact solve leaves behind is dominated
# by those modes, so the solution increments of the pressure passes span them: BiCGStab then
# iterates on P A with P = I − AW E Wᵀ and E = (Wᵀ A W)⁻¹, where W holds the last `kmax`
# increments (orthonormalized) of the scaled system.
mutable struct AdjointDeflation{A, V}
    const W::A              # per velocity component, the vectors in the slices of one array
    const AW::A             # their images: differences of true residuals, so they cost no sweep
    E::Matrix{Float64}      # (Wᵀ A W)⁻¹ on the filled slots, zero elsewhere
    const slots::Vector{Int}    # filled slots, oldest first
    const x0::V             # iterate and scaled residual at the start of the pass
    const r0::V
    const scratch::V
end

function AdjointDeflation(vectors, kmax)
    slices() = map(A -> fill!(similar(A, size(A)..., kmax), 0), vectors())
    return AdjointDeflation(slices(), slices(), zeros(kmax, kmax), Int[], vectors(), vectors(), vectors())
end

# one column per vector, so that all vectors are handled by a single matrix product
@inline columns(A) = reshape(A, :, size(A, ndims(A)))

# Wᵀ y for all slots at once, summed over the velocity components and the MPI ranks
function deflation_dots(W, y)
    c = sum(map((W, y) -> Array(columns(W)' * vec(y)), W, y))
    return MPI.Allreduce(c, MPI.SUM, MPI.COMM_WORLD)
end

on_backend(c, A) = copyto!(similar(A, length(c)), c)

# y ← P y. Returns c = E Wᵀ y, which also gives the iterate correction Q x = x − W c for y = A x.
deflate!(y, ::Nothing) = nothing
function deflate!(y, deflation::AdjointDeflation)
    isempty(deflation.slots) && return nothing
    c = on_backend(deflation.E * deflation_dots(deflation.W, y), first(y))
    for d in eachindex(y)
        mul!(vec(y[d]), columns(deflation.AW[d]), c, -1.0, 1.0)
    end
    return c
end

# λ += s W c, in the unscaled unknowns
correct_iterate!(λVi, deflation, ::Nothing, s, isqrtD) = nothing
function correct_iterate!(λVi, deflation, c, s, isqrtD)
    for d in eachindex(λVi)
        mul!(vec(deflation.scratch[d]), columns(deflation.W[d]), c)
        @. λVi[d] += s * deflation.scratch[d] * isqrtD[d]
    end
    return nothing
end

# Remember where the pass starts and solve for the deflated modes: λ += W E Wᵀ r, r ← P r.
start_deflated_pass!(::Nothing, λVi, r, isqrtD) = nothing
function start_deflated_pass!(deflation, λVi, r, isqrtD)
    for d in eachindex(r)
        deflation.x0[d] .= λVi[d]
        deflation.r0[d] .= r[d]
    end
    correct_iterate!(λVi, deflation, deflate!(r, deflation), 1.0, isqrtD)
    return nothing
end

# Add the increment of the finished pass to the deflation space. `R` is the true residual of
# the new `λVi`, for the same λP as at the start of the pass.
recycle_increment!(::Nothing, λVi, R, isqrtD) = nothing
function recycle_increment!(deflation, λVi, R, isqrtD)
    (; W, AW, slots) = deflation
    kmax = size(deflation.E, 1)
    Δ, AΔ = deflation.x0, deflation.r0
    for d in eachindex(Δ)
        @. Δ[d] = ifelse(isqrtD[d] > 0, (λVi[d] - Δ[d]) / isqrtD[d], 0.0)
        @. AΔ[d] -= R[d] * isqrtD[d]
    end
    scale = sqrt(adjoint_dot(Δ, Δ))
    # orthogonalize against the vectors that stay; the oldest makes room when all slots are full
    kept = length(slots) == kmax ? slots[2:end] : slots
    h = zeros(kmax)
    h[kept] = deflation_dots(W, Δ)[kept]
    h = on_backend(h, first(Δ))
    for d in eachindex(Δ)
        mul!(vec(Δ[d]), columns(W[d]), h, -1.0, 1.0)
        mul!(vec(AΔ[d]), columns(AW[d]), h, -1.0, 1.0)
    end
    nrm = sqrt(adjoint_dot(Δ, Δ))
    # an increment that is already in the space adds nothing
    isfinite(nrm) && nrm > 1.0e-8 * scale || return nothing
    slot = length(slots) == kmax ? popfirst!(slots) : first(setdiff(1:kmax, slots))
    for d in eachindex(Δ)
        selectdim(W[d], ndims(W[d]), slot) .= Δ[d] ./ nrm
        selectdim(AW[d], ndims(AW[d]), slot) .= AΔ[d] ./ nrm
    end
    push!(slots, slot)
    G = sum(map((W, AW) -> Array(columns(W)' * columns(AW)), W, AW))
    G = MPI.Allreduce(G, MPI.SUM, MPI.COMM_WORLD)
    factorization = lu(G[slots, slots]; check = false)
    fill!(deflation.E, 0)
    if issuccess(factorization)
        deflation.E[slots, slots] = inv(factorization)
    else
        empty!(slots)
    end
    return nothing
end

# Work arrays of `adjoint_bicgstab!`, allocated once per adjoint solve on the backend of `D`.
# `deflation` is the number of recycled deflation vectors, 0 for none.
function adjoint_bicgstab_workspace(stokes_ad, D, deflation = 0)
    direction = (zero(stokes_ad.λV.Vx), zero(stokes_ad.λV.Vy))
    direction_i = map(
        (A, D) -> @view(A[2:(size(D, 1) + 1), 2:(size(D, 2) + 1)]), direction, D,
    )
    vectors() = map(zero, D)
    return (;
        direction, direction_i, zeroP = zero(stokes_ad.λP), isqrtD = vectors(),
        r = vectors(), r̂ = vectors(), p = vectors(), v = vectors(), t = vectors(),
        deflation = iszero(deflation) ? nothing : AdjointDeflation(vectors, deflation),
    )
end

"""
    adjoint_bicgstab!(
        residual!, momentum_error, work, λVi, R, D, err, ϵ_vel, iter, iterMax,
        total_iterMax; verbose = false, nout = 1,
    )

BiCGStab iterations of one inner velocity solve (see [`AdjointBiCGStab`](@ref)), on the
symmetrically scaled system `D^-½ M D^-½`, with `D` the DYREL diagonal, deflated by the
recycled vectors of `work.deflation` if there are any.

`residual!(λV, λP; seed)` writes the Schur-corrected momentum residual of `(λV, λP)` to `R`,
with or without the objective seed; on entry `R` holds that of the current `λVi`.
`momentum_error()` returns the relative momentum residual of `R`, the stopping measure. Returns
`(err, iter)`; `iter` counts reverse sweeps, two per BiCGStab iteration. After a breakdown the
solve warns, returns early and restarts from the true residual in the next pass.
"""
function adjoint_bicgstab!(
        residual!, momentum_error, work, λVi, R, D, err, ϵ_vel, iter, iterMax,
        total_iterMax; verbose = false, nout = 1,
    )
    (; direction, direction_i, zeroP, isqrtD, r, r̂, p, v, t, deflation) = work
    components = eachindex(r)
    dot = adjoint_dot
    # out = P D^-½ A D^-½ x, by one reverse sweep without the objective seed: R = −A·direction.
    # Returns the deflation coefficients of the sweep, `nothing` without deflation.
    function apply!(out, x)
        for d in components
            @. direction_i[d] = x[d] * isqrtD[d]
        end
        residual!(direction, zeroP; seed = false)
        for d in components
            @. out[d] = -R[d] * isqrtD[d]
        end
        return deflate!(out, deflation)
    end
    for d in components
        # rows without a diagonal are not unknowns
        @. isqrtD[d] = ifelse(D[d] > 0, inv(sqrt(D[d])), 0.0)
        @. r[d] = R[d] * isqrtD[d]
    end
    start_deflated_pass!(deflation, λVi, r, isqrtD)
    for d in components
        copyto!(r̂[d], r[d])
        fill!(p[d], 0)
        fill!(v[d], 0)
    end
    ρ = α = ω = one(eltype(r[1]))
    itPT = 0
    while err > ϵ_vel && itPT < iterMax && iter < total_iterMax
        itPT += 2
        iter += 2
        ρ_new = dot(r̂, r)
        β = (ρ_new / ρ) * (α / ω)
        ρ = ρ_new
        for d in components
            @. p[d] = r[d] + β * (p[d] - ω * v[d])
        end
        c = apply!(v, p)
        α = ρ / dot(r̂, v)
        if !isfinite(α)
            @warn "adjoint BiCGStab broke down (α = $α); restarting from the true residual" maxlog = 5
            break
        end
        for d in components
            @. λVi[d] += α * p[d] * isqrtD[d]
            @. r[d] -= α * v[d]
        end
        correct_iterate!(λVi, deflation, c, -α, isqrtD)
        c = apply!(t, r)
        ω = dot(t, r) / dot(t, t)
        if !isfinite(ω) || iszero(ω)
            @warn "adjoint BiCGStab broke down (ω = $ω); restarting from the true residual" maxlog = 5
            break
        end
        for d in components
            @. λVi[d] += ω * r[d] * isqrtD[d]
            @. r[d] -= ω * t[d]
            # the unscaled residual, for the norm of the stopping criterion
            @. R[d] = r[d] * D[d] * isqrtD[d]
        end
        correct_iterate!(λVi, deflation, c, -ω, isqrtD)
        err = momentum_error()
        verbose && iszero(itPT % 2nout) && @printf("it = %d, iter = %d, err = %1.3e \n", itPT, iter, err)
    end
    return err, iter
end
