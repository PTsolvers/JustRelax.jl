import JustRelax.JustRelax2D: get_shear_modulus, compute_∇V!, update_ρg!, compute_strain_rate!, cache_tensors, get_bulk_modulus, clamped_indices, av_clamped

include("pressure_kernels.jl")
include("Gershgorin.jl")

function _solve!(
        stokes::JustRelax.StokesArrays,
        pt_stokes,
        di::NTuple{2, T},
        flow_bcs::AbstractFlowBoundaryConditions,
        ρg,
        phase_ratios::JustPIC.PhaseRatios,
        rheology,
        args,
        dt,
        igg::IGG;
        strain_increment = false,
        viscosity_cutoff = (-Inf, Inf),
        viscosity_relaxation = 1.0e-2,
        λ_relaxation = 0.2,
        iterMax = 50.0e3,
        iterMin = 1.0e2,
        free_surface = false,
        nout = 500,
        b_width = (4, 4, 0),
        verbose = true,
        kwargs...,
    ) where {T}

    # unpack

    _di = inv.(di)
    _dt = inv.(dt)
    (; ϵ_rel, ϵ_abs, r, θ_dτ, ηdτ) = pt_stokes
    (; η, η_vep) = stokes.viscosity
    ni = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)

    # errors
    err_it1 = 1.0
    err = 1.0
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]
    # sizehint!(norm_Rx, Int(iterMax))
    # sizehint!(norm_Ry, Int(iterMax))
    # sizehint!(norm_∇V, Int(iterMax))
    # sizehint!(err_evo1, Int(iterMax))
    # sizehint!(err_evo2, Int(iterMax))

    # solver loop
    @copy stokes.P0 stokes.P
    wtime0 = 0.0
    relλ = λ_relaxation
    θ = deepcopy(stokes.P)
    λ = @zeros(ni...)
    λv = @zeros(ni .+ 1...)
    η0 = deepcopy(η)
    do_visc = true

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0
    end

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg, phase_ratios, rheology, args)
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)
    displacement2velocity!(stokes, dt, flow_bcs)

    # while iter ≤ iterMax
    #     iterMin < iter && ((err / err_it1) < ϵ_rel || err < ϵ_abs) && break

    for itPh in 1:1000

        wtime0 += @elapsed begin

            @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), _di)

            if strain_increment
                @parallel (@idx ni) compute_∇V!(stokes.∇U, @displacement(stokes), _di)
            end

            compute_RP!(
                stokes.R.RP,
                stokes.P,
                stokes.P0,
                stokes.∇V,
                stokes.Q, # volumetric source/sink term
                ηb,
                rheology,
                phase_ratios,
                dt,
                args,
            ) 

            update_ρg!(ρg, phase_ratios, rheology, args)

            if strain_increment
                @parallel (@idx ni .+ 1) compute_strain_rate!(
                    @strain_increment(stokes)..., stokes.∇U, @displacement(stokes)..., _di...
                )

                @parallel (@idx ni .+ 1) compute_strain_rate_from_increment!(
                    @strain(stokes)..., @strain_increment(stokes)..., _dt
                )
            else
                @parallel (@idx ni .+ 1) compute_strain_rate!(
                    @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
                )
            end

            update_viscosity_τII!(
                stokes,
                phase_ratios,
                args,
                rheology,
                viscosity_cutoff;
                relaxation = viscosity_relaxation,
            )

            compute_stress_DRYEL!(stokes, rheology, phase_ratios, 1, dt)
            update_halo!(stokes.τ.xy)

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.P,
                    @stress(stokes)...,
                    ηdτ,
                    ρg...,
                    ητ,
                    _di...,
                    dt * free_surface,
                )
                # apply boundary conditions
                velocity2displacement!(stokes, dt)
                free_surface_bcs!(stokes, flow_bcs, η, rheology, phase_ratios, dt, di)
                flow_bcs!(stokes, flow_bcs)
                update_halo!(@velocity(stokes)...)
            end
        end

        iter += 1

        if iter % nout == 0 && iter > 1
            # er_η = norm_mpi(@.(log10(η) - log10(η0)))
            # er_η < 1e-3 && (do_visc = false)
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                @velocity(stokes)...,
                stokes.P,
                @stress(stokes)...,
                ρg...,
                _di...,
                dt * free_surface,
            )

            errs = (
                norm_mpi(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) /
                    √((nx_g() - 2) * (ny_g() - 1)),
                norm_mpi(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) /
                    √((nx_g() - 1) * (ny_g() - 2)),
                norm_mpi(stokes.R.RP) / √(nx_g() * ny_g()),
            )

            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            err = maximum_mpi(errs)
            push!(err_evo1, err)
            push!(err_evo2, iter)
            err_it1 = maximum_mpi([norm_Rx[1], norm_Ry[1], norm_∇V[1]])
            rel_err = err / err_it1

            if igg.me == 0 #&& ((verbose && err > ϵ_rel) || iter == iterMax)
                @printf(
                    "Total steps = %d, abs_err = %1.3e , rel_err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    rel_err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && ((err / err_it1) < ϵ_rel || (err < ϵ_abs))
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    # compute vorticity
    @parallel (@idx ni .+ 1) compute_vorticity!(
        stokes.ω.xy, @velocity(stokes)..., inv.(di)...
    )

    # Interpolate shear components to cell center arrays
    shear2center!(stokes.ε)
    shear2center!(stokes.ε_pl)
    shear2center!(stokes.Δε)

    # accumulate plastic strain tensor
    accumulate_tensor!(stokes.EII_pl, stokes.ε_pl, dt)

    @parallel (@idx ni .+ 1) multi_copy!(@tensor(stokes.τ_o), @tensor(stokes.τ))
    @parallel (@idx ni) multi_copy!(@tensor_center(stokes.τ_o), @tensor_center(stokes.τ))

    return (
        iter = iter,
        err_evo1 = err_evo1,
        err_evo2 = err_evo2,
        norm_Rx = norm_Rx,
        norm_Ry = norm_Ry,
        norm_∇V = norm_∇V,
    )
end

# # Residuals
# Rx    .= (.-(Pt[2:end,:] .- Pt[1:end-1,:])./Δx .- (ΔPψ[2:end,:] .- ΔPψ[1:end-1,:])./Δx .+ (Txx[2:end,:] .- Txx[1:end-1,:])./Δx .+ (Txy[2:end-1,2:end] .- Txy[2:end-1,1:end-1])./Δy)
# Ry    .= (.-(Pt[:,2:end] .- Pt[:,1:end-1])./Δy .- (ΔPψ[:,2:end] .- ΔPψ[:,1:end-1])./Δy .+ (Tyy[:,2:end] .- Tyy[:,1:end-1])./Δy .+ (Txy[2:end,2:end-1] .- Txy[1:end-1,2:end-1])./Δx)
# Rp    .= .-∇V .- comp*(Pt.-Pt0)./ηb 
         
