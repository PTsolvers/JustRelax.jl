function heatdiffusion_PT!(thermal, args...; kwargs)
    return heatdiffusion_PT!(backend(thermal), thermal, args...; kwargs = kwargs)
end

function heatdiffusion_PT!(::CPUBackendTrait, thermal, args...; kwargs)
    return _heatdiffusion_PT!(thermal, args...; kwargs...)
end

"""
    heatdiffusion_PT!(thermal, pt_thermal, K, ρCp, dt, di; iterMax, nout, verbose)

Heat diffusion solver using Pseudo-Transient iterations. Both `K` and `ρCp` are n-dimensional arrays.
"""
function _heatdiffusion_PT!(
        thermal::JustRelax.ThermalArrays,
        pt_thermal::JustRelax.PTThermalCoeffs,
        thermal_bc::TemperatureBoundaryConditions,
        K::AbstractArray,
        ρCp::AbstractArray,
        dt,
        di;
        igg = nothing,
        b_width = (4, 4, 1),
        iterMax = 50.0e3,
        nout = 1.0e3,
        verbose = true,
        kwargs...,
    )
    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    ni = size(thermal.Tc)
    @copy thermal.Told thermal.T

    # errors
    iter_count = Int64[]
    norm_ResT = Float64[]
    sizehint!(iter_count, Int(iterMax))
    sizehint!(norm_ResT, Int(iterMax))

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0.0e0
    err = 2 * ϵ

    println("\n ====================================\n")
    println("Starting thermal diffusion solver...\n")

    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            @parallel flux_range(ni...) compute_flux!(
                @qT(thermal)..., @qT2(thermal)..., thermal.T, K, pt_thermal.θr_dτ, _di...
            )
            update_T(
                nothing,
                b_width,
                thermal,
                ρCp,
                pt_thermal,
                thermal_bc.dirichlet,
                _dt,
                _di,
                ni,
            )
            thermal_bcs!(thermal, thermal_bc)
            update_halo!(thermal.T)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel residual_range(ni...) check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    @qT2(thermal)...,
                    thermal.H,
                    thermal.shear_heating,
                    ρCp,
                    thermal_bc.dirichlet,
                    _dt,
                    _di...,
                )
            end

            err = norm(thermal.ResT) * _sq_len_RT

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if verbose
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    println("\n ...solver finished in $(round(wtime0, sigdigits = 5)) seconds \n")
    println("====================================\n")

    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)
    temperature2center!(thermal)

    return (iter_count = iter_count, norm_ResT = norm_ResT)
end

"""
    heatdiffusion_PT!(thermal, pt_thermal, rheology, dt, di; iterMax, nout, verbose)

Heat diffusion solver using Pseudo-Transient iterations.
"""
function _heatdiffusion_PT!(
        thermal::JustRelax.ThermalArrays,
        pt_thermal::JustRelax.PTThermalCoeffs,
        thermal_bc::TemperatureBoundaryConditions,
        rheology,
        args::NamedTuple,
        dt,
        di;
        igg = nothing,
        phase = nothing,
        stokes = nothing,
        b_width = (4, 4, 4),
        iterMax = 50.0e3,
        nout = 1.0e3,
        verbose = true,
        kwargs...,
    )
    phases = get_phase(phase)

    # Compute some constant stuff
    _dt = inv(dt)
    _di = inv.(di)
    _sq_len_RT = inv(sqrt(length(thermal.ResT)))
    ϵ = pt_thermal.ϵ
    ni = size(thermal.Tc)
    @copy thermal.Told thermal.T
    !isnothing(phase) && update_pt_thermal_arrays!(pt_thermal, phase, rheology, args, _dt)

    # compute constant part of the adiabatic heating term
    adiabatic_heating!(thermal, stokes, rheology, phases, di)

    # errors
    iter_count = Int64[]
    norm_ResT = Float64[]
    sizehint!(iter_count, Int(iterMax))
    sizehint!(norm_ResT, Int(iterMax))

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0.0e0
    err = 2 * ϵ

    println("\n ====================================\n")
    println("Starting thermal diffusion solver...\n")

    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            update_thermal_coeffs!(pt_thermal, rheology, phase, args, dt)
            @parallel flux_range(ni...) compute_flux!(
                @qT(thermal)...,
                @qT2(thermal)...,
                thermal.T,
                rheology,
                phases,
                pt_thermal.θr_dτ,
                _di...,
                args,
            )
            update_T(
                nothing,
                b_width,
                thermal,
                rheology,
                phases,
                pt_thermal,
                thermal_bc.dirichlet,
                _dt,
                _di,
                ni,
                args,
            )
            thermal_bcs!(thermal, thermal_bc)
            update_halo!(thermal.T)

            !isnothing(phase) &&
                update_pt_thermal_arrays!(pt_thermal, phase, rheology, args, _dt)
        end

        iter += 1

        if iter % nout == 0
            wtime0 += @elapsed begin
                @parallel residual_range(ni...) check_res!(
                    thermal.ResT,
                    thermal.T,
                    thermal.Told,
                    @qT2(thermal)...,
                    thermal.H,
                    thermal.shear_heating,
                    thermal.adiabatic,
                    rheology,
                    phases,
                    thermal_bc.dirichlet,
                    _dt,
                    _di...,
                    args,
                )
            end

            err = norm(thermal.ResT) * _sq_len_RT

            push!(norm_ResT, err)
            push!(iter_count, iter)

            if verbose
                @printf("iter = %d, err = %1.3e \n", iter, err)
            end
        end
    end

    println("\n ...solver finished in $(round(wtime0, sigdigits = 5)) seconds \n")
    println("====================================\n")

    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)
    temperature2center!(thermal)

    return (iter_count = iter_count, norm_ResT = norm_ResT)
end

@inline flux_range(nx, ny) = @idx (nx + 3, ny + 1)
@inline flux_range(nx, ny, nz) = @idx (nx, ny, nz)

@inline update_range(nx, ny) = @idx (nx + 1, ny - 1)
@inline update_range(nx, ny, nz) = residual_range(nx, ny, nz)

@inline residual_range(nx, ny) = update_range(nx, ny)
@inline residual_range(nx, ny, nz) = @idx (nx - 1, ny - 1, nz - 1)
