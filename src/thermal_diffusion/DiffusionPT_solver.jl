"""
    heatdiffusion_PT!(thermal, args...; kwargs...)

Dispatch pseudo-transient thermal diffusion to the backend associated with
`thermal`.

See the `_heatdiffusion_PT!` methods below for the supported argument groups:
constant `K` and `ρCp` fields, or rheology-driven properties with optional phase
ratios and Stokes fields for adiabatic heating.
"""
function heatdiffusion_PT!(thermal, args...; kwargs)
    return heatdiffusion_PT!(backend(thermal), thermal, args...; kwargs = kwargs)
end

function heatdiffusion_PT!(::CPUBackendTrait, thermal, args...; kwargs)
    return _heatdiffusion_PT!(thermal, args...; kwargs...)
end

"""
    _heatdiffusion_PT!(thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid;
        igg, b_width, iterMax, nout, verbose)

Solve the heat equation with pseudo-transient iterations using precomputed,
cell-centered material properties.

`K` is the thermal conductivity field and `ρCp` is the volumetric heat-capacity
field on the thermal grid. `pt_thermal` supplies the pseudo-transient
coefficients, `thermal_bc` applies the temperature boundary conditions after
each update, and `grid` provides metric terms and halo layout.

Returns a named tuple containing the iteration numbers and residual norms
sampled every `nout` iterations.
"""
function _heatdiffusion_PT!(
        thermal::JustRelax.ThermalArrays,
        pt_thermal::JustRelax.PTThermalCoeffs,
        thermal_bc::TemperatureBoundaryConditions,
        K::AbstractArray,
        ρCp::AbstractArray,
        dt,
        grid::Geometry;
        igg = nothing,
        b_width = (4, 4, 1),
        iterMax = 50.0e3,
        nout = 1.0e3,
        verbose = true,
        kwargs...,
    )
    # Compute some constant stuff
    di = grid.di
    _di = grid._di
    _dt = inv(dt)

    ϵ = pt_thermal.ϵ
    ni = size(thermal.H)
    _sq_len_RT = inv(sqrt(prod(ni)))
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

    if isnothing(igg) || igg.me == 0
        println("\n====================================\n")
        println("Starting thermal diffusion solver...\n")
    end

    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            if length(ni) == 2
                @parallel flux_range(ni...) compute_flux!(
                    @qT(thermal)...,
                    @qT2(thermal)...,
                    thermal.T,
                    K,
                    pt_thermal.θr_dτ,
                    _di.center,
                    thermal_bc.constant_flux,
                )
            else
                @parallel flux_range(ni...) compute_flux!(
                    @qT(thermal)...,
                    @qT2(thermal)...,
                    thermal.T,
                    K,
                    pt_thermal.θr_dτ,
                    _di.center,
                    thermal_bc.constant_flux,
                )
            end
            update_T(
                nothing,
                b_width,
                thermal,
                ρCp,
                pt_thermal,
                thermal_bc.dirichlet,
                _dt,
                _di.center,
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
                    _di.center,
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

    if isnothing(igg) || igg.me == 0
        println("\n ...solver finished in $(round(wtime0, sigdigits = 5)) seconds \n")
        println("====================================\n")
    end

    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)
    return (iter_count = iter_count, norm_ResT = norm_ResT)
end

function _heatdiffusion_PT!(
        thermal::JustRelax.ThermalArrays,
        pt_thermal::JustRelax.PTThermalCoeffs,
        thermal_bc::TemperatureBoundaryConditions,
        K::AbstractArray,
        ρCp::AbstractArray,
        dt,
        di::Union{NTuple{N, <:Real}, NamedTuple};
        kwargs...,
    ) where {N}
    grid = JustRelax.legacy_uniform_grid(size(thermal.H), di)
    return _heatdiffusion_PT!(thermal, pt_thermal, thermal_bc, K, ρCp, dt, grid; kwargs.data...)
end

"""
    _heatdiffusion_PT!(thermal, pt_thermal, thermal_bc, rheology, args, dt, grid;
        igg, phase, stokes, b_width, iterMax, nout, verbose)

Solve the heat equation with pseudo-transient iterations using thermal
properties derived from `rheology`.

`args` is a named tuple of thermodynamic fields sampled at thermal-cell centers,
typically including `T` and `P`. When `phase` is provided, pseudo-transient
coefficients are recomputed from the local phase ratios each iteration. When
`stokes` is provided, `thermal.adiabatic` is refreshed before the iteration loop
to include the adiabatic heating contribution.

Returns a named tuple containing the sampled iteration counts and residual
history.
"""
function _heatdiffusion_PT!(
        thermal::JustRelax.ThermalArrays,
        pt_thermal::JustRelax.PTThermalCoeffs,
        thermal_bc::TemperatureBoundaryConditions,
        rheology,
        args::NamedTuple,
        dt,
        grid::Geometry;
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
    di = grid.di
    _di = grid._di
    _dt = inv(dt)
    ϵ = pt_thermal.ϵ
    ni = size(thermal.H)
    _sq_len_RT = inv(sqrt(prod(ni)))
    @copy thermal.Told thermal.T
    !isnothing(phase) && update_pt_thermal_arrays!(pt_thermal, phase, rheology, args, _dt)

    # compute constant part of the adiabatic heating term
    adiabatic_heating!(thermal, stokes, rheology, phases, _dt, grid)

    # errors
    iter_count = Int64[]
    norm_ResT = Float64[]
    sizehint!(iter_count, Int(iterMax))
    sizehint!(norm_ResT, Int(iterMax))

    # Pseudo-transient iteration
    iter = 0
    wtime0 = 0.0e0
    err = 2 * ϵ

    if isnothing(igg) || igg.me == 0
        println("\n====================================\n")
        println("Starting thermal diffusion solver...\n")
    end

    while err > ϵ && iter < iterMax
        wtime0 += @elapsed begin
            update_thermal_coeffs!(pt_thermal, rheology, phase, args, dt)
            if length(ni) == 2
                @parallel flux_range(ni...) compute_flux!(
                    @qT(thermal)...,
                    @qT2(thermal)...,
                    thermal.T,
                    rheology,
                    phases,
                    pt_thermal.θr_dτ,
                    _di.center,
                    args,
                    thermal_bc.constant_flux,
                )
            else
                @parallel flux_range(ni...) compute_flux!(
                    @qT(thermal)...,
                    @qT2(thermal)...,
                    thermal.T,
                    rheology,
                    phases,
                    pt_thermal.θr_dτ,
                    _di.center,
                    args,
                    thermal_bc.constant_flux,
                )
            end
            update_T(
                nothing,
                b_width,
                thermal,
                rheology,
                phases,
                pt_thermal,
                thermal_bc.dirichlet,
                _dt,
                _di.center,
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
                    _di.center,
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

    if isnothing(igg) || igg.me == 0
        println("\n ...solver finished in $(round(wtime0, sigdigits = 5)) seconds \n")
        println("====================================\n")
    end

    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)
    return (iter_count = iter_count, norm_ResT = norm_ResT)
end

function _heatdiffusion_PT!(
        thermal::JustRelax.ThermalArrays,
        pt_thermal::JustRelax.PTThermalCoeffs,
        thermal_bc::TemperatureBoundaryConditions,
        rheology,
        args::NamedTuple,
        dt,
        di::Union{NTuple{N, <:Real}, NamedTuple};
        kwargs...,
    ) where {N}
    grid = JustRelax.legacy_uniform_grid(size(thermal.H), di)
    return _heatdiffusion_PT!(thermal, pt_thermal, thermal_bc, rheology, args, dt, grid; kwargs...)
end

@inline flux_range(nx, ny) = @idx (nx + 1, ny + 1)
@inline flux_range(nx, ny, nz) = @idx (nx + 1, ny + 1, nz + 1)

@inline update_range(nx, ny) = @idx (nx, ny)
@inline update_range(nx, ny, nz) = @idx (nx, ny, nz)

@inline residual_range(nx, ny) = update_range(nx, ny)
@inline residual_range(nx, ny, nz) = update_range(nx, ny, nz)
