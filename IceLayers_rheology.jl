using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheologies()

    η_reg   = 1.0e15
    C       = 10.0e6
    ϕ       = 15
    Ψ       = 0.0
    soft_C  = NonLinearSoftening(; ξ₀ = C, Δ = C / 1.0e5) # nonlinear softening law
    pl      = DruckerPrager_regularised(; C = C * MPa, ϕ = ϕ, η_vp = (η_reg) * Pas, Ψ = Ψ, softening_C = soft_C)
    G0      = 25.0e9Pa        # elastic shear modulus
    G_magma = 10.0e9Pa        # elastic shear modulus magma

    el = ConstantElasticity(; G = G0, ν = 0.45)
    el_magma = ConstantElasticity(; G = G_magma, ν = 0.45)
    β = 1 / el.Kb.val
    Cp = 1200.0

    #dislocation laws
    disl_crust  = SetDislocationCreep(Dislocation.wet_quartzite_Hirth_2001)
    disl_litho  = SetDislocationCreep(Dislocation.dry_olivine_Karato_2003)
    # diffusion laws
    diff_litho  = SetDiffusionCreep(Diffusion.dry_olivine_Hirth_2003)

    # Define rheolgy struct
    return rheology = (
        # Name = "Upper crust",
        SetMaterialParams(;
            Phase = 1,
            # Density = PT_Density(; ρ0 = 2.7e3, T0 = 273.15, β = β),
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            # CompositeRheology = CompositeRheology((disl_top, el, pl)),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e21), el, pl)),
            # Melting = MeltingParam_Smooth3rdOrder(a = 517.9, b = -1619.0, c = 1699.0, d = -597.4), #mafic melting curve
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name = "Lower crust",
        SetMaterialParams(;
            Phase = 2,
            # Density = PT_Density(; ρ0 = 2.75e3, T0 = 273.15, β = β),
            Density = ConstantDensity(; ρ = 2.7e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            # CompositeRheology = CompositeRheology((disl_bot, el, pl)),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e21), el, pl)),
            # Melting = MeltingParam_Smooth3rdOrder(a = 517.9, b = -1619.0, c = 1699.0, d = -597.4), #mafic melting curve
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "lithosphere mantle",
        SetMaterialParams(;
            Phase = 3,
            # Density = PT_Density(; ρ0 = 2.4e3, T0 = 273.15, β = β_magma),
            Density = ConstantDensity(; ρ = 3.3e3),
            Conductivity = ConstantConductivity(; k = 3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            CompositeRheology = CompositeRheology( (LinearViscous(; η=1e20), el_magma)),
        ),
        # Name              = "Ice",
        SetMaterialParams(;
            Phase = 4,
            Density = ConstantDensity(; ρ = 1e3),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e21), el, pl)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 5,
            Density = ConstantDensity(; ρ = 0.0e0),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = 2.5),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e22), el, pl)),
            Gravity = ConstantGravity(; g = 9.81),
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    return @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N, T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii = I[1] + offi
            jj = I[2] + offj

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue

            xvᵢ = (
                xvi[1][ii],
                xvi[2][jj],
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end


function remove_ice!(chain, particles, pPhases, ice_phase, air_phase)

    chain_y = chain.h_vertices
    (; coords, index) = particles;
    yp = coords[2]

    @parallel_indices (I...) function _remove_ice!(chain_y, yp, index, pPhases, ice_phase, air_phase)
        # check if the cell is occupied by ice and rock
        is_rock = false
        is_ice  = false
        modify_chain = false
        for ip in cellaxes(index)
            @inbounds @index(index[ip, I...]) || continue
            phase_p = @inbounds @index pPhases[ip, I...]

            phase_p == ice_phase && (is_ice = true)
            !(phase_p == ice_phase) && !(phase_p == air_phase) && (is_rock = true)

            if is_ice && is_rock
                modify_chain = true
            end
        end

        if modify_chain
            y_min = Inf
            for ip in cellaxes(index)
                @inbounds @index(index[ip, I...]) || continue
                yp_cell = @inbounds @index yp[ip, I...]
                phase_p = @inbounds @index pPhases[ip, I...]

                if phase_p == ice_phase
                    if yp_cell < y_min
                        y_min = yp_cell
                    end
                    @index pPhases[ip, I...] = air_phase
                end
            end
            chain_y[I[1]] = y_min
        end

        return nothing
    end

    @parallel (@idx size(index)) _remove_ice!(chain_y, yp, index, pPhases, ice_phase, air_phase)
    return nothing
end

function force_air_above_chain!(chain, particles, pPhases, air_phase)

    (; coords, index) = particles
    chain_x = chain.cell_vertices
    chain_y = chain.h_vertices

    # Iterate over the chain vertices force air above the chain
    @parallel_indices (I...) function _force_air_above_chain!(chain_x, chain_y, coords, index, pPhases, air_phase)
        i  = first(I)
        p1 = JustPIC._2D.GridGeometryUtils.Point(chain_x[i], chain_y[i])
        p2 = JustPIC._2D.GridGeometryUtils.Point(chain_x[i+1], chain_y[i+1])
        l  = JustPIC._2D.GridGeometryUtils.Line(p1, p2)

        for ip in cellaxes(index)
            @index(index[ip, I...]) || continue
            phase_p = @index pPhases[ip, I...]

            if !(phase_p == air_phase)
                px = @index coords[1][ip, I...]
                py = @index coords[2][ip, I...]

                yline = l.slope * px + l.intercept

                py < yline && continue # skip if particle is below the line

                @index pPhases[ip, I...] = air_phase
            end
        end
        return nothing
    end

    @parallel (@idx size(index)) _force_air_above_chain!(chain_x, chain_y, coords, index, pPhases, air_phase)

    return nothing
end

