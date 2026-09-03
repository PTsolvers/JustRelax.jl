using GeoParams.Dislocation
using GeoParams.Diffusion

function init_rheologies(CharDim; plastic = true, linear = false)

    #diff_wet_olivine = SetDiffusionCreep(
    #    Diffusion.dry_olivine_Hirth_2003;
    #    E = 375.0e3 * J * mol^-1.0,
    #    V = 2.0e-6m^3 * mol^-1.0
    #) #530 #10e-6
    #disl_wet_olivine = SetDislocationCreep(
    #    Dislocation.dry_olivine_Hirth_2003;
    #    E = 530.0e3 * J * mol^-1.0,
    #    V = 3.5e-6m^3 * mol^-1.0
    #) #375 # 14e-6

    diff_wet_olivine = SetDiffusionCreep(
        Diffusion.dry_olivine_Hirth_2003;
        A = 1.5e9 * MPa^-1.0 * μm^3 * s^-1.0,
        V = 14.0e-6m^3 * mol^-1.0,
        d = 1.0e-3m
    )

    disl_wet_olivine = SetDislocationCreep(
        Dislocation.dry_olivine_Hirth_2003;
        V = 14.0e-6m^3 * mol^-1.0,
    )

    #     diff_wet_olivine = SetDiffusionCreep(
    #     Diffusion.dry_olivine_Hirth_2003;
    #     A = 1.5e9 * MPa^-1.0 * μm^3 * s^-1.0,
    #     V = 6.0e-6m^3 * mol^-1.0,
    #     d = 1.0e-3m
    # )

    # disl_wet_olivine = SetDislocationCreep(
    #     Dislocation.dry_olivine_Hirth_2003;
    #     V = 10.0e-6m^3 * mol^-1.0,
    # )

    # disl_law_crust = SetDislocationCreep(Dislocation.granite_Carter_1987)
    disl_law_crust = SetDislocationCreep(Dislocation.strong_diabase_Mackwell_1998)
    diff_law_crust = SetDiffusionCreep(Diffusion.dry_anorthite_Rybacki_2006)

    disl_law_crust_cont = SetDislocationCreep(Dislocation.wet_quartzite_Hirth_2001)
    diff_law_crust_cont = SetDiffusionCreep(Diffusion.dry_anorthite_Rybacki_2006)

    #diff_wet_olivine = DiffusionCreep(
    #    #A = 4.5e-15 * Pa^-1.0 * s^-1.0,
    #    A = 1.5e9 * MPa^-1.0 * μm^3 * s^-1.0,
    #    n = 1.0NoUnits,
    #    p = 3.0NoUnits,
    #    r = 0.0NoUnits,
    #    E = 375.0e3 * J * mol^-1.0,
    #    V = 3.0e-6m^3 * mol^-1.0,
    #    Apparatus=AxialCompression
    #)

    #disl_wet_olivine = DislocationCreep(
    #    #A = 7.4e-15 * Pa^-3.5 * s^-1.0,
    #    A = 1.1e5 * MPa^-3.5 * s^-1.0,
    #    n = 3.5NoUnits,
    #    r = 0.0NoUnits,
    #    E = 530.0e3 * J * mol^-1.0,
    #    V = 14.0e-6m^3 * mol^-1.0,
    #    Apparatus=AxialCompression
    #)

    #diff_wet_olivine = SetDiffusionCreep(Diffusion.dry_olivine_Hirth_2003)#;E = 375e3*J*mol^-1.0,V=2e-6m^3*mol^-1.0) #530 #10e-6
    #disl_wet_olivine = SetDislocationCreep(Dislocation.dry_olivine_Hirth_2003)#;E = 530e3*J*mol^-1.0,V=6e-6m^3*mol^-1.0) #375 # 14e-6
    # 32 # 32
    el = ConstantElasticity(; G = 60.0GPa, ν = 0.45)
    α  = 3.0e-5 / K #2.4e-5 # 1 / K #3e-5
    β  = 1 / el.Kb.val / GPa #(1.5e-11)
    Cp = 1000 * J / kg / K    # J / kg K # 750
    k  = 3.0 * Watt / K / m # 2.5
    # T0 = 1350# + 273
    T0 = (20 + 273)K

    # Regularized Drucker-Prager plasticity for the lithospheric mantle. The low friction
    # angle (sin ϕ = 0.1) caps lithospheric strength at a few hundred MPa so the slab can
    # bend; η_vp regularizes the shear bands and must stay above the solver's lower
    # viscosity cutoff.
    plastic == true ? Coh = 10.0MPa : Coh = Inf

    pl = DruckerPrager_regularised(;
        # C = 10.0MPa,
        # ϕ = asind(0.1),
        C = Coh,
        ϕ = 30.0,
        Ψ = 0.0,
        η_vp = 1.0e22Pa * s,
    )

    pl_oc = DruckerPrager_regularised(;
        C = 10.0MPa,
        ϕ = 5.0,
        Ψ = 0.0,
        η_vp = 1.0e22Pa * s,
    )

    mantle_rheology      = linear == true ? CompositeRheology((LinearViscous(; η = 1.0e22Pa * s),)) : CompositeRheology((disl_wet_olivine, diff_wet_olivine, el, pl))
    lithosphere_rheology = linear == true ? CompositeRheology((LinearViscous(; η = 1.0e22Pa * s),)) : CompositeRheology((disl_wet_olivine, diff_wet_olivine, el, pl))
    crustal_rheology     = linear == true ? CompositeRheology((LinearViscous(; η = 1.0e22Pa * s),)) : CompositeRheology((disl_law_crust_cont, diff_law_crust_cont, el, pl))
    crustal_rheology_oc  = linear == true ? CompositeRheology((LinearViscous(; η = 1.0e22Pa * s),)) : CompositeRheology((disl_law_crust, diff_law_crust, el, pl_oc))
    rheology = (
        # Name = "Asthenoshpere",
        SetMaterialParams(;
            Phase = 1,
            #Density = ConstantDensity(; ρ = 3.3e3),
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, α = α, β = β, T0 = T0),  #273 + 1474
            # CompositeRheology = CompositeRheology((LinearViscous(; η = 3.0e20),el)),
            CompositeRheology = mantle_rheology,
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = k),
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
        # Name              = "Left lithosphere",
        SetMaterialParams(;
            Phase = 2,
            #Density = ConstantDensity(; ρ = 3.365e3),
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, α = α, β = β, T0 = T0),
            # CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23Pa*s),el)),
            CompositeRheology = lithosphere_rheology,
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = k),
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
        # Name              = "Right lithosphere",
        SetMaterialParams(;
            Phase = 3,
            #Density = ConstantDensity(; ρ = 3.365e3),
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, α = α, β = β, T0 = T0),
            CompositeRheology = lithosphere_rheology,
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = k),
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
        # Name              = "Left crust",
        SetMaterialParams(;
            Phase = 4,
            #Density = ConstantDensity(; ρ = 3.365e3),
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, α = α, β = β, T0 = T0),
            CompositeRheology = crustal_rheology_oc,
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = k),
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
        # Name              = "Right Crust",
        SetMaterialParams(;
            Phase = 5,
            #Density = ConstantDensity(; ρ = 2.9e3),
            Density = PT_Density(; ρ0 = 2.9e3kg / m^3, α = α, β = β, T0 = T0),
            CompositeRheology = crustal_rheology,
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = k),
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
        # Name              = "Weak Zone Upper Crust",
        SetMaterialParams(;
            Phase = 6,
            #Density = ConstantDensity(; ρ = 3.365e3),
            Density = PT_Density(; ρ0 = 3.3e3kg / m^3, α = α, β = β, T0 = T0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e18Pa * s),el,pl)),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = k),
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
        # Name              = "Right Crust marker",
        SetMaterialParams(;
            Phase = 7,
            #Density = ConstantDensity(; ρ = 2.9e3),
            Density = PT_Density(; ρ0 = 2.9e3kg / m^3, α = α, β = β, T0 = T0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e23Pa * s), el, pl)),
            HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            Conductivity = ConstantConductivity(; k = k),
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase = 8,
            # Density = ConstantDensity(; ρ = 1.293kg / m^3),
            Density = ConstantDensity(; ρ = 100.0kg / m^3),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e19Pa * s),)),
            HeatCapacity = ConstantHeatCapacity(; Cp = 1.0e3 * 1.0e3 * J / kg / K),
            Conductivity = ConstantConductivity(; k = 1.0 * Watt / K / m),
            # HeatCapacity = ConstantHeatCapacity(; Cp = Cp),
            # Conductivity = ConstantConductivity(; k = k),
            Gravity = ConstantGravity(; g = 9.81m / s^2),
            CharDim = CharDim,
        ),
    )

    return rheology

end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    return @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

"""
    reset_ridge!(pPhases, particles, x_ridge, z_crust, z_lith; phases = (4, 2, 1))

Hold the particle phases left of `x_ridge` at the accreting-plate layering: oceanic
crust down to `z_crust`, mantle lithosphere down to `z_lith`, asthenosphere below.
New crust is therefore supplied as the plate migrates away from the boundary, without
imposing any velocity on it. `phases` are the crust, mantle lithosphere and
asthenosphere indices and must match the initial setup.

Temperature is left to the thermal solver: the left boundary is no-flux, so pinning it
would erase the thermal boundary layer and the margin would neck under slab pull.
"""
function reset_ridge!(pPhases, particles, x_ridge, z_crust, z_lith; phases = (4, 2, 1))
    @parallel (@idx size(pPhases)) _reset_ridge!(
        pPhases, particles.coords, particles.index, x_ridge, z_crust, z_lith, phases
    )
    return nothing
end

@parallel_indices (I...) function _reset_ridge!(pPhases, pcoords, index, x_ridge, z_crust, z_lith, phases)
    for ip in cellaxes(pPhases)
        @index(index[ip, I...]) == 0 && continue

        x = @index pcoords[1][ip, I...]
        z = @index pcoords[2][ip, I...]
        # the sticky air above z = 0 is owned by the marker chain
        (x > x_ridge || z > 0) && continue

        @index pPhases[ip, I...] = Float64(z > z_crust ? phases[1] : z > z_lith ? phases[2] : phases[3])
    end

    return nothing
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
