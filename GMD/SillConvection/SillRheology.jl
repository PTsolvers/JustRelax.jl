using Adapt

## Phase diagram + three-phase (melt / crystal / gas) magma
#
# Both phases use the same melt law, selected by the `melting` keyword — see the long
# note at the selection site below for why `:analytic` is the default and what it costs.
# The MAGEMin diagrams are now generated at the sill's real bulk water content
# (2.91 wt%, "Heise Sill" / "Heise Host rock" in Phase_diagrams/Heise.dat), so the two
# phases are mutually consistent and the earlier hybrid workaround is gone.
#
# The `oxd_wt_*` tuples must match the diagram they are paired with: the melt fraction
# comes from the diagram (or a fit to it) and the melt viscosity/density from the tuple,
# so a mismatch silently describes two different rocks. Kilgore tuff = sill,
# pre-Kilgore = host rock.
#
# Density is a `ThreePhase_Density` mixture rather than the diagram's own `Rho` column,
# because the lookup cannot see exsolved gas — and the gas volume fraction is what makes
# a water-rich sill buoyant enough to convect. Second boiling comes from the melt's
# absolute water capacity mH2O_diss(T,P)·ϕ shrinking as ϕ falls.
#
# `ArrayType = CuArray` adapts the lookup tables onto the device; both
# `PhaseDiagram_LookupTable` and its `LinearInterpolator` fields are
# `Adapt.@adapt_structure`d in GeoParams, so the interpolations can be evaluated
# from inside GPU kernels.
function init_rheologies(
        oxd_wt_sill, oxd_wt_host_rock;
        scaling = 1e0Pas, magma = true, CharDim = nothing, ArrayType = Array,
        G_host = 10.0e9Pa, G_sill = 6.0e9Pa, ν = 0.25,
        C_host = 10.0e6Pa, ϕ_fric_host = 30.0, η_reg = 1.0e12Pas,
        melting = :analytic, k_sol = 0.15, k_liq = 0.02,
    )
    # Resolve relative to this file, not the run's cwd.
    sill_PD      = joinpath(@__DIR__, "Phase_diagrams", "Heise_Sill.in")
    host_rock_PD = joinpath(@__DIR__, "Phase_diagrams", "Heise_Host_rock.in")

    # Melting law. Two mutually exclusive options, selected by `melting`:
    #
    #   :lookup   — the MAGEMin diagrams directly, ϕ(T,P). Thermodynamically exact and
    #               pressure-dependent, but the solidus is a *sub-grid step*: ϕ jumps
    #               0 → 0.75 (sill) / 0 → 0.69 (host) across one table interval (1.27 K),
    #               giving dϕ/dT up to 0.54 /K. That is an 8-order viscosity jump between
    #               adjacent cells and a ~180x latent-heat spike in Cp_eff, neither of
    #               which the Stokes or thermal solver can resolve at any grid we can run.
    #
    #   :analytic — smoothed 5th-order fits to those *same* diagrams at 113.8 MPa. The
    #               fits reproduce MAGEMin to better than 0.006 in ϕ at the sill's actual
    #               operating temperature (850-900 °C); all the deliberate disagreement is
    #               in the solidus region, where `k_sol` spreads the step over ~1/k_sol K.
    #               `compute_dϕdT` is then analytic instead of a finite difference of a
    #               step, so the latent-heat term is differentiable and consistent.
    #
    T_s_sill, T_l_sill = 973.735937, 1250.493750   # K, from the 513² diagram at 113.8 MPa
    T_s_host, T_l_host = 968.657812, 1355.864844
    melting_sill, melting_host = if melting === :lookup
        # Diagrams stay dimensional here; CharDim (if any) is applied by SetMaterialParams.
        Adapt.adapt(ArrayType, MAGEMin_Diagram(sill_PD)),
        Adapt.adapt(ArrayType, MAGEMin_Diagram(host_rock_PD))
    elseif melting === :analytic
        SmoothMelting(;
            p = MeltingParam_5thOrder(
                a = -1.02203092053987e-12 / K^5,
                b =  5.85021047611182e-09 / K^4,
                c = -1.33376749670975e-05 / K^3,
                d =  1.51339525604033e-02 / K^2,
                e = -8.54297201421483e+00 / K,
                f =  1.91931372823878e+03 * NoUnits,
                T_s = (T_s_sill - 2 / k_sol)K, T_l = T_l_sill * K,
            ), k_sol = k_sol / K, k_liq = k_liq / K,
        ),
        SmoothMelting(;
            p = MeltingParam_5thOrder(
                a =  5.85791572818118e-13 / K^5,
                b = -3.48698820726664e-09 / K^4,
                c =  8.28918985424353e-06 / K^3,
                d = -9.83566052485603e-03 / K^2,
                e =  5.82541815444112e+00 / K,
                f = -1.37699575079578e+03 * NoUnits,
                T_s = (T_s_host - 2 / k_sol)K, T_l = T_l_host * K,
            ), k_sol = k_sol / K, k_liq = k_liq / K,
        )
    else
        error("`melting` must be :analytic or :lookup, got $(repr(melting))")
    end

    # Both the melt viscosity and the melt density take the per-cell dissolved water
    # `mH2O` from `args`, overriding their own `oxd_wt[9]`. This is the main control on
    # whether the sill convects, and it is why `mH2O_melt` is threaded through the solver.
    sill = magma ? ViscosityPartialMelt_Costa_etal_2009(η = GiordanoMeltViscosity(oxd_wt = oxd_wt_sill, η0 = scaling)) : LinearViscous(η = 1.0e4Pa*s)
    host_rock = magma ? ViscosityPartialMelt_Costa_etal_2009(η = GiordanoMeltViscosity(oxd_wt = oxd_wt_host_rock, η0 = scaling)) : LinearViscous(η = 1.0e13Pa*s)

    # ρ = ϕ_m·ρmelt + ϕ_gas·ρgas + ϕ_x·ρsolid. The gas EOS is only evaluated where
    # ϕ_gas != 0, so RedlichKwong never sees a cell outside its calibration window.
    ρ_host = ThreePhase_Density(
        ρmelt  = Melt_DensityX(oxd_wt = oxd_wt_host_rock),
        ρsolid = PT_Density(; ρ0 = 2700kg/m^3, α = 3e-5/K),
        ρgas   = RedlichKwong_Density(),
    )
    ρ_sill = ThreePhase_Density(
        ρmelt  = Melt_DensityX(oxd_wt = oxd_wt_sill),
        ρsolid = PT_Density(; ρ0 = 2700kg/m^3, α = 3e-5/K),
        ρgas   = RedlichKwong_Density(),
    )

    # Define rheolgy struct
    rheology = (
        # Name              = "host_rock",
        SetMaterialParams(;
            Phase             = 1,
            Density           = ρ_host,
            HeatCapacity      = Latent_HeatCapacity(Cp=T_HeatCapacity_Whittington(), Q_L=350e3J/kg),
            Conductivity      = T_Conductivity_Whittington(),
            CompositeRheology = CompositeRheology((host_rock)),
            Melting           = melting_host,
            Solubility        = Liu2005_Solubility(),
            Gravity           = ConstantGravity(),
            CharDim           = CharDim,
        ),
        # Name              = "Sill",
        SetMaterialParams(;
            Phase             = 2,
            Density           = ρ_sill,
            HeatCapacity      = Latent_HeatCapacity(Cp=T_HeatCapacity_Whittington(), Q_L=350e3J/kg),
            Conductivity      = T_Conductivity_Whittington(),
            CompositeRheology = CompositeRheology((sill, )),
            Melting           = melting_sill,
            Solubility        = Liu2005_Solubility(),
            CharDim           = CharDim,
        ),
    )
    return rheology
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
