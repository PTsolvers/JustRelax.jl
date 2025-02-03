function calc_sensitivity_2D!(
    stokes,
    stokesAD,
    η,
    Vx_on_Vy,
    ρg,
    _di,
    dt,
    free_surface,
    θ,
    λ,
    λv,
    relλ,
    rheology,
    phase_ratios,
    θ_dτ,
    ni)

    print("############################################\n")
    print("Calculating Sensitivities\n")
    print("############################################\n")

    if  isdefined(Main,:CUDA)
        mode = Enzyme.Reverse
    else
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse,true)
    end

    stokesAD.τ.xx   .= 0.0
    stokesAD.τ.yy   .= 0.0
    stokesAD.τ.xy_c .= 0.0
    stokesAD.τ.xy   .= 0.0

    stokesAD.Gv  .= 0.0
    stokesAD.Gc  .= 0.0
    stokesAD.G   .= 0.0
    stokesAD.frv .= 0.0
    stokesAD.frc .= 0.0
    stokesAD.fr  .= 0.0
    stokesAD.Cv  .= 0.0
    stokesAD.Cc  .= 0.0
    stokesAD.C   .= 0.0

    ηb      = @zeros(size(η))
    ρb      = @zeros(size(ρg[2]))
    Gvb     = @zeros(size(stokesAD.Gv))
    Gcb     = @zeros(size(stokesAD.Gc))
    Gb      = @zeros(size(stokesAD.Gc))
    frvb    = @zeros(size(stokesAD.frv))
    frcb    = @zeros(size(stokesAD.frc))
    frb     = @zeros(size(stokesAD.frc))
    Cvb     = @zeros(size(stokesAD.Cv))
    Ccb     = @zeros(size(stokesAD.Cc))
    Cb      = @zeros(size(stokesAD.Cc))

    stokesAD.R.Rx .= -stokesAD.VA.Vx[2:end-1,2:end-1]
    stokesAD.R.Ry .= -stokesAD.VA.Vy[2:end-1,2:end-1]


    @parallel (@idx ni) configcall=compute_Res!(
        stokes.R.Rx,
        stokes.R.Ry,
        @velocity(stokes)...,
        Vx_on_Vy,
        stokes.P,
        @stress(stokes)...,
         ρg...,
         _di...,
         dt * free_surface
         ) AD.autodiff_deferred!(
            mode,
            Const(compute_Res!),
            Const{Nothing},
            DuplicatedNoNeed(stokes.R.Rx, stokesAD.R.Rx),
            DuplicatedNoNeed(stokes.R.Ry, stokesAD.R.Ry),
            Const(stokes.V.Vx),
            Const(stokes.V.Vy),
            Const(Vx_on_Vy),
            Const(stokes.P),
            DuplicatedNoNeed(stokes.τ.xx,stokesAD.τ.xx),
            DuplicatedNoNeed(stokes.τ.yy,stokesAD.τ.yy),
            DuplicatedNoNeed(stokes.τ.xy,stokesAD.τ.xy),
            Const(ρg[1]),
            DuplicatedNoNeed(ρg[2],ρb),
            Const(_di[1]),
            Const(_di[2]),
            Const(dt * free_surface))

    @parallel (@idx ni.+1) assemble_parameter_matrices!(
        stokes.EII_pl,
        Gvb,
        Gcb,
        frvb,
        frcb,
        Cvb,
        Ccb,
        rheology,
        phase_ratios.center,
        phase_ratios.vertex)

    @parallel (@idx ni.+1) configcall=update_stresses_center_vertex_psSensTest!(
            @strain(stokes),
            @tensor_center(stokes.ε_pl),
            stokes.EII_pl,
            @tensor_center(stokes.τ),
            (stokes.τ.xy,),
            @tensor_center(stokes.τ_o),
            (stokes.τ_o.xy,),
            θ,
            stokes.P,
            stokes.viscosity.η,
            λ,
            λv,
            stokes.τ.II,
            stokes.viscosity.η_vep,
            relλ,
            dt,
            θ_dτ,
            rheology,
            phase_ratios.center,
            phase_ratios.vertex,
            Gvb,
            Gcb,
            frvb,
            frcb,
            Cvb,
            Ccb
        ) AD.autodiff_deferred!(
            mode,
            Const(update_stresses_center_vertex_psSensTest!),
            Const{Nothing},
            Const(@strain(stokes)),
            Const(@tensor_center(stokes.ε_pl)),
            Const(stokes.EII_pl),
            DuplicatedNoNeed(@tensor_center(stokes.τ),
            @tensor_center(stokesAD.τ)),
            DuplicatedNoNeed((stokes.τ.xy,),(stokesAD.τ.xy,)),
            Const(@tensor_center(stokes.τ_o)),
            Const((stokes.τ_o.xy,)),
            Const(θ),
            Const(stokes.P),
            DuplicatedNoNeed(stokes.viscosity.η,ηb),
            Const(λ),
            Const(λv),
            Const(stokes.τ.II),
            Const(stokes.viscosity.η_vep),
            Const(relλ),
            Const(dt),
            Const(θ_dτ),
            Const(rheology),
            Const(phase_ratios.center),
            Const(phase_ratios.vertex),
            DuplicatedNoNeed(Gvb,stokesAD.Gv),
            DuplicatedNoNeed(Gcb,stokesAD.Gc),
            DuplicatedNoNeed(frvb,stokesAD.frv),
            DuplicatedNoNeed(frcb,stokesAD.frc),
            DuplicatedNoNeed(Cvb,stokesAD.Cv),
            DuplicatedNoNeed(Ccb,stokesAD.Cc))

        vertex2center!(stokesAD.G, stokesAD.Gv)
        stokesAD.G .+= stokesAD.Gc
        vertex2center!(stokesAD.fr, stokesAD.frv)
        stokesAD.fr .+= stokesAD.frc
        vertex2center!(stokesAD.C, stokesAD.Cv)
        stokesAD.C .+= stokesAD.Cc

        return ηb, ρb

end
