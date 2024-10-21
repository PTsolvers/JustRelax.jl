# unpack

_di = inv.(di)
(; ϵ, r, θ_dτ, ηdτ) = pt_stokes
(; η, η_vep) = stokes.viscosity
ni = size(stokes.P)

# ~preconditioner
ητ = deepcopy(η)
# @hide_communication b_width begin # communication/computation overlap
JR.compute_maxloc!(ητ, η; window=(1, 1))
# update_halo!(ητ)
# end

iterMax          =        50e3
nout             =         1e3
viscosity_cutoff = (-Inf, Inf)
free_surface     =       false

# errors
err = 2 * ϵ
iter = 0
err_evo1 = Float64[]
err_evo2 = Float64[]
norm_Rx = Float64[]
norm_Ry = Float64[]
norm_∇V = Float64[]
sizehint!(norm_Rx, Int(iterMax))
sizehint!(norm_Ry, Int(iterMax))
sizehint!(norm_∇V, Int(iterMax))
sizehint!(err_evo1, Int(iterMax))
sizehint!(err_evo2, Int(iterMax))

# solver loop
@copy stokes.P0 stokes.P
wtime0 = 0.0
relλ = 0.2
θ = deepcopy(stokes.P)
λ = @zeros(ni...)
λv = @zeros(ni .+ 1...)
η0 = deepcopy(η)
do_visc = true

for Aij in @tensor_center(stokes.ε_pl)
    Aij .= 0.0
end
Vx_on_Vy = @zeros(size(stokes.V.Vy))

# compute buoyancy forces and viscosity
compute_ρg!(ρg[end], phase_ratios, rheology, args)
compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

while iter ≤ iterMax
    iterMin < iter && err < ϵ && break

    wtime0 += @elapsed begin
        JR.compute_maxloc!(ητ, η; window=(1, 1))
        # update_halo!(ητ)

        @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes), ϕ, _di)

        @parallel (@idx ni)  compute_P!(
            θ, 
            stokes.P0, 
            stokes.R.RP, 
            stokes.∇V, 
            ητ, 
            rheology,
            phase_ratios.center,
            ϕ,
            dt,
            pt_stokes.r,
            pt_stokes.θ_dτ
        )

        JR.update_ρg!(ρg[2], phase_ratios, rheology, args)

        @parallel (@idx ni .+ 1) JR.compute_strain_rate!(
            @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
        )

        # if rem(iter, nout) == 0
        #     @copy η0 η
        # end
        # if do_visc
        # update_viscosity!(
        #     stokes,
        #     phase_ratios,
        #     args,
        #     rheology,
        #     viscosity_cutoff;
        #     relaxation=viscosity_relaxation,
        # )
        # end

        @parallel (@idx ni .+ 1) update_stresses_center_vertex_ps!(
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
            ϕ,
        )
        # update_halo!(stokes.τ.xy)

        @parallel (1:(size(stokes.V.Vy, 1) - 2), 1:size(stokes.V.Vy, 2)) JR.interp_Vx∂ρ∂x_on_Vy!(
            Vx_on_Vy, stokes.V.Vx, ρg[2], _di[1]
        )

        # @hide_communication b_width begin # communication/computation overlap
            @parallel compute_V!(
                @velocity(stokes)...,
                Vx_on_Vy,
                stokes.P,
                @stress(stokes)...,
                ηdτ,
                ρg...,
                ητ,
                ϕ.Vx,
                ϕ.Vy,
                _di...,
                dt * free_surface,
            )
            # apply boundary conditions
            # velocity2displacement!(stokes, dt)
            free_surface_bcs!(stokes, flow_bcs, η, rheology, phase_ratios, dt, di)
            flow_bcs!(stokes, flow_bcs)
        #     update_halo!(@velocity(stokes)...)
        # end
    end

    iter += 1

    if iter % nout == 0 && iter > 1
        # er_η = norm_mpi(@.(log10(η) - log10(η0)))
        # er_η < 1e-3 && (do_visc = false)
        @parallel (@idx ni) compute_Res!(
            stokes.R.Rx,
            stokes.R.Ry,
            @velocity(stokes)...,
            Vx_on_Vy,
            stokes.P,
            @stress(stokes)...,
            ρg...,
            ϕ.Vx,
            ϕ.Vy,
            _di...,
            dt * free_surface,
        )
        # errs = maximum_mpi.((abs.(stokes.R.Rx), abs.(stokes.R.Ry), abs.(stokes.R.RP)))
        errs = (
            norm(@views stokes.R.Rx[2:(end - 1), 2:(end - 1)]) / length(stokes.R.Rx),
            norm(@views stokes.R.Ry[2:(end - 1), 2:(end - 1)]) / length(stokes.R.Ry),
            norm(stokes.R.RP) / length(stokes.R.RP),
        )
        push!(norm_Rx, errs[1])
        push!(norm_Ry, errs[2])
        push!(norm_∇V, errs[3])
        err = maximumi(errs)
        push!(err_evo1, err)
        push!(err_evo2, iter)

        if igg.me == 0 #&& ((verbose && err > ϵ) || iter == iterMax)
            @printf(
                "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                iter,
                err,
                norm_Rx[end],
                norm_Ry[end],
                norm_∇V[end]
            )
        end
        isnan(err) && error("NaN(s)")
    end

    if igg.me == 0 && err ≤ ϵ
        println("Pseudo-transient iterations converged in $iter iterations")
    end
end

# # compute vorticity
# @parallel (@idx ni .+ 1) compute_vorticity!(
#     stokes.ω.xy, @velocity(stokes)..., inv.(di)...
# )

# # accumulate plastic strain tensor
# @parallel (@idx ni) accumulate_tensor!(stokes.EII_pl, @tensor_center(stokes.ε_pl), dt)


update_rock_ratio!(ϕ, phase_ratios, air_phase)

heatmap(stokes.V.Vx)
heatmap(stokes.V.Vy)
heatmap(stokes.P)
heatmap(stokes.τ.xx)
heatmap(stokes.τ.yy)
heatmap(stokes.τ.xy)


heatmap(log10.(abs.(stokes.R.Rx)))
heatmap(log10.(abs.(stokes.R.Ry)))
heatmap(log10.(abs.(stokes.V.Vy)))
heatmap(ϕ.Vy)
heatmap(ϕ.Vx)
heatmap(ϕ.vertex)
heatmap(ϕ.center)


f,ax,h = heatmap(ϕ.vertex)
Colorbar(f[1,2], h);f

f,ax,h = heatmap(ϕ.center)
Colorbar(f[1,2], h);f

@parallel compute_V!(
    @velocity(stokes)...,
    stokes.R.Rx,
    stokes.R.Ry,
    stokes.P,
    @stress(stokes)...,
    ηdτ,
    ρg...,
    ητ,
    ϕ.Vx,
    ϕ.Vy,
    _di...,
)

extrema(stokes.R.Ry)
extrema(stokes.V.Vy)

f,ax,h = heatmap(stokes.V.Vy)
Colorbar(f[1,2], h, label="Vy")

-d_ya(P, ϕ.center) + d_ya(τyy, ϕ.center) + d_xi(τxy, ϕ.vertex) - av_ya(ρgy)

d_yi(A, ϕ) = _d_yi(A, ϕ, _dy, i, j)
d_ya(A, ϕ) = _d_ya(A, ϕ, _dy, i, j)
_dx, _dy = inv.(di)

a = [_d_yi(stokes.τ.xy, ϕ.vertex, _dy, i, j) for i in 1:nx, j in 1:ny-1]
heatmap(a)

b = [_d_ya(stokes.P, ϕ.center, _dy, i, j) for i in 1:nx, j in 1:ny-1]
f,ax,h = heatmap(b)
Colorbar(f[1,2], h, label="Vy")

c = [_d_ya(stokes.τ.yy, ϕ.center, _dy, i, j) for i in 1:nx, j in 1:ny-1]
heatmap(c)
d = @. a-b+c

f,ax,h=heatmap(d)
Colorbar(f[1,2], h, label="Vy")

lines(stokes.P[1,:])
lines(stokes.V.Vy[1,:])

v = [isvalid_c(ϕ, i, j) for i in 1:nx, j in 1:ny]
heatmap(xci..., v)

v = [isvalid_v(ϕ, i, j) for i in 1:nx, j in 1:ny]
heatmap(xvi..., v)


vx = [isvalid_vx(ϕ, i, j) for i in 1:nx+1, j in 1:ny]
heatmap(vx)

vy = [isvalid_vy(ϕ, i, j) for i in 1:nx, j in 1:ny+1]
heatmap(vy)

heatmap(
    xci[1].*1e-3, xci[2].*1e-3, 
    Array([argmax(p) for p in phase_ratios.center]), 
    colormap = :grayC)
px, py = particles

heatmap(
    xvi[1].*1e-3, xvi[2].*1e-3, 
    Array([argmax(p) for p in phase_ratios.vertex]), 
    colormap = :grayC)


heatmap(xvi..., ϕ.vertex)
# heatmap(xci..., ϕ.center)
heatmap(xci..., P)
scatter!(px.data[:], py.data[:], color=pPhases.data[:], markersize=5.0, marker=:circle, colormap=:grayC)
scatter(px.data[:], py.data[:], color=pPhases.data[:], markersize=5.0, marker=:circle, colormap=:grayC)

P = copy(stokes.P)
P = copy(ϕ.vertex)
P[iszero.(P)] .= NaN
heatmap(xci..., P)

id = particles.index.data[:]
px, py = particles.coords    
scatter!(px.data[:], py.data[:], color=pPhases.data[:], markersize=5.0, marker=:circle)
scatter(px.data[:][id], py.data[:][id], color=pPhases.data[:][id], markersize=5.0, marker=:circle)
