# 2D variational Stokes

The variational Stokes solver is a matrix-free, volume-fraction discretization
of the embedded liquid domain. It is intended for free surfaces represented by
a `JustPIC.MarkerChain`, where cells and staggered-grid locations may be only
partly occupied by liquid. The implementation follows the weighted formulation
of Larionov, Batty and Bridson (2017), with the physical geodynamic body-force
terms and JustRelax's pseudo-transient iteration added separately.

The implementation described here is currently 2D. The governing equations and
the weighted operators are implemented in
[`src/variational_stokes/`](https://github.com/PTsolvers/JustRelax.jl/tree/main/src/variational_stokes).

The equations below distinguish the mathematical target from the current
matrix-free implementation. Pressure null-space elimination, liquid-volume
weights, bounded active-face mass, and the implicit density-gradient correction
are implemented. The fully coupled weighted continuity/momentum operator shown
in the eliminated system remains a derivation target; the current continuity
stencil uses the unweighted staggered velocity difference together with the
binary active-cell policy.

## Continuous problem

Let \(\Omega_L\) be the liquid part of the computational domain. For a
Newtonian, incompressible liquid, the steady equations are

\[
\begin{aligned}
 -\nabla\!\cdot\boldsymbol{\tau} + \nabla p &= \boldsymbol{f},\\
 \nabla\!\cdot\boldsymbol{u} &= 0,\\
 \boldsymbol{\tau} &= 2\eta\,\boldsymbol{\varepsilon}(\boldsymbol{u}),\\
 \boldsymbol{\varepsilon}(\boldsymbol{u})
 &= \tfrac12\left(\nabla\boldsymbol{u}+\nabla\boldsymbol{u}^{T}\right).
\end{aligned}
\]

Here \(\boldsymbol{f}\) includes the buoyancy force \(\rho\boldsymbol{g}\).
The Larionov et al. formulation is written for unsteady Stokes without a body
force; gravity is therefore an additional geodynamic term in this package.

## Weighted variational formulation

The embedded-boundary discretization is obtained by restricting the weak form
to \(\Omega_L\). After discretization, the liquid volume is represented by
diagonal weights:

\[
 W_L^u=\operatorname{diag}(w^u_f),\qquad
 W_L^p=\operatorname{diag}(w^p_c),\qquad
 W_L^\tau=\operatorname{diag}(w^\tau_v).
\]

The subscripts identify the discrete locations:

| Weight | `RockRatio` field | Location |
| --- | --- | --- |
| \(W_L^p\) | `ϕ.center` | cell-centred pressure and normal stresses |
| \(W_L^\tau\) | `ϕ.vertex` | vertex shear stress |
| \(W_L^u\) | `ϕ.Vx`, `ϕ.Vy` | staggered face velocities |

With \(G\) the discrete gradient and \(D\) the discrete divergence, the
mathematical target weighted saddle-point system has the schematic form

\[
\begin{bmatrix}
 A^T W_L^\tau A + P W_L^u & G^T W_L^p\\
 W_L^p G & 0
\end{bmatrix}
\begin{bmatrix}\boldsymbol{u}\\p\end{bmatrix}
=
\begin{bmatrix}\boldsymbol{b}_u\\0\end{bmatrix}.
\]

The exact block factors depend on the staggered-grid constitutive operator,
but the important rule is that a liquid weight belongs to the location of the
unknown it measures. In particular, \(W_L^u\) must not be replaced by a cell
weight when assembling a face-velocity row.

Eliminating the velocity gives the pressure system described by Larionov et al.:

\[
\left[
 W_L^p G\,P^{-1}(W_L^u)^{-1}W_F^u G^T W_L^p
 + A_p
\right]p = b_p,
\]

where \(P\) is the diagonal velocity preconditioner, \(W_F^u\) denotes any
solid/fluid weight, and \(A_p\) contains compressibility or pressure
regularization. JustRelax does not explicitly assemble this matrix; its
operators apply the same weighted blocks matrix-free.

## Staggered-grid operators

For a cell \((i,j)\), the divergence is evaluated from the surrounding face
velocities. A liquid cell is active only if the pressure cell and the required
velocity faces are connected to liquid degrees of freedom. In the current 2D
implementation this reduced-space rule is represented by `isvalid_c`,
`isvalid_vx`, and `isvalid_vy` in
[`mask.jl`](https://github.com/PTsolvers/JustRelax.jl/blob/main/src/variational_stokes/mask.jl).

The normal strain components are stored at cell centres and the shear component
at vertices. The active normal components are weighted by the centre fraction:

\[
\varepsilon_{xx,ij}
 = w^p_{ij}\left(\partial_x u_x - \tfrac13\nabla\!\cdot u\right),
\qquad
\varepsilon_{yy,ij}
 = w^p_{ij}\left(\partial_y u_y - \tfrac13\nabla\!\cdot u\right),
\]

while the shear component is weighted by the vertex fraction:

\[
\varepsilon_{xy,ij}
 = w^\tau_{ij}\,\tfrac12
 \left(\partial_y u_x+\partial_x u_y\right).
\]

Pressure and stress derivatives use the corresponding centre or vertex weight
in their masked finite-difference stencils. Inactive rows are set to zero and
are not allowed to contribute to the pressure or velocity residual.

## Pseudo-transient iteration

JustRelax solves a matrix-free approximation to this weighted system with
accelerated pseudo-transient iterations rather than a direct sparse solve. The
velocity update has the form

\[
 u_f^{k+1}=u_f^k+
 \frac{\eta_{d\tau}}{w_f^u\,\eta_{\tau,f}}R_{u,f},
\]

where \(R_{u,f}\) is the weighted momentum residual, \(\eta_{\tau,f}\) is the
local diagonal preconditioner, and \(\eta_{d\tau}\) is the pseudo-transient
coefficient. To avoid an unbounded inverse in nearly dry faces, the current
implementation uses

\[
 \widehat w_f^u=\max(w_f^u,0.1)
\]

in the diagonal update. A face with zero liquid volume is still inactive; the
floor is only a bounded preconditioning approximation for active sliver faces.

Pressure is updated from the active-cell divergence and the constitutive
material parameters. The pressure residual includes the centre liquid fraction,
but the current divergence stencil itself is not face-weighted; this is one of
the coupled-operator items still requiring re-derivation. For finite bulk and
shear moduli, the pressure kernel uses

\[
 R_p = -(p-p_0)(K\Delta t)^{-1}
       -w^p\nabla\!\cdot u
       +Q\Delta t^{-1},
\]

with the corresponding relaxation formula. Thermal expansion and melt terms,
when supplied, enter the same pressure kernel through the volumetric source.
Invalid pressure cells are explicitly set to zero rather than being left as
unconstrained null-space unknowns.

## Surface null-space elimination

The liquid weights are continuous fractions, but degree-of-freedom elimination
is binary. A zero-weight unknown is removed from the reduced system; a positive
weight remains active. For a cell-centred pressure unknown, the current 2D rule
requires the pressure cell and all four surrounding velocity faces to be active:

\`\`\`text
                 Vy[i, j+1]
                       o
                       |
        Vx[i, j]  o--- p[i,j] ---o  Vx[i+1, j]
                       |
                       o
                 Vy[i, j]

       p[i,j] is retained only when all four o faces have liquid weight > 0
\`\`\`

Thus a pressure cell cut away by the surface is not merely assigned a small
coefficient. Its pressure row and residual are eliminated (P = RP = 0), and
the associated disconnected velocity rows are also set to zero. A face with a
positive but small liquid fraction remains active; its preconditioner uses the
bounded mass \(\widehat w^u=\max(w^u,0.1)\). Replacing a small positive fraction
by zero changes the reduced operator, while dividing by the raw fraction makes
sliver faces numerically unstable.

The vertex shear degree of freedom follows the analogous connectivity rule. It
is retained only when its vertex fraction and the neighbouring locations needed
by the shear stencil are active. Consequently, the surface system is a reduced
staggered system, not a full-grid system with air material properties assigned
to the missing rows.

\`\`\`text
                 liquid surface
             ~~~~~~~~~~~~~~~~~~~~~
             x   x   o   o   x   x
                 ^       ^
          eliminated   retained
          pressure/    pressure/
          face rows    face rows
\`\`\`

The marker chain supplies the geometry used to compute these fractions. The
solver applies the reduced-space rule locally through isvalid_c, isvalid_v,
isvalid_vx, and isvalid_vy; no global sparse matrix containing air unknowns is
assembled.

## Density-gradient free-surface correction

When the solver keyword `free_surface` is enabled, the vertical momentum row
contains the density-gradient correction

\[
 c_f=u_y\,\partial_y(\rho g_y)\,\Delta t.
\]

This term is nonzero near a liquid/air density jump. Treating it explicitly is
unstable for geodynamic timesteps because \(\Delta t\) is a physical timestep,
not the small pseudo-time increment. The variational implementation therefore
incorporates it into the face-local diagonal:

\[
 u_y^{k+1}=u_y^k+
 \frac{\eta_{d\tau}\left(R_y^0+u_y^k\,\partial_y(\rho g_y)\Delta t\right)}
 {\widehat w_y^u\eta_{\tau,y}
  -\eta_{d\tau}\partial_y(\rho g_y)\Delta t},
\]

where \(R_y^0\) is the momentum residual without the correction. This is the
matrix-free equivalent of treating the correction implicitly in the local
face row, and it uses the same liquid face mass as the variational velocity
operator.

The marker chain remains the geometric free-surface representation. It is not
the same as the top-boundary condition in `VelocityBoundaryConditions`; users
should not enable both mechanisms accidentally.

## Marker-chain coupling

One physical timestep follows this order:

1. Advect particles and move them into their new cells.
2. Advect the marker chain with the same velocity field. The recommended
   free-surface path is the semi-Lagrangian update, which backtracks fixed
   surface vertices, limits steep slopes, conserves mean height, and rebuilds
   the chain. The fully Lagrangian alternative is described below.
3. Invalidate particles lying on the wrong side of the updated chain.
4. Replenish invalid particle slots from neighbouring valid particles.
5. Recompute particle phase ratios at centres, vertices, and velocity faces.
6. Recompute `RockRatio` from the marker chain.

The last two fields have different roles: phase ratios determine material
properties such as \(\rho\) and \(\eta\), while `RockRatio` supplies the liquid
weights in the variational operator. Replenishment must occur before phase-ratio
normalization; otherwise an empty particle cell can produce a zero-weight
normalization and hence `NaN` phase fractions. JustPIC's phase-aware
replenishment currently chooses replacement coordinates randomly inside
depleted quadrants and copies the nearest valid particle's phase. Increasing
particle count and avoiding an unnecessarily high replenishment threshold can
reduce the resulting sampling noise, but does not make the process deterministic.

For the recommended semi-Lagrangian surface update, use:

```julia
semilagrangian_advection_markerchain!(
    chain, RungeKutta2(), @velocity(stokes), grid_vxi, xvi, dt
)
```

The non-semi-Lagrangian alternative moves the chain markers directly and has a
different signature:

```julia
advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
```

Do not pass `xvi` to `advect_markerchain!`; that routine obtains the chain's
vertex grid from the `MarkerChain` itself. It also performs marker movement,
cell reassignment, resampling, topography reconstruction, and mean-height
correction internally.

## Minimal usage

```julia
ϕ = RockRatio(backend, ni)
compute_rock_fraction!(ϕ, chain, xvi, di)

solve_VariationalStokes!(
    stokes, pt_stokes, grid, flow_bcs, ρg, phase_ratios, ϕ,
    rheology, args, dt, igg;
    air_phase = 1,
    free_surface = true,
)
```

The same weighted operator is available with dynamic relaxation in 2D. Build
`DYREL` with the `RockRatio` and call `solve_VariationalDYREL!`; see
[DYREL](./DYREL.md#2d-variational-dyrel). The variational DYREL path reuses the
center, vertex, and face weights described here and is currently 2D-only.

For a marker-chain free surface, `air_phase` must identify the phase removed
from liquid material averages. Omitting it preserves API compatibility but does
not produce the intended air/liquid buoyancy or viscosity weighting.

## Reference

Larionov, A., Batty, C. and Bridson, R. (2017), *Variational Stokes: A Unified
Pressure-Viscosity Solver for Accurate Viscous Liquids*, ACM Transactions on
Graphics, 36(4), Article 101. See the
[publisher record and mathematical reference](https://doi.org/10.1145/3099564.3099569).
