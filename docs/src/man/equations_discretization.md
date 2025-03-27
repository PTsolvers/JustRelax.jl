# Spatial discretization of the governing equations

We discretize both the Stokes and heat diffusion equations using a Finite Differences approach on a staggered grid (ref Taras book here).

## Heat diffusion
![Staggered Temperature Grid](../assets/temp_stag2D.png)


## Stokes equations
![Staggered Velocity Grid](../assets/stokes_stag2D.png)
where dotted lines represent the velocity ghost nodes.

# Time and pseudo-time discretization of the APT equations

We discretize both the Stokes and heat diffusion equations using a Finite Differences approach on a staggered grid (ref Taras book here).

## Heat diffusion
$\begin{align}
\widetilde{\theta}\frac{q^{n+1}_{x} - q^{n}_{x}}{\Delta \psi} + q^{n+1}_{x} &= -\kappa\frac{T^n + T^n}{\Delta x} \\
\widetilde{\theta}\frac{q^{n+1}_{y} - q^{n}_{y}}{\Delta \psi} + q^{n+1}_{y} &= -\kappa\frac{T^n + T^n}{\Delta x} \\
\widetilde{\rho}\frac{T^{n+1} + T^n}{\Delta \psi} + \rho C_p \frac{ T^{n+1} + T^t}{\Delta t} &=
-\left(\frac{\partial q_{x}}{\partial x} + \frac{\partial q_{y}}{\partial y}\right)
\end{align}$

Upon convergence we recover

$\begin{align}
T^{t+\Delta t}     = T^{n+1}      \\
q^{t+\Delta t}_{x} = q^{n+1}_{x}  \\
q^{t+\Delta t}_{y} = q^{n+1}_{y}  \\
\end{align}$

## Stokes equations

### Conservation of momentum

$\begin{align}
\widetilde{\rho}\frac{u^{n+1}_x - u^n_x}{\Delta\psi} + \nabla\cdot\boldsymbol{\tau} -
\frac{p^{n+1} - p^n}{\Delta x} =
0 \\
\end{align}$

$\begin{align}
\widetilde{\rho}\frac{u^{n+1}_y - u^n_y}{\Delta\psi} + \nabla\cdot\boldsymbol{\tau} -
\frac{p^{n+1} - p^n}{\Delta y} =
\rho g_y \\
\end{align}$

### Conservation of mass

$\begin{align}
\frac{1}{\widetilde{K}}\frac{p^{n+1} - p^{n}}{\Delta\psi} +
\left(\frac{u^{n+1}_x - u^n_x}{\Delta x} + \frac{u^{n+1}_y - u^n_y}{\Delta y} \right) =
\frac{1}{\Delta t} \left( \beta (p^{t+\Delta t} - p^{t}) + \alpha (T^{t+\Delta t} - T^{t}) \right) \\
\end{align}$

### Constitutive equation

$\begin{align}
\frac{1}{2\widetilde{G}} \frac{\boldsymbol{\tau}^{n+1} - \boldsymbol{\tau}^n}{\Delta\psi} +
\frac{1}{2G}\frac{\boldsymbol{\tau}^{n+1} - \boldsymbol{\tau}^t}{\Delta t} +
\frac{\boldsymbol{\tau^{n+1}}}{2\eta} =
\dot{\boldsymbol{\varepsilon}}
\end{align}$
