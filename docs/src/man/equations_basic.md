# Governing equations

## Stokes equations

Deformation of compressible viscous flow is described by the equations of conservation of momentum and mass:

$\begin{align}
\nabla\cdot\boldsymbol{\tau} - \nabla p = \boldsymbol{f}
\end{align}$

$\begin{align}
\nabla\cdot\boldsymbol{v} = \beta \frac{\partial p}{\partial t} + \alpha \frac{\partial T}{\partial t}
\end{align}$

where $\boldsymbol{\tau}$ is the deviatoric stress tensor, $p$ is pressure, $f$ is the external forces vector, $\boldsymbol{v}$ is the velocity vector, $\beta$ is the compressibility coefficient, $\alpha$ is the thermal expansivity coefficient and $T$ is temperature.

## Constitutive equation

To close the system of equations (1)-(2), we further need the constitutive relationship between stress and deformation. In its simplest linear form this is:

$\begin{align}
\boldsymbol{\tau} = 2 \eta \boldsymbol{\dot\varepsilon}
\end{align}$

where $\eta$ is the shear viscosity and  $\boldsymbol{\dot\varepsilon}$ is the deviatoric strain tensor.
## Heat diffusion
The pseudo-transient heat-diffusion equation is:

$\begin{align}
\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (\kappa\nabla T) + \boldsymbol{\tau}:\boldsymbol{\dot\varepsilon} + \alpha T(\boldsymbol{v} \cdot \nabla P) + H
\end{align}$

where $\rho$ is density, $C_p$ is specific heat capacity, $\kappa$ is thermal conductivity, $T$ is temperature $\boldsymbol{\tau}:\boldsymbol{\dot\varepsilon}$ is the energy dissipated by viscous deformation (shear heating), $\alpha T(\boldsymbol{v} \cdot \nabla P)$ is adiabatic heating, and $H$ is the sum any other source term, such as radiogenic heat production.
