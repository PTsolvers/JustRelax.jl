The Accelerated Pseudo-Transient (APT) method consists in augmenting the right-hand-side of the target PDE with a pseudo-time derivative (where $\psi$ is the pseudo-time) of the primary variables. We then solve the resulting system of equations with an iterative method. The pseudo-time derivative is then gradually reduced, until the original PDE is solved and the changes in the primary variables are below a preset tolerance.

## Heat diffusion
The APT heat-diffusion equation is:

$\begin{align}
\widetilde{\rho}\frac{\partial T}{\partial \psi} + \rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (\kappa\nabla T) = -\nabla q
\end{align}$

We use a second order APT scheme were continuation is also done on the flux, so that:

$\begin{align}
\widetilde{\theta}\frac{\partial q}{\partial \psi} + q  = -\kappa\nabla T
\end{align}$

## Stokes equations

For example, the APT formulation of the Stokes equations yields:

$\begin{align}
\widetilde{\rho}\frac{\partial \boldsymbol{u}}{\partial \psi} + \nabla\cdot\boldsymbol{\tau} - \nabla p = \boldsymbol{f}
\end{align}$

$\begin{align}
\frac{1}{\widetilde{K}}\frac{\partial p}{\partial \psi} + \nabla\cdot\boldsymbol{v} = \beta \frac{\partial p}{\partial t} + \alpha \frac{\partial T}{\partial t}
\end{align}$

## Constitutive equations
A APT continuation is also done on the constitutive law:

$\begin{align}
\frac{1}{2\widetilde{G}} \frac{\partial\boldsymbol{\tau}}{\partial\psi}+ \frac{1}{2G}\frac{D\boldsymbol{\tau}}{Dt} + \frac{\boldsymbol{\tau}}{2\eta} = \dot{\boldsymbol{\varepsilon}}
\end{align}$

where the wide tile denotes the effective damping coefficients and $\psi$ is the pseudo-time step. These are defined as in [Räss et al. (2022)](https://gmd.copernicus.org/articles/15/5757/2022/):

$\begin{align}
\widetilde{\rho} = Re\frac{\eta}{\widetilde{V}L}, \qquad \widetilde{G} = \frac{\widetilde{\rho} \widetilde{V}^2}{r+2}, \qquad \widetilde{K} = r \widetilde{G}
\end{align}$

and

$\begin{align}
\widetilde{V} = \sqrt{ \frac{\widetilde{K} +2\widetilde{G}}{\widetilde{\rho}}}, \qquad r = \frac{\widetilde{K}}{\widetilde{G}}, \qquad Re = \frac{\widetilde{\rho}\widetilde{V}L}{\eta}
\end{align}$

where the P-wave $\widetilde{V}=V_p$ is the characteristic velocity scale for Stokes, and $Re$ is the Reynolds number.

<!--
### Physical parameters

| Symbol                           | Parameter              |
| :------------------------------- | :--------------------: |
| $T$                              | Temperature            |
| $q$                              | Flux                   |
| $\boldsymbol{\tau}$              | Deviatoric stress      |
| $\dot{\boldsymbol{\varepsilon}}$ | Deviatoric strain rate |
| $\boldsymbol{u}$                 | Velocity               |
| $\boldsymbol{f}$                 | External forces        |
| $P$                              | Pressure               |
| $\eta$                           | Viscosity              |
| $\rho$                           | Density                |
| $\beta$                          | Compressibility        |
| $G$                              | Shear modulus          |
| $\alpha$                         | Thermal expansivity    |
| $C_p$                            | Heat capacity          |
| $\kappa$                         | Heat conductivity      |
-->

### Pseudo-transient parameters

| Symbol               | Parameter                     |
| :------------------- | :---------------------------: |
| $\psi$                | Pseudo time step              |
| $\widetilde{K}$      | Pseudo bulk modulus           |
| $\widetilde{G}$      | Pseudo shear modulus          |
| $\widetilde{V}$      | Characteristic velocity scale |
| $\widetilde{\rho}$   | Pseudo density                |
| $\widetilde{\theta}$ | Relaxation time               |
| $Re$                 | Reynolds number               |
