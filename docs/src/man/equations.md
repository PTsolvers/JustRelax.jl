# Pseudo-transient iterative method
The pseudo-transient method consists in augmenting the right-hand-side of the target PDE with a pseudo-time derivative (where $\psi$ is the pseudo-time) of the primary variables. We then solve the resulting system of equations with an iterative method. The pseudo-time derivative is then gradually reduced, until the original PDE is solved and the changes in the primary variables are below a preset tolerance.  

## Heat diffusion
The pseudo-transient heat-diffusion equation is:

$\widetilde{\rho}\frac{\partial T}{\partial \psi} + \rho C_p \frac{\partial T}{\partial t} = \nabla(K\nabla T) = -\nabla q$

We use a second order pseudo-transient scheme were continuation is also done on the flux, so that:

$\widetilde{\theta}\frac{\partial q}{\partial \psi} + q  = -K\nabla T$

## Stokes equations

 For example, the pseudo-transient formulation of the Stokes equations yields:

$\widetilde{\rho}\frac{\partial \boldsymbol{u}}{\partial \psi} + \nabla\cdot\boldsymbol{\tau} - \nabla p = \boldsymbol{f}$

$\frac{1}{\widetilde{K}}\frac{\partial p}{\partial \psi} + \nabla\cdot\boldsymbol{v} = \beta \frac{\partial p}{\partial t} + \alpha \frac{\partial T}{\partial t}$


## Constitutive equations
A pseudo-transient continuation is also done on the constitutive law:

$\frac{1}{2\widetilde{G}} \frac{\partial\boldsymbol{\tau}}{\partial\psi}+ \frac{1}{2G}\frac{D\boldsymbol{\tau}}{Dt} + \frac{\boldsymbol{\tau}}{2\eta} = \dot{\boldsymbol{\varepsilon}}$

where the wide tile denotes the effective damping coefficients and $\psi$ is the pseudo-time step. These are defined as in [Räss et al. (2022)](https://gmd.copernicus.org/articles/15/5757/2022/):

$\widetilde{\rho} = Re\frac{\eta}{\widetilde{V}L}, \qquad \widetilde{G} = \frac{\widetilde{\rho} \widetilde{V}^2}{r+2}, \qquad \widetilde{K} = r \widetilde{G}$

and

$\widetilde{V} = \sqrt{ \frac{\widetilde{K} +2\widetilde{G}}{\widetilde{\rho}}}, \qquad r = \frac{\widetilde{K}}{\widetilde{G}}, \qquad Re = \frac{\widetilde{\rho}\widetilde{V}L}{\eta}$

where the P-wave $\widetilde{V}=V_p$ is the characteristic velocity scale for Stokes, and $Re$ is the Reynolds number. 
