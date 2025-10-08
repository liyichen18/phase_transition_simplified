"""
File: phase_field_pressure_distribution.py
Description: Visualization of phase-field profile, its gradient, and pressure distribution 
             across a diffuse interface using an analytical solution.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
epsilon = 1.0          # Interface thickness parameter
lambda_psi = 1.0       # Interfacial energy coefficient
rho = 1.0              # Density
p0 = 1.0               # Reference pressure

# x-axis domain
x = np.linspace(-6, 6, 400)

# Phase-field solution: tanh-type interface (transition from 0 to 1)
psi = 0.5 * (1 + np.tanh(x / (np.sqrt(2) * epsilon)))

# Gradient of phase-field dpsi/dx
dpsi_dx = (1/(2*np.sqrt(2)*epsilon)) * (1/np.cosh(x/(np.sqrt(2)*epsilon)))**2

# Pressure distribution: p(x) = p0 - rho * lambda_psi * (dpsi/dx)^2
p = p0 - rho * lambda_psi * dpsi_dx**2

# Plotting
plt.figure(figsize=(8,6))

plt.plot(x, psi, color="green", label=r"$\psi(x)$")
plt.plot(x, dpsi_dx, color="red", label=r"$\frac{d\psi}{dx}$")
plt.plot(x, p, color="purple", label=r"$p(x)$")

plt.axhline(p0, color="black", linestyle="--", linewidth=0.8, label=r"$p_0$")

plt.xlabel("x")
plt.ylabel("Value")
plt.title("Phase Field, Gradient, and Pressure Distribution")
plt.legend()
plt.grid(True)
plt.show()
