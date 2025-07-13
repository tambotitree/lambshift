import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
from scipy.constants import c, hbar, m_e, e, epsilon_0, alpha as alpha_fs

# Constants
alpha = alpha_fs
Z = 1  # Hydrogen
a0 = 5.29177210903e-11  # Bohr radius (m)
rz = 3.8616e-13  # zitterbewegung radius (m)
r_p = 0.8409e-15  # Proton charge radius (CODATA 2018) in meters

# Modified Coulomb potential to include finite nuclear size
def V_modified(r):
    if r > r_p:
        return -e**2 / (4 * np.pi * epsilon_0 * r)
    else:
        return -e**2 / (4 * np.pi * epsilon_0 * r_p) * (1.5 - (r**2) / (2 * r_p**2))

# Dirac relativistic energy
def E_dirac(n, kappa):
    gamma = np.sqrt(1 - (Z * alpha)**2)
    return m_e * c**2 * (1 + (Z * alpha)**2 / ((n - abs(kappa) + gamma)**2))**-0.5

# Dirac radial functions (unnormalized), using modified potential for r < r_p
def G_2S1(r):
    rho = 2 * Z * r / a0
    prefactor = (2 * Z / a0)**1.5
    energy_factor = np.sqrt(m_e * c**2 - E_dirac(2, -1)) / np.sqrt(2 * m_e * c**2)
    V_corr = V_modified(r) / (m_e * c**2)  # Normalize potential to dimensionless form
    return prefactor * rho * np.exp(-rho/2) * (1 - rho/2 + V_corr) * energy_factor

def F_2S1(r):
    rho = 2 * Z * r / a0
    prefactor = (2 * Z / a0)**1.5
    energy_factor = np.sqrt(m_e * c**2 + E_dirac(2, -1)) / np.sqrt(2 * m_e * c**2)
    V_corr = V_modified(r) / (m_e * c**2)
    return prefactor * rho * np.exp(-rho/2) * (1 - rho/2 + V_corr) * energy_factor

# Normalize radial functions
def normalize(G_func, F_func):
    r_min = 1e-15
    r_max = 5 * a0  # go out to 5 Bohr radii
    integrand = lambda r: r**2 * (G_func(r)**2 + F_func(r)**2)
    norm_factor, _ = quad(integrand, r_min, r_max, limit=500, epsabs=1e-6, epsrel=1e-6)
    norm_factor = np.sqrt(norm_factor)
    return lambda r: G_func(r)/norm_factor, lambda r: F_func(r)/norm_factor

# Inner radial integral
def I_nk(k, G_func, F_func):
    r_min = 1e-15
    r_max = 5 * a0
    integrand = lambda r: r**2 * (G_func(r)**2 + F_func(r)**2) * np.sinc(k * r / np.pi)
    result, _ = quad(integrand, r_min, r_max, limit=500, epsabs=1e-6, epsrel=1e-6)
    return result

# Outer integral debug
def delta_E(G_func, F_func):
    k_min = 0
    k_max = 1e10  # cap k for debug
    def integrand(k):
        value = I_nk(k, G_func, F_func)
        print(f"k={k:.2e}, I_nk={value:.3e}")
        return value
    result, _ = quad(integrand, k_min, k_max, limit=200, epsabs=1e-6, epsrel=1e-6)
    energy_joules = (alpha / np.pi) * hbar * c * result
    energy_eV = energy_joules / e  # J → eV
    print(f"Partial energy shift (k_max={k_max:.1e}): {energy_eV:.6e} eV")
    return energy_eV

# Normalize with finite-size corrections
G_norm, F_norm = normalize(G_2S1, F_2S1)

# Diagnostic normalization check
norm_check = quad(lambda r: r**2 * (G_norm(r)**2 + F_norm(r)**2), 1e-15, 5*a0)[0]
print(f"Normalization integral: {norm_check:.6f}")

# Main
if __name__ == "__main__":
    print(f"Relativistic Dirac Energy (2S1/2): {E_dirac(2, -1)/e:.6f} eV")

    for k in [0.1, 1.0, 10.0]:
        val = I_nk(k, G_norm, F_norm)
        print(f"I_nk({k:.1f}) = {val:.6e}")

    shift = delta_E(G_norm, F_norm)
    print(f"Lamb shift correction (2S1/2) with finite proton size: {shift:.6f} eV")
