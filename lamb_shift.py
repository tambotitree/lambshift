import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os

# Constants
alpha = 1 / 137.035999084  # fine-structure constant
hbar_c = 197.3269804e-15  # Planck * c in eV·m
rz = 3.8616e-13  # zitterbewegung radius (m)
eV = 1.602176634e-19  # J

# Create plots directory if missing
if not os.path.exists("plots"):
    os.makedirs("plots")

# Approximate Dirac-Coulomb radial functions for Hydrogen (simplified)
def P_2S1(r):
    a0 = 5.29177210903e-11  # Bohr radius in m
    return (1 / np.sqrt(a0**3)) * (1 - r/(2*a0)) * np.exp(-r/(2*a0))

def Q_2S1(r):
    a0 = 5.29177210903e-11  # Bohr radius in m
    return (1 / np.sqrt(a0**3)) * (r/(2*a0)) * np.exp(-r/(2*a0))

# Inner radial integral
def I_nk(k, P_func, Q_func):
    def integrand(r):
        return r**2 * (P_func(r)**2 + Q_func(r)**2) * np.sinc(k * r / np.pi)
    result, error = quad(integrand, 0, 1e-9, limit=500)
    return result

# Outer integral
def delta_E(P_func, Q_func):
    def integrand(k):
        return I_nk(k, P_func, Q_func)
    result, error = quad(integrand, 0, 1/rz, limit=500)
    energy_joules = (alpha / np.pi) * result
    energy_eV = energy_joules / eV
    return energy_eV

# Main program
if __name__ == "__main__":
    # Compute energy shift
    energy_shift = delta_E(P_2S1, Q_2S1)
    print(f"Lamb shift correction (2S_1/2): {energy_shift:.6f} eV")

    # Plot inner integral
    k_vals = np.linspace(0.01, 1e10, 100)
    I_vals = [I_nk(k, P_2S1, Q_2S1) for k in k_vals]

    plt.figure()
    plt.plot(k_vals, I_vals)
    plt.title("Inner Integral I_nk(k) for 2S1/2")
    plt.xlabel("k (1/m)")
    plt.ylabel("I_nk(k)")
    plt.grid()
    plt.savefig("plots/inner_integral_2S1_2P1.png")
