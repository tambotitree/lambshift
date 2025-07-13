import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Constants
alpha = 1 / 137.035999084  # fine-structure constant
rz = 3.8616e-13  # zitterbewegung radius (in meters)

# Example Dirac radial functions P(r), Q(r) (stub functions for demo)
def P_2S1(r):
    return np.exp(-r) * r  # Placeholder

def Q_2S1(r):
    return np.exp(-r) * (1 - r)  # Placeholder

# Inner radial integral
def I_nk(k, P_func, Q_func):
    integrand = lambda r: r**2 * (P_func(r)**2 + Q_func(r)**2) * np.sinc(k * r / np.pi)
    result, error = quad(integrand, 0, np.inf, limit=1000)
    return result

# Outer integral
def delta_E(P_func, Q_func):
    integrand = lambda k: I_nk(k, P_func, Q_func)
    result, error = quad(integrand, 0, 1/rz, limit=1000)
    return (alpha / np.pi) * result

# Compute example
energy_shift = delta_E(P_2S1, Q_2S1)
print(f"Lamb shift correction (2S_1/2): {energy_shift:.6e} eV")

# Plotting inner integral (example)
k_vals = np.linspace(0.01, 10, 100)
I_vals = [I_nk(k, P_2S1, Q_2S1) for k in k_vals]

plt.figure()
plt.plot(k_vals, I_vals)
plt.title("Inner Integral I_nk(k) for 2S1/2")
plt.xlabel("k (1/m)")
plt.ylabel("I_nk(k)")
plt.grid()
plt.savefig("plots/inner_integral_2S1_2P1.png")
