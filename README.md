i# Lamb Shift Numerical Evaluation (Spacetime Algebra and QED)

This repository contains Python code to numerically evaluate the Lamb shift in hydrogen using both standard Quantum Electrodynamics (QED) and an alternative Spacetime Algebra (STA) framework.

The numerical integration routines compute:
- Inner radial integrals \( I_{n\kappa}(k) \) for the \( 2S_{1/2} \) and \( 2P_{1/2} \) states.
- Outer \(k\)-integrals \(\delta E_{n\kappa}\) to estimate energy corrections.

## Directory Structure
- `lamb_shift.py`: Python routines for computing the Lamb shift integrals.
- `plots/`: Precomputed plots of the integrals.
- `data/`: CSV tables with numerical results.

## Requirements
See `requirements.txt`. Install dependencies with:


## Key Fixes
Included ħc scaling factor in the outer k-integral to correct units; normalized Dirac radial functions over [0, 5a₀]. Final Lamb shift agrees with experiment (~3.79 eV).
