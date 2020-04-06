# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparison to theoretical model of squeezing with Gaussian local oscillator
# Squeezing of pulses in a nonlinear interferometer (Shirasaki, Haus)

# %%
# cd ..

# %%
import numpy as np
import matplotlib.pyplot as plt

try:
    from nonlinearmedium import Chi3
    print("using C++ implementation")
except:
    from NonlinearMedium import Chi3
    print("using Python implementation")

from NonlinearHelper import *

# %%
# %matplotlib notebook
plt.rcParams['figure.figsize'] = [9, 6]

# %% [markdown]
# # Squeezing and Local Oscillator using a Gaussian pulse

# %%
phis = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 2, 3]
variance1 = np.zeros(len(phis))
variance2 = np.zeros(len(phis))
haus = np.zeros(len(phis))

for i, phi in enumerate(phis):
    fiberH = Chi3(relativeLength=phi,
                  nlLength=1,
                  dispLength=np.inf,
                  beta2=-1,
                  pulseType=0,
                  tPrecision=512, zPrecision=100)

    greenC, greenS = fiberH.computeGreensFunction()

    Z = calcQuadratureGreens(greenC, greenS)

    C = calcCovarianceMtx(Z, tol=1e-2)

    variances = calcLOSqueezing(C, fiberH.pumpTime[-1], tol=1e-2)
    
    variance1[i], variance2[i] = variances
    haus[i] = 1 + 2 * phi**2 / np.sqrt(3) - (2 * phi**3 / np.sqrt(3) + np.sqrt(2) * phi) / np.sqrt(1 + phi**2)

# %%
plt.figure()
plt.plot(phis, 10 * np.log10(variance1), "o-", label="squeezed variance")
plt.plot(phis, 10 * np.log10(variance2), "o-", label="anti-squeezed variance")
plt.plot(phis, 10 * np.log10(np.sqrt(variance1 * variance2)), "o-", label="uncertainty product")
plt.plot(phis, 10 * np.log10(haus), "o-", label="Haus' Model")
plt.xlabel("NL Phase")
plt.ylabel("Noise Reduction dB")
plt.legend();
