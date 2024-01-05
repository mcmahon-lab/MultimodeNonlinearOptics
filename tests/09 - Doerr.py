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
# # Comparison to theoretical model of squeezing with soliton local oscillator
# Simulation of pulsed squeezing in optical fiber with chromatic dispersion
# (Doerr, Shirasaki, Khatri)

# %%
# cd ..

# %%
import numpy as np
import matplotlib.pyplot as plt

from nonlinearmedium import Chi3

from multimode import calcQuadratureGreens, calcCovarianceMtx, calcLOSqueezing

# %%
# %matplotlib notebook
plt.rcParams["figure.figsize"] = [9, 6]

# %% [markdown]
# Comparing to Figure 2 (while the shapes match, there appears to be a scaling issue)

# %%
lengths = [0.5, 1.0, 1.5, 2.0] # units of nonlinear lengths
params = [(25/20, -1), (1, -1), (15/20, -1), (10/20, -1), (5/20, -1), (1, 0),
          (5/20, 1), (10/20, 1), (15/20, 1)]

variances1 = np.zeros((len(lengths), len(params)))
variances2 = np.zeros((len(lengths), len(params)))

for i, param in enumerate(params):
    for j, length in enumerate(lengths):
        fiberD = Chi3(relativeLength=length * param[0],
                      nlLength=param[0],
                      beta2=param[1],
                      pulseType=1,
                      tPrecision=512, zPrecision=int(100 / min(1, length)))

        greenC, greenS = fiberD.computeGreensFunction(nThreads=4)

        Z = calcQuadratureGreens(greenC, greenS)
        C = calcCovarianceMtx(Z, tol=np.inf)

        variances = calcLOSqueezing(C, fiberD.pumpTime[-1], tol=np.inf)
        variances1[j, i], variances2[j, i] = variances

# %%
pms = np.array(params)

plt.figure()
for j, length in enumerate(lengths):
    plt.plot(pms[:, 1] * pms[:, 0], 10 * np.log10(variances1[j]), "-", label="L=%.1f"%length)
plt.title("Noise Reduction")
plt.xlabel("Dispersion")
plt.ylabel("Relative Variance dB")
plt.legend()

plt.figure()
for j, length in enumerate(lengths):
    plt.plot(pms[:, 1] * pms[:, 0], 10 * np.log10(variances2[j]), "-", label="L=%.1f"%length)
plt.title("Noise Increase")
plt.xlabel("Dispersion")
plt.ylabel("Relative Variance dB")
plt.legend()

plt.figure()
for j, length in enumerate(lengths):
    plt.plot(pms[:, 1] * pms[:, 0], 10 * np.log10(np.sqrt(variances1[j] * variances2[j])),
             "-", label="L=%.1f"%length)
plt.title("Uncertainty Product")
plt.xlabel("Dispersion")
plt.ylabel("Relative Uncertainty dB")
plt.legend();
