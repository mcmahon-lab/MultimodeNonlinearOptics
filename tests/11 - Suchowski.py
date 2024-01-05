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
# # Adiabatic Frequency Conversion Simulation
# "Geometrical representation of sum frequency generation and adiabatic frequency conversion" Haim Suchowski, Dan Oron, Ady Arie, and Yaron Silberberg
# https://doi.org/10.1103/PhysRevA.78.063821
# Comparison to Figure 3.

# %%
# cd ..

# %%
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

from nonlinearmedium import Chi2SFG

from classical import calculateDispLength, calculateChi2NlLength, findFrameOfReference
from poling import linearPoling

# %%
# %matplotlib notebook
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [9, 6]

# %%
from materials import KTPz

# %%
c = 299792458 # m / s

# %%
temperatureKTP = 22.5

# %%
signalWav = 1537 # nm
pumpWav   = 1064 # nm
finalWav  = 628.9 # nm

pumpFreq   = 2 * np.pi * c / pumpWav   # 2pi GHz
signalFreq = 2 * np.pi * c / signalWav # 2pi GHz
finalFreq  = 2 * np.pi * c / finalWav  # 2pi GHz

# Bandwidth limited pulse time scale, FWHM
timeScale = 6e3 / (2 * np.log(1 + np.sqrt(2))) # ps

# %%
# Intensity, timescale and length
length    = 1.7e-2 # m

peakPower = 40e6 # W
pulseRad  = np.sqrt(1e-4 / np.pi) # 1 cm^-2

# %%
# Nonlinear Coefficient
d = 16 # pm / V

# Group Velocity Dispersion -- ps^2 / km
beta2s = 1e27 * KTPz.gvd(signalWav*1e-3, temperatureKTP)
beta2p = 1e27 * KTPz.gvd(pumpWav*1e-3,   temperatureKTP)
beta2f = 1e27 * KTPz.gvd(finalWav*1e-3,  temperatureKTP)

# Relative Group Velocity
ngs = KTPz.ng(signalWav*1e-3, temperatureKTP)
ngp = KTPz.ng(pumpWav*1e-3,   temperatureKTP)
ngf = KTPz.ng(finalWav*1e-3,  temperatureKTP)

# Index of refraction
indexS = KTPz.n(signalWav*1e-3, temperatureKTP)
indexP = KTPz.n(pumpWav*1e-3,   temperatureKTP)
indexF = KTPz.n(finalWav*1e-3,  temperatureKTP)

# TOD -- ps^3 / km
beta3s = 1e39 * KTPz.beta3(signalWav*1e-3, temperatureKTP)
beta3p = 1e39 * KTPz.beta3(pumpWav*1e-3,   temperatureKTP)
beta3f = 1e39 * KTPz.beta3(finalWav*1e-3,  temperatureKTP)

# %%
# Walk-off
beta1f, beta1p, beta1s = findFrameOfReference(ngf, ngp, ngs) # ps / km

# Phase velocity mismatch
diffBeta0sfg = 2 * np.pi * (indexP / pumpWav + indexS / signalWav - indexF / finalWav) * 1e12 # km^-1

# %%
# Characteristic lengths
DS  = calculateDispLength(beta2p, timeScale, pulseTypeFWHM=None)
NLo = calculateChi2NlLength(d, peakPower, pulseRad, indexP, indexS, signalFreq)
NLf = calculateChi2NlLength(d, peakPower, pulseRad, indexP, indexF, finalFreq)

# Normalized quantities
diffBeta0sfgN = diffBeta0sfg * timeScale**2 / abs(beta2p)

beta1sN = beta1s * timeScale / abs(beta2p)
beta1pN = beta1p * timeScale / abs(beta2p)
beta1fN = beta1f * timeScale / abs(beta2p)

beta2sN = beta2s / abs(beta2p)
beta2fN = beta2f / abs(beta2p)

beta3sN = beta3s / (abs(beta2p) * timeScale)
beta3pN = beta3p / (abs(beta2p) * timeScale)
beta3fN = beta3f / (abs(beta2p) * timeScale)

relLength = length / DS
relNlLength = NLf / DS
relNlLength2 = NLo / DS

afcParams = {"relativeLength": relLength,
             "beta2":  np.sign(beta2p),
             "beta2s": beta2fN,
             "beta2o": beta2sN,
             "beta3":  beta3pN,
             "beta3s": beta3fN,
             "beta3o": beta3sN,
             "beta1":  beta1pN,
             "beta1s": beta1fN,
             "beta1o": beta1sN,
             "diffBeta0":  diffBeta0sfgN,
             "nlLength": relNlLength,
             "nlLengthOrig": relNlLength2,
             }

amplitudeRatio = NLo / NLf

# %% [markdown]
# # Axes

# %%
nt = 2**9
tMax = 0.0012

# normalized
tau   = (2 * tMax / nt) * ifftshift(np.arange(-nt / 2, nt / 2))
omega = (-np.pi / tMax) *  fftshift(np.arange(-nt / 2, nt / 2))

wMax = np.max(omega)
angFreqMax = wMax / timeScale

# dimensionful
time    = tau * timeScale   # ps
angFreq = omega / timeScale # 2 pi THz
wavelengthP = 2 * np.pi * c / (1000 * angFreq + pumpFreq) # nm
wavelengthS = 2 * np.pi * c / (1000 * angFreq + signalFreq) # nm
wavelengthF = 2 * np.pi * c / (1000 * angFreq + finalFreq) # nm

# %% [markdown]
# # Pulses

# %% [markdown]
# ### Pump

# %%
pumpProfTemp = 1 / np.cosh(tau)

# %%
plt.figure()
plt.plot(fftshift(time), fftshift(pumpProfTemp)**2)
plt.title("Pump Time Intensity")
plt.xlabel("t / ps");

# %% [markdown]
# ### Signal Pulse

# %%
signalProfFreq = np.zeros(nt)
signalProfFreq[np.logical_and(wavelengthS >= 1470, wavelengthS <= 1610)] = 1
signalProfTime = ifft(signalProfFreq)

norm = np.sqrt(np.sum(np.abs(signalProfTime)**2))
signalProfTime /= norm
signalProfFreq /= norm

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.plot(fftshift(time), fftshift(np.abs(signalProfTime)**2))
plt.title("Signal Time Intensity")
plt.xlabel("t / ps");

ax = fig.add_subplot(1, 2, 2)
plt.plot(fftshift(wavelengthS), fftshift(np.abs(signalProfFreq)**2))
plt.title("Signal Wavelength Intensity")
plt.xlabel("$\lambda$ / nm");

# %% [markdown]
# # Poling

# %%
mindk = 2 * np.pi / (16.2e-6 / DS)
maxdk = 2 * np.pi / (14.6e-6 / DS)

# %%
tolerance = 1e-8 / DS # Assume domain sizes must be multiples of 0.01 um
convertPoling = linearPoling(mindk, maxdk, afcParams["relativeLength"], tolerance)

print("crystal length vs sum of domains: {:.4f} {:.4f}".format(afcParams["relativeLength"] * DS,
                                                               np.sum(convertPoling) * DS))
print("domain size range {:.2f} - {:.2f} um".format(convertPoling[0]  * DS * 1e6,
                                                    convertPoling[-2] * DS * 1e6))

# %% [markdown]
# # Simulation

# %%
params = {**afcParams,
          "tPrecision": nt, "tMax": tMax,
          "zPrecision": max(2000, int(3 * convertPoling.size
                                    * min(1, afcParams["nlLength"] / np.max(np.abs(pumpProfTemp[0])) / afcParams["relativeLength"]))),
          "poling": convertPoling,
          "customPump": pumpProfTemp / np.max(np.abs(pumpProfTemp[0])),
          "nlLength":     afcParams["nlLength"]     / np.max(np.abs(pumpProfTemp[0])),
          "nlLengthOrig": afcParams["nlLengthOrig"] / np.max(np.abs(pumpProfTemp[0])),
          }

# %%
crystal = Chi2SFG(**params)

# %%
print("Simulation size", crystal.pumpTime.shape,
      "NL Lengths {:0.2f}".format(params["relativeLength"] / params["nlLength"]))

# %%
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
spatialFreqs = np.abs(fftshift(fft(crystal.poling)))
plt.plot(2 * np.pi / (1e6 * length) * np.arange(-crystal.signalTime.shape[0] / 2, crystal.signalTime.shape[0] / 2),
       spatialFreqs / np.max(spatialFreqs))
plt.xlabel("$k_p ~ [\mu m^{-1}]$")
plt.ylabel("$\chi^{(2)}_{eff}(k_z) ~/~ \chi^{(2)}$")
plt.title("Poling frequency profiles")

plt.plot([-diffBeta0sfg * 1e-9, -diffBeta0sfg * 1e-9, diffBeta0sfg * 1e-9, diffBeta0sfg * 1e-9],
         [1, 0, 0, 1], label="SFG Central Mismatch")
plt.legend()

ax = fig.add_subplot(2, 1, 2)
plt.plot(length * np.arange(-crystal.signalTime.shape[0] / 2, crystal.signalTime.shape[0] / 2),
       crystal.poling)
plt.xlabel("$p ~ [mm]$")
plt.ylabel("$\chi^{(2)}_{eff}(z) ~/~ \chi^{(2)}$");

# %%
crystal.runPumpSimulation()

# %%
plt.figure()
crystal.runPumpSimulation()
plt.imshow(np.abs(fftshift(crystal.pumpTime[::50], axes=1)), cmap="Reds", aspect="auto", origin="lower",
         extent=[-tMax * timeScale, tMax * timeScale, 0, 1000 * length])
plt.colorbar()
plt.xlabel("time / ps")
plt.ylabel("length / mm")
plt.title("Pump Field Profiles");

# %%
energies = np.array([10/40, 15/40, 20/40, 40/40])

conversion = np.zeros((energies.size, nt))
for i, e in enumerate(energies):
  crystal.setPump(pumpProfTemp / np.max(np.abs(pumpProfTemp[0])) * np.sqrt(e))
  crystal.runPumpSimulation()
  crystal.runSignalSimulation(signalProfFreq, inTimeDomain=False, inputMode=1)
  conversion[i] = 1 - np.abs(crystal.signalFreqs(1)[-1])**2 / np.abs(signalProfFreq)**2

# %%
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
# plt.imshow(np.abs(fftshift(crystal.signalTimes(0), axes=1)), cmap="Reds", aspect="auto", origin="lower",
#            extent=[-tMax * timeScale, tMax * timeScale, 0, 1000 * length])
plt.imshow(np.abs(fftshift(crystal.signalFreqs(0)[::50], axes=1)), cmap="Reds", aspect="auto", origin="lower",
         extent=[angFreqMax, -angFreqMax, 0, 1000 * length])
plt.colorbar()
plt.ylabel("length / mm")
plt.title("Converted Signal")

ax = fig.add_subplot(2, 1, 2)
# plt.imshow(np.abs(fftshift(crystal.signalTimes(1), axes=1)), cmap="Reds", aspect="auto", origin="lower",
#            extent=[-tMax * timeScale, tMax * timeScale, 0, 1000 * length])
plt.imshow(np.abs(fftshift(crystal.signalFreqs(1)[::50], axes=1)), cmap="Reds", aspect="auto", origin="lower",
         extent=[angFreqMax, -angFreqMax, 0, 1000 * length])
plt.colorbar()
plt.ylabel("length / mm")
plt.title("Original Signal")

# plt.xlabel("time / ps")
plt.xlabel("$\omega$ / $2 \pi$ THz")

plt.suptitle("Signal Field Profiles");

# %%
fig = plt.figure()
plt.plot(np.linspace(0, 1e3*length, crystal.signalFreqs(0).shape[0]),
         np.sum(np.abs(crystal.signalFreqs(0))**2, axis=1) / nt / amplitudeRatio)
plt.plot(np.linspace(0, 1e3*length, crystal.signalFreqs(0).shape[0]),
         np.sum(np.abs(crystal.signalFreqs(1))**2, axis=1) / nt)

plt.xlabel("z / mm")
plt.ylabel("Relative Energy")
plt.title("Cumulative Conversion");

# %%
plt.figure()
for i, e in enumerate(energies):
  plt.plot(fftshift(wavelengthS[np.logical_and(wavelengthS >= 1470, wavelengthS <= 1610)]),
           fftshift(conversion[i][np.logical_and(wavelengthS >= 1470, wavelengthS <= 1610)]),
           label="{:d} MW/cm$^2$".format(int(40 * e)))

plt.ylim(0, 1)
plt.legend()
plt.xlabel("$\lambda$ / nm")
plt.ylabel("Efficiency")
plt.title("Conversion Efficiency");
