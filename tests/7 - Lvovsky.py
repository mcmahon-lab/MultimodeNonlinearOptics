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
# # Comparisons to published simulations of Green's Functions and Bloch Messiah Decomposition
# Decomposing a pulsed optical parametric amplifier into independent squeezers
# (Lvovsky, Wasilewski, Banaszek)
# Comparisons of the Green's function and highest order supermodes for different nonlinearities.

# %%
# cd ..

# %%
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

try:
    from nonlinearmedium import Chi2PDC
    print("using C++ implementation")
except:
    from NonlinearMedium import Chi2PDC
    print("using Python implementation")

from NonlinearHelper import *
from decompositions import bloch_messiah

# %%
# %matplotlib notebook
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 6]

# %% [markdown]
# Rough estimate of the paper's parameters

# %%
relDispLength = 3.25 # mm
chirpLength = -0.5 / relDispLength
nFreqs=512

# %%
crys1 = Chi2PDC(relativeLength=1 / relDispLength,
                nlLength=10 / relDispLength,
                dispLength=1,
                beta2=1,
                beta2s=1 / 3,
                beta1s=10,
                pulseType=0,
                chirp=chirpLength,
                tPrecision=nFreqs, zPrecision=200)

# %%
crys2 = Chi2PDC(relativeLength=1 / relDispLength,
                nlLength=1 / relDispLength,
                dispLength=1,
                beta2=1,
                beta2s=1 / 3,
                beta1s=10,
                pulseType=0,
                chirp=chirpLength,
                tPrecision=nFreqs, zPrecision=200)

# %%
crys3 = Chi2PDC(relativeLength=1 / relDispLength,
                nlLength=0.1 / relDispLength,
                dispLength=1,
                beta2=1,
                beta2s=1 / 3,
                beta1s=10,
                pulseType=0,
                chirp=chirpLength,
                tPrecision=nFreqs, zPrecision=200)

# %%
gC1, gS1 = crys1.computeGreensFunction(nThreads=4)

# %%
gC2, gS2 = crys2.computeGreensFunction(nThreads=4)

# %%
gC3, gS3 = crys3.computeGreensFunction(nThreads=4)

# %% [markdown]
# # Figure 3
# Green's functions for $L_{NL} = 0.1, 1, 10$

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.abs(gC1), origin='lower')
plt.title('$|C(\omega)|$')
plt.xlabel("$\omega$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()
ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.abs(gS1), origin='lower')
plt.title('$|S(\omega)|$')
plt.xlabel("$\omega$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()
plt.suptitle("$L/L_{NL}=0.1$");

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.abs(gC2), origin='lower')
plt.title('$|C(\omega)|$')
plt.xlabel("$\omega$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()
ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.abs(gS2), origin='lower')
plt.title('$|S(\omega)|$')
plt.xlabel("$\omega$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()
plt.suptitle("$L/L_{NL}=1$");

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.abs(gC3), origin='lower')
plt.title('$|C(\omega)|$')
plt.xlabel("$\omega$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()
ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.abs(gS3), origin='lower')
plt.title('$|S(\omega)|$')
plt.xlabel("$\omega$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()
plt.suptitle("$L/L_{NL}=10$");

# %% [markdown]
# # Supermodes

# %%
Z1 = calcQuadratureGreens(gC1, gS1)
Z2 = calcQuadratureGreens(gC2, gS2)
Z3 = calcQuadratureGreens(gC3, gS3)

# %%
psi1, D1, phi1 = bloch_messiah(Z1, tol=5e-5)
psi2, D2, phi2 = bloch_messiah(Z2, tol=5e-5)
psi3, D3, phi3 = bloch_messiah(Z3, tol=5e-5)
eigs1 = D1.diagonal()
eigs2 = D2.diagonal()
eigs3 = D3.diagonal()

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
plt.semilogy(eigs1[nFreqs:], "s-", markerfacecolor="none", label="squeezed variance")
plt.semilogy(eigs1[:nFreqs], "s-", markerfacecolor="none", label="anti-squeezed variance")
plt.semilogy(np.sqrt(eigs1[nFreqs:] * eigs1[:nFreqs]), "s-", markerfacecolor="none", label="uncertainty product")
plt.xlabel("supermodes")
plt.ylabel("squeezing")
plt.title("$L/L_{NL}=0.1$")
ax.set_xlim(-1, 50)

ax = fig.add_subplot(1, 3, 2)
plt.semilogy(eigs2[nFreqs:], "s-", markerfacecolor="none", label="squeezed variance")
plt.semilogy(eigs2[:nFreqs], "s-", markerfacecolor="none", label="anti-squeezed variance")
plt.semilogy(np.sqrt(eigs2[nFreqs:] * eigs2[:nFreqs]), "s-", markerfacecolor="none", label="uncertainty product")
plt.xlabel("supermodes")
plt.title("$L/L_{NL}=1$")
ax.set_xlim(-1, 50)

ax = fig.add_subplot(1, 3, 3)
plt.semilogy(eigs3[nFreqs:], "s-", markerfacecolor="none", label="squeezed variance")
plt.semilogy(eigs3[:nFreqs], "s-", markerfacecolor="none", label="anti-squeezed variance")
plt.semilogy(np.sqrt(eigs3[nFreqs:] * eigs3[:nFreqs]), "s-", markerfacecolor="none", label="uncertainty product")
plt.xlabel("supermodes")
plt.title("$L/L_{NL}=10$")
ax.set_xlim(-1, 50)
plt.legend();

# %% [markdown]
# # Figure 5
# The squeezing lengths, defined as $\Lambda_n = \zeta_n L_{NL}$, which are approximately invariant for $0 \leq L/L_{NL} \leq 15$.

# %%
fig = plt.figure()
ax = fig.gca()
ax.set_xlim(-1, 50)
ax.set_ylim(0.01, 1)
plt.semilogy(np.abs(np.log(eigs1[nFreqs:]) * 10), label="$L/L_{NL}=0.1$")
plt.semilogy(np.abs(np.log(eigs2[nFreqs:]) * 1), label="$L/L_{NL}=1$")
plt.semilogy(np.abs(np.log(eigs3[nFreqs:]) * 0.1), label="$L/L_{NL}=10$")
plt.title("Approximately Invariant Squeezing Lengths")
plt.xlabel("supermodes")
plt.ylabel("squeezing length")
plt.legend();

# %% [markdown]
# # Figure 4
# Spectral intensity profiles of the characteristic modes exhibiting strongest squeezing

# %%
fig = plt.figure()
plt.plot(fftshift(crys1.omega), np.abs(psi1[:nFreqs, 0] + 1j * psi1[nFreqs:, 0])**2, label="$L/L_{NL}=0.1$")
plt.plot(fftshift(crys2.omega), np.abs(psi2[:nFreqs, 0] + 1j * psi2[nFreqs:, 0])**2, label="$L/L_{NL}=1$")
plt.plot(fftshift(crys3.omega), np.abs(psi3[:nFreqs, 0] + 1j * psi3[nFreqs:, 0])**2, label="$L/L_{NL}=10$")
plt.title("1st Output Supermode")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend();

fig = plt.figure()
plt.plot(fftshift(crys1.omega), np.abs(psi1[:nFreqs, 1] + 1j * psi1[nFreqs:, 1])**2, label="$L/L_{NL}=0.1$")
plt.plot(fftshift(crys2.omega), np.abs(psi2[:nFreqs, 1] + 1j * psi2[nFreqs:, 1])**2, label="$L/L_{NL}=1$")
plt.plot(fftshift(crys3.omega), np.abs(psi3[:nFreqs, 1] + 1j * psi3[nFreqs:, 1])**2, label="$L/L_{NL}=10$")
plt.title("2nd Output Supermode")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend();

fig = plt.figure()
plt.plot(fftshift(crys1.omega), np.abs(psi1[:nFreqs, 2] + 1j * psi1[nFreqs:, 2])**2, label="$L/L_{NL}=0.1$")
plt.plot(fftshift(crys2.omega), np.abs(psi2[:nFreqs, 2] + 1j * psi2[nFreqs:, 2])**2, label="$L/L_{NL}=1$")
plt.plot(fftshift(crys3.omega), np.abs(psi3[:nFreqs, 2] + 1j * psi3[nFreqs:, 2])**2, label="$L/L_{NL}=10$")
plt.title("3rd Output Supermode")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend();

# %%
fig = plt.figure()
plt.plot(fftshift(crys1.omega), np.abs(phi1[0, :nFreqs] + 1j * phi1[0, nFreqs:])**2, label="$L/L_{NL}=0.1$")
plt.plot(fftshift(crys2.omega), np.abs(phi2[0, :nFreqs] + 1j * phi2[0, nFreqs:])**2, label="$L/L_{NL}=1$")
plt.plot(fftshift(crys3.omega), np.abs(phi3[0, :nFreqs] + 1j * phi3[0, nFreqs:])**2, label="$L/L_{NL}=10$")
plt.title("1st Input Supermode")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend()

fig = plt.figure()
plt.plot(fftshift(crys1.omega), np.abs(phi1[1, :nFreqs] + 1j * phi1[1, nFreqs:])**2, label="$L/L_{NL}=0.1$")
plt.plot(fftshift(crys2.omega), np.abs(phi2[1, :nFreqs] + 1j * phi2[1, nFreqs:])**2, label="$L/L_{NL}=1$")
plt.plot(fftshift(crys3.omega), np.abs(phi3[1, :nFreqs] + 1j * phi3[1, nFreqs:])**2, label="$L/L_{NL}=10$")
plt.title("2nd Input Supermode")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend()

fig = plt.figure()
plt.plot(fftshift(crys1.omega), np.abs(phi1[2, :nFreqs] + 1j * phi1[2, nFreqs:])**2, label="$L/L_{NL}=0.1$")
plt.plot(fftshift(crys2.omega), np.abs(phi2[2, :nFreqs] + 1j * phi2[2, nFreqs:])**2, label="$L/L_{NL}=1$")
plt.plot(fftshift(crys3.omega), np.abs(phi3[2, :nFreqs] + 1j * phi3[2, nFreqs:])**2, label="$L/L_{NL}=10$")
plt.title("3rd Input Supermode")
plt.xlabel("angular frequency")
plt.ylabel("field")
plt.legend();
