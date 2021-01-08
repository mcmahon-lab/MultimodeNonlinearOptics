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
# # Comparisons to published simulations of Covariance Matrix and Bloch Messiah Decomposition
# Comparing to Multimode quantum theory of nonlinear propagation in optical fibers (Appendix A)
# (Aruto Hosaka, Taiki Kawamori, and Fumihiko Kannari)

# %% [markdown]
# Squeezing due to soliton propagation

# %%
# cd ..

# %%
import numpy as np

from numpy.fft import fftshift

from nonlinearmedium import Chi3

from NonlinearHelper import calcQuadratureGreens, calcCovarianceMtx, normalizedCov
from decompositions import bloch_messiah
from scipy.linalg import sqrtm, eig

# %%
# %matplotlib notebook
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 6]

# %%
nFreqs = 512
fiberS = Chi3(relativeLength=np.pi / 2,
              nlLength=1,
              beta2=-1,
              pulseType=1,
              tPrecision=nFreqs, zPrecision=100)

C, S = fiberS.computeGreensFunction(nThreads=4)

# %%
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)

tau = np.max(fiberS.tau)

plt.imshow(np.abs(fftshift(fiberS.pumpTime, axes=1))**2, origin='lower',
           extent=[-tau, tau, 0, np.pi / 2], aspect=2 * tau / np.pi)
plt.title("Soliton Power Profile")
plt.colorbar()
plt.ylabel("$\zeta$")

ax = fig.add_subplot(2, 1, 2)
plt.imshow(np.angle(fftshift(fiberS.pumpTime, axes=1)), cmap="hsv", origin='lower',
           extent=[-tau, tau, 0, np.pi / 2], aspect=2 * tau / np.pi)
plt.title("Soliton Phase Profile")
plt.colorbar()
plt.xlabel("$\\tau$")
plt.ylabel("$\zeta$")


# %% [markdown]
# # Green's Function

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.abs(C), origin='lower')
plt.title("$|C(\omega, \omega')|$")
plt.xlabel("$\omega'$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.title("$|S(\omega, \omega')|$")
plt.xlabel("$\omega'$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()

ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.abs(S), origin='lower')
plt.title("$|C(\omega, \omega')|$")
plt.xlabel("$\omega'$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.title("$|S(\omega, \omega')|$")
plt.xlabel("$\omega'$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()

# %% [markdown]
# # Find covariance Matrix and Compare to Hosaka
# Check the determinant is 1

# %%
# X and P output transformation matrix [xf, pf] = Z [xi, pi]
Z = calcQuadratureGreens(C, S)

assert not np.iscomplexobj(Z), "Complex object"

# Covariance Matrix
Cov = calcCovarianceMtx(Z, tol=1e-3)
print("Det(Cov) =", np.linalg.det(Cov))

assert not np.iscomplexobj(Cov), "Complex object"

normalizedC = normalizedCov(Cov)

# %%
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(normalizedC)
plt.title("normalized Cov")
plt.xlabel("$\omega$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar()

ax = fig.add_subplot(1, 2, 2)
plt.imshow(Cov)
plt.title("Cov")
plt.xlabel("$\omega$")
plt.ylabel("$\omega$")
plt.gca().get_xaxis().set_ticks([])
plt.gca().get_yaxis().set_ticks([])
plt.colorbar();

# %% [markdown]
# # Zoomed in view

# %%
fig = plt.figure()
arnd = 30
ax = fig.add_subplot(2, 2, 1)
plt.imshow(normalizedC[255-arnd: 255+arnd, 255-arnd: 255+arnd], vmin=-0.5, vmax=0.5)
plt.axis('off')
ax = fig.add_subplot(2, 2, 2)
plt.imshow(normalizedC[255-arnd: 255+arnd, 767-arnd: 767+arnd], vmin=-0.5, vmax=0.5)
plt.axis('off')
ax = fig.add_subplot(2, 2, 3)
plt.imshow(normalizedC[767-arnd: 767+arnd, 255-arnd: 255+arnd], vmin=-0.5, vmax=0.5)
plt.axis('off')
ax = fig.add_subplot(2, 2, 4)
im = plt.imshow(normalizedC[767-arnd: 767+arnd, 767-arnd: 767+arnd], vmin=-0.5, vmax=0.5)
plt.axis('off')
fig.suptitle("normalized Covariance")
fig.colorbar(im);

# %% [markdown]
# # Compare Supermode Squeezing to Hosaka

# %%
O1, D, O2 = bloch_messiah(Z, tol=5e-5)
print("O1 @ D @ O2 = Z", np.allclose(O1 @ D @ O2, Z, atol=5e-5))
diagSqueezing = D.diagonal()

# %%
plt.figure()
plt.plot(20 * np.log10(diagSqueezing[nFreqs:]), "s-", markerfacecolor="none", label="squeezed variance")
plt.plot(20 * np.log10(diagSqueezing[:nFreqs]), "s-", markerfacecolor="none", label="anti-squeezed variance")
plt.plot(10 * np.log10(diagSqueezing[nFreqs:] * diagSqueezing[:nFreqs]), label="uncertainty product")
plt.xlabel("supermodes")
plt.ylabel("Noise Reduction dB")
plt.legend();

# %% [markdown]
# # Bloch-Messiah: obtain squeezing values via eigenvalue formula

# %% [markdown]
# We want to factorize
# \begin{equation}
# C = O \Delta O'
# \end{equation}
# Rewrite to
# \begin{equation}
# C = (O \Delta O^T)(OO') = \Sigma U,
# \end{equation}
# Where solutions can be found as
# \begin{equation}
# \Sigma = (C C^T)^{1/2},
# U = (C C^T)^{-1/2} C.
# \end{equation}

# %%
sigma = sqrtm(Z @ Z.T)
eigenvalues, eigenvectors = eig(sigma)
sortedEig = np.sort(eigenvalues).real
print("real eigenvalues: ", np.allclose(np.imag(eigenvalues),0))
# above recreated in function blochMessiahEigs

# %%
fig = plt.figure()
plt.plot(20 * np.log10(sortedEig[-1:-nFreqs//2-1:-1]), "s-", markerfacecolor="none", label="squeezed variance")
plt.plot(20 * np.log10(sortedEig[:nFreqs//2]), "s-", markerfacecolor="none", label="anti-squeezed variance")
plt.plot(10 * np.log10(sortedEig[-1:-nFreqs//2-1:-1] * sortedEig[:nFreqs//2]), label="uncertainty product")
plt.xlabel("supermodes")
plt.ylabel("Noise Reduction dB")
plt.legend();
