"""
Finite Difference Mode Solver.
Based on Fallahkhair, Murphy, JLT 2007
"Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides"
https://www.mathworks.com/matlabcentral/fileexchange/12734-waveguide-mode-solver
"""

import numpy as np
from scipy.sparse.linalg import eigen
from scipy.sparse import coo_matrix

class ModeSolverSemiVectorial():
  """
  This function calculates the modes of a dielectric waveguide using the
  semivectorial finite difference method. It is slightly faster than the
  full-vectorial mode-solver, but does not work with non-isotropic permittivity.

  wl : float
      Wavelength. Units must be consistent with dx, dy.
  dx : float or array of floats
      Horizontal distances between grid points in permittivity grid.
      In the case of an array the value must represent the distance between centers.
  dy : float or array of floats
      vertical distances between grid points in permittivity grid.
      In the case of an array the value must represent the distance between centers.
  relPermittivity : ny * nx array
      The relative permittivity cross-section of a waveguide.
  boundaryCond : str
      Specifies the boundary conditions for the north, south, east, west boundaries,
      out of the following options:
      '0' - field is zero immediately outside of the boundary.
      'S' - field is symmetric.
      'A' - field is antisymmetric.
  method : str
      "Ex", "Ey", or "scalar"; chooses the field that will be calculated.

      In the semivectorial approximation, one transverse component of the
      electric field (either Ex or Ey) dominates over the other component.
      We also do away with the boundary conditions on the normal
      component of the electric field at all dielectric interfaces.

      We therefore solve for E and β = n_eff(ω) ω / c in
      (∇ 1/n^2 ∇ n^2 + n^2 k_0^2) E = β^2 E
      with appropriate boundary conditions at all interfaces (continuous E field).
      For piecewise dielectrics, this becomes the same as the scalar approximation.

      In the scalar field approximation we also assume that the index
      differences among different regions in the structure are relatively small,
      and do away with the boundary conditions on the normal component of the
      electric field at all dielectric interfaces.

      We solve for E and β = n_eff(ω) ω / c such that
      (∇^2 + n^2 k_0^2) E = β^2 E
  """
  def __init__(self, relPermittivity, dx, dy, wl, boundaryCond="0000", method="Ex"):
    self.build_matrix(relPermittivity, dx, dy, wl, boundaryCond, method)

  def build_matrix(self, relPermittivity, dx, dy, wl, boundaryCond, method):

    ny, nx = relPermittivity.shape
    self.wl, self.ny, self.nx = wl, ny, nx

    k = 2 * np.pi / wl

    if np.isscalar(dx):
      uniformX = True
    else:
      uniformX = False
      if dx.shape[0] != nx - 1:
        raise ValueError("Array dx must be the width of permittivity array - 1")
      dx = np.pad(dx, (1, 1), "symmetric")
      # horizontal distances between (the center of) a point and the east and west neighbors
      e = np.broadcast_to(dx[1:],  (ny, nx)).ravel()
      w = np.broadcast_to(dx[:-1], (ny, nx)).ravel()

    if np.isscalar(dy):
      uniformY = True
    else:
      uniformY = False
      if dy.shape[0] != ny - 1:
        raise ValueError("Array dy must be the height of permittivity array - 1")
      dy = np.pad(dy, (1, 1), "symmetric")
      # vertical distances between (the center of) a point and the north and south neighbors
      n = np.broadcast_to(dy[1: ].reshape(-1, 1), (ny, nx)).ravel()
      s = np.broadcast_to(dy[:-1].reshape(-1, 1), (ny, nx)).ravel()

    # permittivity of each point
    ep = relPermittivity.ravel()

    An, As, Ae, Aw, Ap = (np.empty(nx * ny) for _ in range(5))
    if method.lower() == "ey":
      # permittivity of a point's east and west neighbors
      eps = np.pad(relPermittivity, ((0, 0), (1, 1)), "symmetric")
      ee = eps[:, 2:].ravel()
      ew = eps[:,:-2].ravel()

      if uniformY:
        An[:] = 1 / dy**2
        As[:] = 1 / dy**2
      else:
        An[:] = 2 / (n * (n + s))
        As[:] = 2 / (s * (n + s))

      if uniformX:
        norm = (ep + ee) * (ep + 3 * ew) + (ep + ew) * (ep + 3 * ee)
        Ae[:] = (8 / dx**2) * (ep + ew) * ee / norm
        Aw[:] = (8 / dx**2) * (ep + ee) * ew / norm
      else:
        # width of each permittivity bin
        p = np.broadcast_to(0.5 * (dx[:-1] + dx[1:]), (ny, nx)).ravel()
        norm = ((p * (ep - ee) + 2 * e * ee) * (p**2 * (ep - ew) + 4 * w**2 * ew) +
                (p * (ep - ew) + 2 * w * ew) * (p**2 * (ep - ee) + 4 * e**2 * ee))
        Ae[:] = 8 * (p * (ep - ew) + 2 * w * ew) * ee / norm
        Aw[:] = 8 * (p * (ep - ee) + 2 * e * ee) * ew / norm

      Ap[:] = ep * k**2 - An - As - (Ae / ee + Aw / ew) * ep

    elif method.lower() == "ex":
      # permittivity of a point's north and south neighbors
      eps = np.pad(relPermittivity, ((1,1), (0,0)), "symmetric")
      en = eps[2:, :].ravel()
      es = eps[:-2,:].ravel()

      if uniformY:
        norm = (ep + en) * (ep + 3 * es) + (ep + es) * (ep + 3 * en)
        An[:] = (8 / dy**2) * (ep + es) * en / norm
        As[:] = (8 / dy**2) * (ep + en) * es / norm
      else:
        # height of each permittivity bin
        q = np.broadcast_to(0.5 * (dy[:-1] + dy[1:]).reshape(-1, 1), (ny, nx)).ravel()
        norm = ((q * (ep - en) + 2 * n * en) * (q ** 2 * (ep - es) + 4 * s ** 2 * es) +
                (q * (ep - es) + 2 * s * es) * (q ** 2 * (ep - en) + 4 * n ** 2 * en))
        An[:] = 8 * (q * (ep - es) + 2 * s * es) * en / norm
        As[:] = 8 * (q * (ep - en) + 2 * n * en) * es / norm

      if uniformX:
        Ae[:] = 1 / dx**2
        Aw[:] = 1 / dx**2
      else:
        Ae[:] = 2 / (e * (e + w))
        Aw[:] = 2 / (w * (e + w))

      Ap[:] = ep * k**2 - (An / en + As / es) * ep - Ae - Aw

    else: #elif method == "scalar":
      if uniformY:
        An[:] = 1 / dy**2
        As[:] = 1 / dy**2
      else:
        An[:] = 2 / (n * (n + s))
        As[:] = 2 / (s * (n + s))

      if uniformX:
        Ae[:] = 1 / dx**2
        Aw[:] = 1 / dx**2
      else:
        Ae[:] = 2 / (e * (e + w))
        Aw[:] = 2 / (w * (e + w))

      Ap[:] = ep * k**2 - An - As - Ae - Aw


    indices = np.arange(nx * ny).reshape(ny, nx)

    for bc, a, bd in zip(boundaryCond, (An, As, Ae, Aw),
                         (indices[-1, :], indices[0, :],
                          indices[:, -1], indices[:, 0])):
      if   bc.lower() == 's': Ap[bd] += a[bd]
      elif bc.lower() == 'a': Ap[bd] -= a[bd]

    allInd = indices.ravel()
    northInd = indices[1:, :].ravel()
    southInd = indices[:-1,:].ravel()
    eastInd  = indices[:, 1:].ravel()
    westInd  = indices[:,:-1].ravel()

    # matrix (i,j) indices
    I = np.r_[allInd, westInd, eastInd, southInd, northInd]
    J = np.r_[allInd, eastInd, westInd, northInd, southInd]
    # corresponding matrix values
    V = np.r_[Ap[allInd], Ae[westInd], Aw[eastInd], An[southInd], As[northInd]]

    self.A = coo_matrix((V, (I, J))).tocsr()

    return self.A


  def solve(self, neigs=1, tol=0, returnModes=True, indexGuess=None, modeGuess=None):
    """
    Solve for the eigenmodes.

    neigs : int
        Number of eigenmodes to find
    tol : float
        Relative accuracy for eigenvalues. The default value of 0 implies machine precision.
    guess : float
        A guess for the refractive index. Only finds eigenvectors with an effective refractive index
        higher than this value.
    """
    if indexGuess is not None:
      guess = (2 * np.pi * indexGuess / self.wl)**2
    else:
      guess = None

    eigs = eigen.eigs(self.A, k=neigs, sigma=guess, which="LM", tol=tol,
                      v0=modeGuess, return_eigenvectors=returnModes)

    if returnModes:
      eigvals, eigvecs = eigs
    else:
      eigvals = eigs

    sortedIdx = np.argsort(eigvals)[::-1]
    neffs = self.wl * np.sqrt(eigvals[sortedIdx]) / (2 * np.pi)

    if returnModes:
      modes = [eigvecs[:, i].reshape(self.ny, self.nx) for i in sortedIdx]

      return (neffs, modes) if neigs > 1 else (neffs[0], modes[0])
    else:
      return neffs if neigs > 1 else neffs[0]
