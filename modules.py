import numpy as np

def compute_neighbors(x, smooth, npart):
    """
    Computes the number of neighbors and the neighbor list for all particles at once.
    """
    # Mask for distances within 2h to get the number of neighbors
    q = np.abs(x[:, None] - x[None, :]) / smooth[:, None]
    mask = q <= 2.0
    nneigh = np.sum(mask, axis=1)
    
    # Create the neighbor list
    neighlist = np.zeros((npart, npart), dtype=int)
    for i in range(npart):
        neighlist[i, :nneigh[i]] = np.where(mask[i])[0]

    return nneigh, neighlist


def kernel(q, h):
    """
    Kernel function for SPH - M4 cubic spline kernel truncated at 2h.
    """
    W = np.zeros_like(q)
    W = np.where(q <= 1.0, 1.0 - 1.5 * q**2.0 + 0.75 * q**3.0, W)
    W = np.where((q > 1.0) & (q < 2.0), 0.25 * (2.0 - q)**3.0, W)
    W = np.where(q >= 2.0, 0.0, W)
    W = 2.0 * W / (3.0 * h)
    return W


def gradkernel(q, h):
    """
    Gradient of the kernel function for SPH - M4 cubic spline kernel truncated at 2h.
    """
    dWa = np.zeros_like(q)
    dWa = np.where(q <= 1.0, -3.0 * q / h + 9.0 * q**2.0 / (4.0 * h), dWa)
    dWa = np.where((q > 1.0) & (q < 2.0), - 3.0 * (2 - q)**2.0 / (4.0 * h), dWa)
    dWa = np.where(q >= 2.0, 0.0, dWa)
    dWa = 2.0 * dWa / (3.0 * h)
    return dWa


def density(i, x, smooth, nneigh, neighlist, mpart):
    """
    Calculates the density of particle i.
    """
    # Extract relevant neighbors and their indices
    neighbors = neighlist[i, :nneigh[i]]

    # Compute pairwise smoothing lengths and distances
    h = 0.5 * (smooth[i] + smooth[neighbors])
    q = np.abs(x[i] - x[neighbors]) / h

    # Mask to filter neighbours satisfying q <= 2.0
    mask = q <= 2.0
    filtered_q = q[mask]
    filtered_h = h[mask]

    # Compute the density using the kernel for filtered neighbors
    rho = np.sum(mpart * kernel(filtered_q, filtered_h))

    return rho


def accel(i, x, v, soundspeed, dens, Press, smooth, nneigh, neighlist, mpart):
    """
    Computes the acceleration at particle i.
    """
    dvadt = 0.0
    
    # Extract relevant neighbors and their indices
    neighbors = neighlist[i, :nneigh[i]]

    # Precompute pairwise values
    h = 0.5 * (smooth[i] + smooth[neighbors])
    dx = x[i] - x[neighbors]
    dv = v[i] - v[neighbors]
    q = np.abs(dx) / h

    # Compute artificial viscosity
    
    # Compute the pressure gradient contribution
    press_term = (
        Press[neighbors] / (dens[neighbors]**2) + Press[i] / (dens[i]**2)
        )

    # Apply mask for valid neighbors and exclude self-contribution (i != k) to acceleration
    mask = (q <= 2.0) & (neighbors != i)
    contrib = press_term * dx * gradkernel(q, h) / (np.abs(dx))
    dvadt = -np.sum(mpart * contrib[mask])

    return dvadt


def energy(i, x, v, soundspeed, dens, Press, smooth, nneigh, neighlist, mpart):
    """
    Computes the energy change at particle i.
    """

    dedt = 0.0

    return dedt
