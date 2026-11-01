import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist

CLASSES = np.array(["inhibitory", "excitatory"])


def place(cfg, rng):
    xyz = rng.uniform(0, cfg.side_um, (cfg.n_neurons, 3))
    cls = (rng.random(cfg.n_neurons) < cfg.frac_exc).astype(int)
    return xyz, cls


def wire(cfg, xyz, cls, rng):
    """Returns synapse counts n[pre, post] (0 = unconnected) and weights W[pre, post]."""
    p = np.asarray(cfg.p_conn)[cls[:, None], cls[None, :]]
    p = p * np.exp(-cdist(xyz, xyz, "sqeuclidean") / (2 * cfg.sigma_um**2))
    np.fill_diagonal(p, 0)

    n = (rng.random(p.shape) < p) * (1 + rng.poisson(cfg.synapses_mean - 1, p.shape))
    sign = np.array([-cfg.inh_gain, 1.0])[cls][:, None]
    strength = rng.lognormal(0, cfg.weight_sigma, p.shape)
    return n, sp.csr_matrix(cfg.coupling * cfg.weight * sign * n * strength)
