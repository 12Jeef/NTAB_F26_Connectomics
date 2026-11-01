import numpy as np


def simulate(cfg, W, rng):
    """Poisson GLM spiking -> AR(1) calcium -> noisy ΔF/F sampled at cfg.fps. Returns (frames, neurons)."""
    n_neurons = W.shape[0]
    steps_per_frame = round(1 / (cfg.fps * cfg.dt_s))
    n_frames = round(cfg.minutes * 60 * cfg.fps)

    base = rng.lognormal(np.log(cfg.rate_hz), cfg.rate_sigma, n_neurons)
    W_in = W.T.tocsr()
    syn_decay = np.exp(-cfg.dt_s / cfg.tau_syn_s)
    ca_decay = np.exp(-cfg.dt_s / cfg.tau_ca_s)

    syn = np.zeros(n_neurons)
    ca = np.zeros(n_neurons)
    dff = np.empty((n_frames, n_neurons), np.float32)
    for frame in range(n_frames):
        for _ in range(steps_per_frame):
            rate = np.minimum(base * np.exp(W_in @ syn), cfg.max_rate_hz)
            spikes = rng.poisson(rate * cfg.dt_s)
            syn = syn * syn_decay + spikes
            ca = ca * ca_decay + cfg.dff_per_spike * spikes
        dff[frame] = ca

    return dff + cfg.noise * rng.standard_normal(dff.shape, dtype=np.float32)
