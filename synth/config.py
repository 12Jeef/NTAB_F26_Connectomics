from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 0

    n_neurons: int = 1000
    side_um: float = 400.0
    frac_exc: float = 0.8

    # [pre][post] peak connection probability; 0 = inhibitory, 1 = excitatory
    p_conn: tuple = ((0.4, 0.4), (0.3, 0.15))
    sigma_um: float = 100.0
    synapses_mean: float = 2.0
    weight: float = 0.05
    weight_sigma: float = 0.5
    inh_gain: float = 4.0
    coupling: float = 3.0

    rate_hz: float = 2.0
    rate_sigma: float = 0.5
    max_rate_hz: float = 100.0
    dt_s: float = 0.01
    tau_syn_s: float = 0.02

    tau_ca_s: float = 1.5
    dff_per_spike: float = 0.2
    fps: float = 6.25
    minutes: float = 20.0
    noise: float = 0.1
