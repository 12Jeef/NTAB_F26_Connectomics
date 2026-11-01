# synth

A toy network with known wiring and simulated calcium imaging, written in the same schema as `official.ipynb`.

This is a **validation**, not a model of cortex. The system is deliberately simpler than MICrONS. The claim it supports is narrow: *in a setting where the ground truth is known, a method recovers connectivity beyond what distance explains, and recovers none when there is none.* This follows the approach of Stetter et al. (2012), who simulated calcium fluorescence from networks with known wiring to benchmark connectivity reconstruction.

## Model

**Neurons.** $N$ points uniform in a cube; each is excitatory with probability 0.8, otherwise inhibitory.

**Wiring.** For a pair $j \to i$ with cell classes $c_j, c_i$ and distance $d_{ij}$:

$$
P(j \to i) = p_{c_j c_i} \exp\!\left(-\frac{d_{ij}^2}{2\sigma^2}\right), \qquad
n_{ji} = \mathbb{1}[\text{connected}] \cdot (1 + \text{Poisson}(\bar n - 1))
$$

$$
W_{ji} = g \, w \, s_{c_j} \, n_{ji} \, \ell_{ji}, \qquad \ell_{ji} \sim \text{LogNormal}(0, \sigma_w^2)
$$

$n_{ji}$ is the synapse count (0 means unconnected), $s_c$ is $+1$ for excitatory and $-4$ for inhibitory, and $g$ is the global `coupling`.

### VAR(1), briefly

A vector autoregressive process of order 1 is

$$
x_t = A x_{t-1} + \varepsilon_t, \qquad \varepsilon_t \overset{\text{iid}}{\sim} (0, \Sigma)
$$

Each variable at time $t$ is a linear combination of every variable at $t-1$ plus fresh noise. $A_{ij} \neq 0$ means "$j$ directly influences $i$", so for a network, the off-diagonal entries of $A$ *are* the directed connectivity. The process is stationary iff the spectral radius $\rho(A) < 1$. To simulate it, start at $x_0 = 0$ and iterate the recursion, drawing new noise each step.

### Spiking: a Poisson VAR(1)

The simulator (`activity.py`) runs the nonlinear, spiking version of this recursion with $\Delta t = 10$ ms. Each neuron $i$ has a synaptic trace $h_i$, a lognormal baseline rate $b_i$, and decay $a = e^{-\Delta t / \tau_\text{syn}}$:

$$
\lambda_t = \min\!\left(b \odot e^{W^\top h_{t-1}},\ \lambda_\text{max}\right), \qquad
s_t \sim \text{Poisson}(\lambda_t \Delta t), \qquad
h_t = a\, h_{t-1} + s_t
$$

Linearizing $e^{u} \approx 1 + u$ gives $\mathbb{E}[s_t \mid h_{t-1}] \approx b\Delta t \odot (1 + W^\top h_{t-1})$, so

$$
h_t \approx \underbrace{\left(a I + \operatorname{diag}(b \Delta t)\, W^\top\right)}_{A} h_{t-1} + b\Delta t + \varepsilon_t
$$

That is, a VAR(1) whose off-diagonal entries are the (scaled) connectome. The rate cap $\lambda_\text{max}$ keeps the nonlinear process bounded. Setting $g = 0$ makes $A$ diagonal, and then the activity carries no information about the wiring.

### Calcium

This follows the fluorescence model of Stetter et al. (2012), without saturation or light scattering. Each neuron's calcium is itself an AR(1) process driven by its spikes:

$$
c_t = \gamma\, c_{t-1} + \alpha\, s_t, \qquad \gamma = e^{-\Delta t / \tau_\text{Ca}}
$$

With $\tau_\text{Ca} = 1.5$ s and $\alpha = 0.2$ ΔF/F per spike (roughly GCaMP6s), $c$ is sampled every 16 steps (6.25 Hz) and Gaussian noise $\mathcal{N}(0, \sigma_F^2)$ is added.

## Outputs

`python -m synth.dataset <out_dir> [key=value ...]`, where `key` is any `Config` field:

| File | Schema |
|---|---|
| `neurons.parquet` | `id` → `cell_class`, `cell_type`, `axon_proof`, `dend_proof`, `x`, `y`, `z` (µm) |
| `synapses.parquet` | (`pre_id`, `post_id`) → `n` (ground truth, complete) |
| `functional.parquet` | (`pre_id`, `post_id`) → `strength` (trace correlation) |
| `coreg.parquet` | `id` → `session`, `scan_idx`, `unit_id` |
| `traces.npy` | (frames, `unit_id`) ΔF/F |
| `config.json` | parameters used |

## Validation

Distance alone predicts connectivity well (AUC ≈ 0.87), so scores use a **distance-matched AUC**: negatives are reweighted to match the positives' distance distribution, and a score above 0.5 means signal beyond distance. `python -m synth.evaluate` checks the |correlation| baseline:

| `coupling` | distance-matched AUC |
|---|---|
| 0 (control) | 0.50 |
| 3 | 0.60 |

A method passes if it beats this baseline at $g > 0$ and stays at about 0.5 at $g = 0$.

## Reference

Stetter, O., Battaglia, D., Soriano, J., & Geisel, T. (2012). Model-free reconstruction of excitatory neuronal connectivity from calcium imaging signals. *PLoS Computational Biology*, 8(8), e1002653.
