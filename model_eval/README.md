# model_eval

An evaluation harness for directed synapse prediction. It takes any model's scores on held-out neuron pairs and reports performance on various metrics.

## Overview

Two things make a raw score misleading here. First, positives are rare, so "above zero" says nothing. Second, distance, shared tuning and node degree all predict connections for reasons we already know. So the harness is built around two ideas:

1. Every metric returns its value together with its chance level, a node-bootstrap confidence interval and p-value, the number of positives, and the difference from each baseline.
2. Confounds are controlled. Negatives are matched on distance and degree, splits are node-disjoint, and the metric comparison is against a functional-similarity baseline, not chance.

It runs on `synth/` datasets and on MICrONS tables. On `synth/`, the coupling=0 control should come back at chance on every metric.

## Code structure

```
model_eval/
├── benchmark.py     # Benchmark: trusted pairs, labels, node-disjoint splits, matched negatives, motif counts
├── metrics.py       # every metric as f(frame, score) -> float, with its chance level
├── stats.py         # node bootstrap, permutation null, likelihood-ratio test, Benjamini–Hochberg
├── diagnostics.py   # motif errors, calibration
└── report.py        # baselines, Result, evaluate() -> Report
```

### Usage

Run from the repo root (or with it on `PYTHONPATH`).

```python
from synth.dataset import Dataset
from model_eval import Benchmark, evaluate

ds = Dataset.load("cache/synth")                 # or the real tables, same schema
bench = Benchmark(ds, split_seed=0)              # node-disjoint train / val / test

train, val, test = bench.pairs("train"), bench.pairs("val"), bench.pairs("test")
# each is a DataFrame indexed by (pre_id, post_id) with columns:
#   label, n_synapses, distance_um, distance_bin, out_degree_pre, in_degree_post, trusted, weight,
#   cascade, fan_out, fan_in, <neuron column>_pre, <neuron column>_post, <functional columns>

scores = model.predict(test.index)               # pd.Series over test.index; higher = more likely i→j

report = evaluate(bench, scores, against="abs_strength")   # real data: against="abs_signal_corr"
report.summary()                                 # metric × score table, plus chance
report.result()                                  # Result for the primary metric
report.by("cell_type_pre")                       # every metric within each stratum, BH-adjusted
report.incremental()                             # likelihood-ratio test of the score beyond distance + `against`
report.null(n=100)                               # permutation null within distance bins
report.motif_errors()                            # false positives by motif class
report.calibration()                             # intercept, slope, ECE, reliability curve
report.to_frame()                                # long format: score, metric, value, chance, ci_low, ci_high, p, n_pos
```

`bench.splits` holds the neuron ids of each split. To repeat over split seeds, build one `Benchmark` per seed and retrain.

`Benchmark` options:

| Argument | Default | Meaning |
|---|---|---|
| `split_seed` | `0` | Seed for the node-disjoint split and negative sampling |
| `split` | `(0.7, 0.15, 0.15)` | Fraction of neurons in train / val / test |
| `bin_um` | `25.0` | Distance bin width for matching and stratification |
| `neg_per_pos` | `10` | Matched negatives per positive |
| `trusted_only` | `True` | Keep only pairs whose sender has a proofread axon (`axon_proof != "none"`) |

A pair table contains every positive, `neg_per_pos` negatives per positive drawn from the same distance bin with probability ∝ out-degree(i) · in-degree(j), and the reverse of every positive. `weight` is the inverse sampling rate of each pair, which recovers natural prevalence.

`evaluate(bench, scores, split="test", against="abs_strength", n_boot=200, seed=0)` returns a `Report`. `Report.result(metric, score)` returns a `Result`:

```python
@dataclass(frozen=True)
class Result:
    value: float
    chance: float
    ci: tuple                 # 95%, node bootstrap
    p: float                  # two-sided bootstrap p-value against chance
    n_pos: int
    vs: dict                  # score name -> difference from that score
```

Note: strata with fewer than `MIN_POS = 20` positives come back as `NaN` with `n_pos` filled in.

## Metrics

**Primary metric:** `ap_ratio_binned` on the test split, compared against `against`.

### Ranking

- **AP ratio** (`ap_ratio`, `ap_ratio_binned`). Average precision divided by the exact expected AP of a random ranking, so chance is exactly 1. With few positives a random ranking's AP is above π, so dividing by π alone would make chance look like more than 1. AP is the most useful metric because the use case is ranking candidate synapses (McDermott et al. 2024), and the ratio follows BEELINE (Pratapa et al. 2020). The binned version is computed within each distance bin and pooled, weighted by positives.
- **Early precision ratio** (`epr`). Precision among the top *k* pairs, where *k* is the number of positives, divided by π. It answers: "if we checked the model's best guesses, how many would be real?"
- **AUROC** (`auroc`). A secondary metric with chance 0.5. Unlike AP, it doesn't depend on π.

### Confound control

- **Covariate-adjusted AUROC** (`auroc_binned`). Each positive is compared only with negatives from the same distance bin (Janes & Pepe 2008).
- **Within-sender and within-receiver AUROC** (`auroc_sender`, `auroc_receiver`). AUROC within each presynaptic (or postsynaptic) neuron's pairs, pooled. This cancels any "this neuron connects a lot" signal, so only pair-specific information scores above 0.5.
- **Incremental value** (`report.incremental()`). A likelihood-ratio test of `label ~ distance + against` against `label ~ distance + against + score`. This is the test of "beyond tuning"; a ΔAUC test is invalid for nested models (Pepe et al. 2013).
- **Residual distance check** (`residual_distance_{10,25,50}`). The correlation of within-bin ranks of score and distance, at 10, 25 and 50 µm bins. If it's nonzero, the score still carries distance at that resolution.

### Direction

- **Direction accuracy** (`direction`). Among pairs connected in exactly one direction, the fraction where score(i→j) > score(j→i). Chance is 0.5, and any symmetric score, such as correlation, gets 0.5.

### Calibration

- **Calibration intercept, slope and reliability curve** (`report.calibration()`). Scores must be probabilities at the sampled prevalence. They are shifted to the natural prevalence (Saerens et al. 2002; King & Zeng 2001) and evaluated with the inverse sampling weights. A perfectly calibrated model has intercept 0 and slope 1 (Van Calster et al. 2019). ECE is reported as a secondary number.

### Strength

- **Strength correlation** (`strength`). The Spearman correlation between score and synapse count, computed among positives only, so it doesn't mix with existence.

### Diagnostics

- **Motif errors** (`report.motif_errors()`). Among the top-*k* false positives, how over-represented each motif is relative to all negatives (1 = not at all). The motifs are *cascade* (some i→b→j exists), *fan-out* (some a→i and a→j, i.e. shared input), *fan-in* (some i→c and j→c), and *reverse* (j→i exists). This follows Marbach et al. (2010). A high fan-out ratio means the model is detecting shared drive; a high reverse ratio means it can't tell direction.
- **Label noise.** Build the benchmark with `trusted_only=False` and call `report.by("trusted")`. The gap estimates how much missing axons make results look worse than they are.

### Nulls and uncertainty

- **Permutation null** (`report.null()`). Labels are shuffled within distance bins and the ranking metrics recomputed (Ojala & Garriga 2010). The strength and residual-distance rows don't depend on labels in a meaningful way and should be ignored there.
- **Node bootstrap.** Confidence intervals resample neurons, not pairs, because pairs that share a neuron aren't independent (Snijders & Borgatti 1999).
- **Multiple comparisons.** `report.by()` adds Benjamini–Hochberg-adjusted p-values across strata.

### Baselines

Every report includes these baselines, computed on the same pairs:

| Baseline | Score | What it rules out |
|---|---|---|
| `distance` | −distance (any monotone fit ranks identically) | Geometry |
| `degree` | true out-degree(i) × in-degree(j) | Node propensity (Aiyappa et al. 2025); an upper bound for degree-only models |
| `abs_<col>` | \|value\| for each column of the functional table | Co-activity; on real data `abs_signal_corr` is shared tuning, the bar to beat |

`synth/` has no stimulus, so its only functional column is `strength` (trace correlation) and the bar is `abs_strength`.

## References

- Aiyappa, R. et al. (2025). Implicit degree bias in the link prediction task. *ICML*.
- Janes, H. & Pepe, M. S. (2008). Adjusting for covariates in studies of diagnostic, screening, or prognostic markers. *Am. J. Epidemiol.* 168, 89–97.
- King, G. & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis* 9, 137–163.
- Marbach, D. et al. (2010). Revealing strengths and weaknesses of methods for gene network inference. *PNAS* 107, 6286–6291.
- McDermott, M. et al. (2024). A closer look at AUROC and AUPRC under class imbalance. *NeurIPS*.
- Ojala, M. & Garriga, G. C. (2010). Permutation tests for studying classifier performance. *JMLR* 11, 1833–1863.
- Pepe, M. S. et al. (2013). Testing for improvement in prediction model performance. *Stat. Med.* 32, 1467–1482.
- Pratapa, A. et al. (2020). Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. *Nat. Methods* 17, 147–154.
- Saerens, M., Latinne, P. & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities. *Neural Comput.* 14, 21–41.
- Snijders, T. A. B. & Borgatti, S. P. (1999). Non-parametric standard errors and tests for network statistics. *Connections* 22, 161–170.
- Van Calster, B. et al. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Med.* 17, 230.
