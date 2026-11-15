# Inferring Synaptic Connectivity from Neural Activity (MICrONS)

Link-prediction benchmark and models for recovering synaptic wiring from calcium
imaging in the MICrONS mouse visual cortex dataset (`minnie65_public`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
pip install -e .
cp .env.example .env
```

Get a CAVE API token:

1. Go to https://global.daf-apis.com/auth/api/v1/create_token
2. Log in with Google and copy the token shown
3. Paste it into `.env` after `TOKEN=`

## Downloading MICrONS

```bash
python scripts/inspect_tables.py      # pins the version, checks table names
python scripts/download_microns.py    # downloads into data/raw/
python scripts/build_anatomical_graph.py              # nodes + proofread synapse edges
python scripts/download_calcium_traces.py             # in-vivo fluorescence, one scan at a time
python scripts/build_functional_knn.py                # kNN from calcium traces (Ding fallback)
python scripts/visualize_functional_knn.py            # draw the functional graph
```

## Link models (branch `model-functional`)

Functional kNN is the **input** graph. Proofread synapses are **labels**. Data lives locally in `data/processed/` or on Modal volume `ntab-functional`.

```bash
python scripts/pull_modal_functional.py          # optional: copy from Modal
python scripts/verify_functional_anatomy.py      # does G_f resemble synapses?
python scripts/train_ncn.py --epochs 15 --pairs 4096
python scripts/train_seal.py --epochs 4 --pairs 1024
python scripts/compare_link_models.py            # Pearson / common-neighbor vs NCN / SEAL
```

NCN is the directed architecture from the group notebook. SEAL is a 1-hop enclosing-subgraph GNN plus a common-neighbor baseline. Coordinates are not given to either model.


The first run pins the latest materialization version to `config/pinned_version.txt`.
Commit that file so table contents can't drift underneath the analysis. Delete it
only if you deliberately want to move to a newer release.

The download keeps:

- **Neurons with calcium imaging data**: EM root IDs coregistered to functional units
- **Neuron pairs with proofread synapses**: synapses where the presynaptic axon and
  postsynaptic dendrite are both proofread and both cells are coregistered

## Layout

```
config/            pinned materialization version
data/raw/          downloaded tables (gitignored)
data/processed/    derived datasets (gitignored)
notebooks/         exploration
scripts/           pipeline steps
  inspect_tables.py
  download_microns.py
  build_anatomical_graph.py              synapse target graph
  download_calcium_traces.py             DANDI fluorescence, one scan per batch
  build_functional_knn.py                kNN from calcium traces (Ding fallback)
  build_functional_knn_digital_twin.py   kNN from digital-twin predictions
  visualize_functional_knn.py
  verify_functional_anatomy.py
  train_ncn.py
  train_seal.py
  compare_link_models.py
src/microns/       shared code (config, CAVE client, NCN, SEAL)
```

## Roadmap

- **Aim 1**: Controlled benchmark. Distance-matched negatives, stratified by layer,
  area, and cell type, with performance reported per distance bin. Baselines are raw
  response correlation and tuning similarity from stimulus-predicted responses.
- **Aim 2**: Models for connection existence, sign, and strength. Two-stream neuron
  embeddings (tuning + residual), compared across an asymmetric bilinear scorer, a
  sequence model, and a SEAL-style GNN. No spatial coordinates are given to any model.
- **Aim 3**: Recoverability limits by cell-type pair, layer, strength, and motif;
  label-noise analysis (proofread vs. unproofread); and porting of GRN inference methods.
