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
python scripts/00_inspect_tables.py      # pins the version, checks table names
python scripts/01_download_microns.py    # downloads into data/raw/
```

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
scripts/           numbered pipeline steps
src/microns/       shared code (config, CAVE client)
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
