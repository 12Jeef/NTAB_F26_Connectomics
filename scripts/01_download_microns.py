"""Download the MICrONS subsets the project needs, at the pinned materialization version.

Outputs (data/raw/):
  coregistered_neurons.parquet   neurons with calcium imaging (EM root ID <-> functional unit)
  proofreading.parquet           proofreading status per root ID
  cell_types.parquet             cell-type labels per root ID
  synapses_proofread.parquet     individual synapses between proofread, coregistered neurons
  connected_pairs.parquet        those synapses aggregated per directed (pre, post) pair
"""

import pandas as pd
from tqdm import tqdm

from microns import config
from microns.cave import get_client


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["t", "true", "1"])


def download_coregistration(client) -> pd.DataFrame:
    available = set(client.materialize.get_tables())
    frames = []
    for table in config.COREG_TABLES:
        if table not in available:
            print(f"  skipping {table} (not in this version)")
            continue
        df = client.materialize.query_table(table, desired_resolution=[1, 1, 1])
        df["coreg_source"] = table
        frames.append(df)
        print(f"  {table}: {len(df):,} rows")

    if not frames:
        raise RuntimeError("No coregistration tables found; run scripts/00_inspect_tables.py")

    coreg = pd.concat(frames, ignore_index=True)
    coreg = coreg[coreg["pt_root_id"] != 0]
    # Prefer manual matches when a unit appears in both tables.
    coreg = coreg.drop_duplicates(subset=["session", "scan_idx", "unit_id"], keep="first")
    return coreg


def download_synapses(client, pre_ids, post_ids) -> pd.DataFrame:
    post_ids = set(post_ids)
    pre_ids = sorted(pre_ids)
    chunks = [
        pre_ids[i : i + config.PRE_ID_CHUNK_SIZE]
        for i in range(0, len(pre_ids), config.PRE_ID_CHUNK_SIZE)
    ]

    frames = []
    for chunk in tqdm(chunks, desc="synapse chunks"):
        df = client.materialize.query_table(
            config.SYNAPSE_TABLE,
            filter_in_dict={"pre_pt_root_id": chunk},
            desired_resolution=[1, 1, 1],
        )
        df = df[df["post_pt_root_id"].isin(post_ids)]
        # Drop autapses / merge-error self-loops.
        df = df[df["pre_pt_root_id"] != df["post_pt_root_id"]]
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    client = get_client()
    print(f"Using {config.DATASTACK} at materialization version {client.materialize.version}\n")

    print("Coregistered neurons (calcium imaging <-> EM):")
    coreg = download_coregistration(client)
    coreg.to_parquet(config.DATA_RAW / "coregistered_neurons.parquet")
    coreg_ids = set(coreg["pt_root_id"])
    print(f"  -> {len(coreg_ids):,} unique coregistered root IDs\n")

    print("Proofreading status:")
    proof = client.materialize.query_table(config.PROOFREADING_TABLE)
    proof.to_parquet(config.DATA_RAW / "proofreading.parquet")
    axon_ok = set(proof.loc[truthy(proof["status_axon"]), "pt_root_id"])
    dend_ok = set(proof.loc[truthy(proof["status_dendrite"]), "pt_root_id"])
    print(f"  {len(axon_ok):,} with proofread axon, {len(dend_ok):,} with proofread dendrite\n")

    print("Cell types:")
    cell_types = client.materialize.query_table(config.CELL_TYPE_TABLE, desired_resolution=[1, 1, 1])
    cell_types.to_parquet(config.DATA_RAW / "cell_types.parquet")
    print(f"  {len(cell_types):,} rows\n")

    # A synapse label is only trustworthy if the presynaptic axon and the
    # postsynaptic dendrite were both proofread; both cells must also be imaged.
    pre_ids = axon_ok & coreg_ids
    post_ids = dend_ok & coreg_ids
    print(f"Synapses: {len(pre_ids):,} candidate presynaptic x {len(post_ids):,} postsynaptic neurons")
    syn = download_synapses(client, pre_ids, post_ids)
    syn.to_parquet(config.DATA_RAW / "synapses_proofread.parquet")

    pairs = (
        syn.groupby(["pre_pt_root_id", "post_pt_root_id"])
        .agg(n_synapses=("id", "size"), total_size=("size", "sum"))
        .reset_index()
    )
    pairs.to_parquet(config.DATA_RAW / "connected_pairs.parquet")
    print(f"  {len(syn):,} synapses across {len(pairs):,} connected directed pairs")
    print(f"\nDone. Files written to {config.DATA_RAW}")


if __name__ == "__main__":
    main()
