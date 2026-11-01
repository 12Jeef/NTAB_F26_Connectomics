"""Collapse coregistered neurons into graph nodes and attach proofread synapses.

Outputs (data/processed/):
  nodes.parquet   one row per EM neuron, with its chosen calcium unit
  edges.parquet   directed proofread synapse edges between those neurons
"""

import pandas as pd

from microns import config
from microns.graph import build_edges, build_nodes, mark_connected_nodes


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    coreg = pd.read_parquet(config.DATA_RAW / "coregistered_neurons.parquet")
    pairs = pd.read_parquet(config.DATA_RAW / "connected_pairs.parquet")

    nodes = build_nodes(coreg)
    edges = build_edges(pairs, nodes)
    nodes = mark_connected_nodes(nodes, edges)

    nodes_path = config.DATA_PROCESSED / "nodes.parquet"
    edges_path = config.DATA_PROCESSED / "edges.parquet"
    nodes.to_parquet(nodes_path)
    edges.to_parquet(edges_path)

    connected = int(nodes["has_proofread_edge"].sum())
    print(f"Nodes: {len(nodes):,} neurons ({connected:,} have a proofread synapse)")
    print(f"  {int((nodes['n_units'] > 1).sum()):,} neurons matched more than one calcium unit")
    print(f"Edges: {len(edges):,} directed pairs, {int(edges['n_synapses'].sum()):,} synapses")
    print(f"\nWrote {nodes_path}")
    print(f"Wrote {edges_path}")


if __name__ == "__main__":
    main()
