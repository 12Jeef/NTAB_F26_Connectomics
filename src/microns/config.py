from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
PINNED_VERSION_FILE = PROJECT_ROOT / "config" / "pinned_version.txt"

DATASTACK = "minnie65_public"

# Table names in the public minnie65 datastack. Names change between releases;
# `scripts/00_inspect_tables.py` lists what exists at the pinned version.
SYNAPSE_TABLE = "synapses_pni_2"
NUCLEUS_TABLE = "nucleus_detection_v0"
PROOFREADING_TABLE = "proofreading_status_and_strategy"
COREG_TABLES = [
    "coregistration_manual_v4",
    "coregistration_auto_phase3_fwd_apl_vess_combined_v2",
]
CELL_TYPE_TABLE = "aibs_metamodel_celltypes_v661"

# Synapse queries are chunked by presynaptic root ID to stay under the
# materialization server's per-query row limit.
PRE_ID_CHUNK_SIZE = 200
