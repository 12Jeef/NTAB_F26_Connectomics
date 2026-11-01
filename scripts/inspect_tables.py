"""List materialization tables at the pinned version and check the ones config.py expects."""

from microns import config
from microns.cave import get_client


def main():
    client = get_client()
    version = client.materialize.version
    tables = sorted(client.materialize.get_tables())

    print(f"Datastack: {config.DATASTACK}  |  version: {version}  |  {len(tables)} tables\n")
    for t in tables:
        print(f"  {t}")

    expected = [
        config.SYNAPSE_TABLE,
        config.NUCLEUS_TABLE,
        config.PROOFREADING_TABLE,
        config.CELL_TYPE_TABLE,
        *config.COREG_TABLES,
    ]
    missing = [t for t in expected if t not in tables]
    print()
    if missing:
        print("Missing tables expected by config.py (update names there):")
        for t in missing:
            print(f"  - {t}")
    else:
        print("All tables expected by config.py are present.")


if __name__ == "__main__":
    main()
