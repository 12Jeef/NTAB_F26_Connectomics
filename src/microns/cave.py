import os

from caveclient import CAVEclient
from dotenv import load_dotenv

from microns.config import DATASTACK, PINNED_VERSION_FILE, PROJECT_ROOT


def get_token() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "No CAVE token found. Copy .env.example to .env and set TOKEN=. "
            "Create one at https://global.daf-apis.com/auth/api/v1/create_token"
        )
    return token


def pinned_version(client: CAVEclient) -> int:
    """Return the pinned materialization version, pinning the latest one on first call."""
    if PINNED_VERSION_FILE.exists():
        return int(PINNED_VERSION_FILE.read_text().strip())

    version = max(client.materialize.get_versions())
    PINNED_VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    PINNED_VERSION_FILE.write_text(f"{version}\n")
    print(f"Pinned materialization version {version} -> {PINNED_VERSION_FILE}")
    return version


def get_client() -> CAVEclient:
    client = CAVEclient(DATASTACK, auth_token=get_token())
    client.materialize.version = pinned_version(client)
    return client
