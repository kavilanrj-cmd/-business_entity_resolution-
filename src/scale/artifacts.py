"""Small JSON manifests for resumability.

Each stage writes a manifest next to its artefacts recording what it produced
(row counts, shard names, index statistics, timings).  A later stage reads the
manifest instead of re-deriving anything, and can skip work that is already
done.  Manifests are JSON -- never a pickle -- so they stay diffable and tiny.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
STORE_MANIFEST = "store_manifest.json"
INDEX_MANIFEST = "index_manifest.json"
CANDIDATE_MANIFEST = "candidate_manifest.json"


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"written_at": time.time(), **payload}
    # Write to a temp file then replace, so an interrupted run never leaves a
    # half-written manifest that a later stage would trust.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def read_manifest(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Ignoring unreadable manifest %s: %s", path, exc)
        return None


def stage_is_current(manifest_path: str | Path, expected: dict[str, Any]) -> bool:
    """True when the manifest exists and its keys match ``expected``.

    Used to skip a completed stage when its inputs have not changed.
    """
    data = read_manifest(manifest_path)
    if data is None:
        return False
    return all(data.get(key) == value for key, value in expected.items())


def require_stage(manifest_path: str | Path, what: str) -> dict[str, Any]:
    data = read_manifest(manifest_path)
    if data is None:
        raise FileNotFoundError(
            f"{what} has not been built. Expected a manifest at {manifest_path}. "
            f"Run the corresponding scripts/build_*.py first."
        )
    return data
