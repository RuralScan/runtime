"""Write synthetic v1 generations for reader tests."""

import hashlib
import json


def canonical(document):
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def record(data=b"x"):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def write_manifest(store, raw, contents=None):
    release_id = hashlib.sha256(raw).hexdigest()
    generation = store / "generations" / release_id
    (generation / "files").mkdir(parents=True)
    (generation / "manifest.json").write_bytes(raw)
    for name, data in (contents or {}).items():
        path = generation / "files" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return release_id, generation
