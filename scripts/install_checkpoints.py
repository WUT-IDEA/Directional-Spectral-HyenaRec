"""Verify downloaded Release assets and install the four checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("downloads", type=Path)
args = parser.parse_args()
for row in json.loads((root / "checkpoints/manifest.json").read_text()):
    target = root / "checkpoints" / row["file"]
    if target.exists():
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != row["expected_sha256"]:
            raise RuntimeError("Existing checkpoint has a different hash: " + str(target))
        print("Verified existing", target)
        continue
    inputs = [args.downloads / name for name in row["assets"]]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pth.tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for path in inputs:
            with path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024**2), b""):
                    digest.update(block)
                    output.write(block)
    if temporary.stat().st_size != row["expected_bytes"] or digest.hexdigest() != row["expected_sha256"]:
        raise RuntimeError("Checkpoint integrity check failed: " + str(target))
    os.replace(temporary, target)
    print("Installed and verified", target)
