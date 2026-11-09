"""Convert the supplied XLong pickle to small, memory-mapped sequence arrays."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/preprocessed/xlong_min_rating0-min_uc5-min_sc5-leave_one_out/dataset.pkl"
CACHE = ROOT / "data/preprocessed/xlong_min_rating0-min_uc5-min_sc5-leave_one_out/arrays_l1000"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1000)
    args = parser.parse_args()
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["max_len"] != args.length or meta["source_bytes"] != SOURCE.stat().st_size:
            raise RuntimeError("Existing XLong cache does not match the source or length")
        for name in ("train", "test", "val_targets", "test_targets", "train_lengths"):
            if not (CACHE / f"{name}.npy").is_file():
                raise RuntimeError(f"Incomplete existing XLong cache: {name}")
        print(json.dumps(meta, indent=2), flush=True)
        return

    with SOURCE.open("rb") as stream:
        dataset = pickle.load(stream)
    users = len(dataset["umap"])
    items = len(dataset["smap"])
    user_ids = sorted(dataset["train"])
    if users != len(user_ids) or user_ids != list(range(1, users + 1)):
        raise ValueError("XLong user IDs must be dense and start at one")
    length = args.length
    train = np.lib.format.open_memmap(CACHE / "train.npy", mode="w+", dtype="int32", shape=(users, length))
    test = np.lib.format.open_memmap(CACHE / "test.npy", mode="w+", dtype="int32", shape=(users, length))
    val_targets = np.lib.format.open_memmap(CACHE / "val_targets.npy", mode="w+", dtype="int32", shape=(users,))
    test_targets = np.lib.format.open_memmap(CACHE / "test_targets.npy", mode="w+", dtype="int32", shape=(users,))
    train_lengths = np.lib.format.open_memmap(CACHE / "train_lengths.npy", mode="w+", dtype="int32", shape=(users,))
    train[:] = 0
    test[:] = 0
    for i, user in enumerate(user_ids):
        history = dataset["train"][user]
        val = dataset["val"][user]
        answer = dataset["test"][user]
        if len(val) != 1 or len(answer) != 1:
            raise ValueError(f"Expected one validation and test item for user {user}")
        suffix = history[-length:]
        test_suffix = (history + val)[-length:]
        train[i, -len(suffix):] = suffix
        test[i, -len(test_suffix):] = test_suffix
        val_targets[i] = val[0]
        test_targets[i] = answer[0]
        train_lengths[i] = len(suffix)
    for array in (train, test, val_targets, test_targets, train_lengths):
        array.flush()
    meta = {
        "users": users, "items": items, "max_len": length,
        "source": str(SOURCE), "source_bytes": SOURCE.stat().st_size,
        "train_min_length": int(train_lengths.min()),
        "train_max_length": int(train_lengths.max()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
