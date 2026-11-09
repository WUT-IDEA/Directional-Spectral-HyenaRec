"""Serial XLong training with the supplied 100/10,000 sampled-item protocol.

The backbone is unchanged. Both the baseline and ExposureSpectral variants use
the same independent output embedding and sampled-softmax head. Test is a
separate, explicit command; validation uses a fixed candidate RNG seed.
"""

import argparse
import gc
import inspect
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import Config
from models import HyenaRec
from models.checkpoints import compatible_state_dict


REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/preprocessed/xlong_min_rating0-min_uc5-min_sc5-leave_one_out/arrays_l1000"


class XLongArrays(Dataset):
    def __init__(self, split, cache=CACHE):
        self.split = split
        self.sequences = np.load(cache / ("test.npy" if split == "test" else "train.npy"), mmap_mode="r")
        self.targets = None if split == "train" else np.load(
            cache / ("val_targets.npy" if split == "val" else "test_targets.npy"), mmap_mode="r"
        )

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, index):
        sequence = torch.from_numpy(self.sequences[index].copy()).long()
        if self.split == "train":
            tokens = torch.zeros_like(sequence)
            tokens[1:] = sequence[:-1]
            return tokens, sequence
        return sequence, torch.tensor(int(self.targets[index]), dtype=torch.long)


def exposure_profile(lengths, max_len):
    # For target length T, causal pairs at lag k are max(T - 1 - k, 0).
    lengths = np.asarray(lengths, dtype=np.int64)
    return [float(np.maximum(lengths - 1 - k, 0).sum()) for k in range(max_len)]


def build_model(settings, num_items, profile):
    """Build the dataset-aware HyenaRec model with the frozen XLong settings."""
    config = Config(
        dataset_code="xlong", hidden_size=settings.hidden_size,
        num_hidden_layers=2, bert_max_len=1000,
        hidden_dropout_prob=settings.dropout, resid_dropout=settings.dropout,
        embed_dropout=settings.dropout, n_classes=num_items, use_head=False,
        finite_history_mode="hard" if settings.mode == "exposure" else "none",
        finite_history_rank=settings.rank,
        finite_history_profile=profile if settings.mode == "exposure" else None,
    )
    return HyenaRec(**config.backbone_cfg, dataset_code="xlong",
                    use_head=False, n_classes=num_items)


def loader(split, batch_size, cache=CACHE):
    return DataLoader(
        XLongArrays(split, cache), batch_size=batch_size, shuffle=split == "train",
        num_workers=0, pin_memory=True,
    )


@torch.no_grad()
def evaluate(model, data_loader, device, seed, max_batches=None):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0
    sums = {f"{metric}@{k}": 0.0 for k in (1, 5, 10, 20) for metric in ("Recall", "NDCG", "MRR")}
    for batch_idx, (tokens, targets) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        tokens, targets = tokens.to(device), targets.to(device)
        negatives = torch.randint(1, model.num_items + 1, (len(targets), 10000), device=device, generator=generator)
        scores = model(tokens, targets=targets, negative_samples=negatives, last_only=True)
        ranked = scores.topk(20, dim=1).indices
        position = torch.arange(1, 21, device=device)[None, :]
        hit = ranked.eq(10000)
        for k in (1, 5, 10, 20):
            found = hit[:, :k]
            sums[f"Recall@{k}"] += found.any(dim=1).float().sum().item()
            sums[f"MRR@{k}"] += (found.float() / position[:, :k]).sum().item()
            sums[f"NDCG@{k}"] += (found.float() / torch.log2(position[:, :k].float() + 1)).sum().item()
        total += len(targets)
    if not total:
        raise RuntimeError("No XLong evaluation examples")
    return {key: value / total for key, value in sums.items()}


def atomic_save(payload, path):
    """Save large CUDA tensors through disk-backed CPU views.

    torch.save normally materializes each CUDA storage on the CPU before
    writing it. XLong's 1 GiB embedding tables can exhaust physical memory
    during that copy, so stage them in bounded chunks instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with tempfile.TemporaryDirectory(prefix=path.name + ".stage.", dir=path.parent) as staging:
        counter = 0

        def stage(value):
            nonlocal counter
            if isinstance(value, torch.Tensor) and value.is_cuda and value.numel() * value.element_size() > 32 * 2**20:
                if not value.is_contiguous():
                    raise RuntimeError("Large checkpoint tensor is not contiguous")
                staged_path = str(Path(staging) / f"tensor_{counter}.bin")
                counter += 1
                mapped = torch.from_file(staged_path, shared=True, size=value.numel(), dtype=value.dtype)
                source = value.detach().view(-1)
                for start in range(0, value.numel(), (16 * 2**20) // value.element_size()):
                    end = min(start + (16 * 2**20) // value.element_size(), value.numel())
                    mapped[start:end].copy_(source[start:end])
                return mapped.view(value.shape)
            if isinstance(value, dict):
                result = value.__class__((key, stage(item)) for key, item in value.items())
                if hasattr(value, "_metadata"):
                    result._metadata = value._metadata
                return result
            if isinstance(value, list):
                return [stage(item) for item in value]
            if isinstance(value, tuple):
                return tuple(stage(item) for item in value)
            return value

        staged_payload = stage(payload)
        torch.save(staged_payload, temporary)
        del staged_payload
        gc.collect()
    os.replace(temporary, path)


def load_checkpoint(path):
    # NumPy 2 stores its RNG state under numpy._core. The remote PyTorch 1.13
    # environment uses NumPy 1.x, where the same reconstruction functions live
    # under numpy.core. Alias those modules only for our trusted checkpoint.
    if int(np.__version__.split(".")[0]) < 2:
        for name, module in list(sys.modules.items()):
            if name == "numpy.core" or name.startswith("numpy.core."):
                sys.modules.setdefault(name.replace("numpy.core", "numpy._core", 1), module)
    options = {"map_location": "cpu", "weights_only": False}
    if "mmap" in inspect.signature(torch.load).parameters:
        options["mmap"] = True
    return torch.load(path, **options)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exposure", "baseline"), default="exposure")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--microbatch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=REPO / "output/tors/runs/xlong_exposure_k32_h64_20260929")
    parser.add_argument("--latest-path", type=Path, default=Path.home() / "HyenaRec_XLong_resume/latest.pth")
    parser.add_argument("--resume-source-dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    return parser.parse_args(argv)


def main(args=None, cache=CACHE):
    """Run with validated settings supplied by the CLI or release entry point."""
    if args is None:
        args = parse_args()
    cache = Path(cache)
    if args.test_only and args.smoke:
        raise ValueError("Choose either --test-only or --smoke")
    if args.test_only:
        print("TEST startup", flush=True)
    if args.train_batch_size != 32 or args.inference_batch_size != 64:
        raise ValueError("XLong protocol requires train batch size 32 and inference batch size 64")
    if args.microbatch_size < 1 or args.microbatch_size > args.train_batch_size:
        raise ValueError("microbatch size must be between 1 and train batch size")
    metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    if args.test_only:
        print("TEST metadata loaded", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.test_only:
        print("TEST RNG seeded", flush=True)
    device = torch.device("cuda")
    profile = exposure_profile(np.load(cache / "train_lengths.npy", mmap_mode="r"), 1000)
    if args.test_only:
        print("TEST profile built", flush=True)
    # During test, load the large mmap-backed checkpoint into CPU parameters
    # first. Direct mmap-to-CUDA load_state_dict can crash the Windows driver.
    model = build_model(args, metadata["items"], profile).to("cpu" if args.test_only else device)
    if args.test_only:
        print("TEST model on CPU", flush=True)
    run = args.run_dir.resolve()
    best_path = run / "best.pth"
    metrics_path = run / "metrics.jsonl"
    settings_path = run / "settings.json"
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("smoke", "test_only", "resume_source_dir", "microbatch_size")}
    if args.test_only:
        if not best_path.is_file():
            raise FileNotFoundError(best_path)
        checkpoint = load_checkpoint(best_path)
        print("TEST checkpoint opened", flush=True)
        model.load_state_dict(compatible_state_dict(checkpoint["model"], model), strict=True)
        print("TEST weights loaded on CPU", flush=True)
        del checkpoint
        gc.collect()
        model.to(device)
        torch.cuda.synchronize()
        print("TEST model on GPU", flush=True)
        result = evaluate(model, loader("test", args.inference_batch_size, cache), device, args.seed + 20001)
        print(json.dumps({"test": result}, indent=2), flush=True)
        return result

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_data = loader("train", args.train_batch_size, cache)
    val_data = loader("val", args.inference_batch_size, cache)

    if args.smoke:
        model.train()
        tokens, targets = next(iter(train_data))
        tokens, targets = tokens.to(device), targets.to(device)
        negatives = torch.randint(1, model.num_items + 1, (*targets.shape, 100), device=device)
        scores = model(tokens, targets=targets, negative_samples=negatives)
        answers = torch.full_like(targets, 100)
        answers[targets == 0] = 0
        loss = F.cross_entropy(scores.flatten(0, 1), answers.flatten(), ignore_index=0, label_smoothing=args.label_smoothing)
        loss.backward()
        optimizer.step()
        validation = evaluate(model, val_data, device, args.seed + 10001, max_batches=1)
        print(json.dumps({"smoke_loss": loss.item(), "validation_batch": validation,
                          "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}, indent=2), flush=True)
        return


    run.mkdir(parents=True, exist_ok=True)
    if settings_path.exists():
        previous = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous != settings:
            changed = {key for key in settings if previous.get(key) != settings[key]}
            if changed != {"patience"} or int(settings["patience"]) <= int(previous["patience"]):
                raise RuntimeError(f"Run settings changed outside a patience increase: {changed}")
            with (run / "settings_changes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"from_patience": previous["patience"],
                                         "to_patience": settings["patience"]}) + "\n")
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    latest = args.latest_path.resolve()
    start_epoch, best_recall, best_epoch = 0, -1.0, -1
    if latest.exists():
        state = load_checkpoint(latest)
        if state.get("run_dir") != str(run) and (
            args.resume_source_dir is None or state.get("run_dir") != args.resume_source_dir
        ):
            raise RuntimeError("Latest checkpoint belongs to a different XLong run")
        model.load_state_dict(compatible_state_dict(state["model"], model), strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        try:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        except RuntimeError:
            if state.get("run_dir") == str(run):
                raise
            torch.cuda.manual_seed_all(args.seed + state["epoch"] + 1)
            print("Cross-device resume: CUDA RNG state was re-seeded", flush=True)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        start_epoch = state["epoch"] + 1
        best_recall, best_epoch = state["best_recall"], state["best_epoch"]
        torch.cuda.synchronize()
        del state
        gc.collect()
        print(f"RESUME epoch={start_epoch} best={best_recall:.6f}", flush=True)
    elif metrics_path.exists() or best_path.exists():
        raise RuntimeError("Existing run has no latest checkpoint; refusing to repeat epochs")
    (run / "runner.pid").write_text(str(os.getpid()), encoding="utf-8")
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        started = time.monotonic()
        loss_sum = 0.0
        for batch_idx, (tokens, targets) in enumerate(train_data):
            tokens, targets = tokens.to(device), targets.to(device)
            negatives = torch.randint(1, model.num_items + 1, (*targets.shape, 100), device=device)
            optimizer.zero_grad(set_to_none=True)
            valid_total = (targets != 0).sum().item()
            batch_loss = 0.0
            for offset in range(0, len(tokens), args.microbatch_size):
                end = offset + args.microbatch_size
                micro_targets = targets[offset:end]
                valid_micro = (micro_targets != 0).sum().item()
                if valid_micro == 0:
                    continue
                scores = model(tokens[offset:end], targets=micro_targets, negative_samples=negatives[offset:end])
                answers = torch.full_like(micro_targets, 100)
                answers[micro_targets == 0] = 0
                loss = F.cross_entropy(scores.flatten(0, 1), answers.flatten(), ignore_index=0,
                                       label_smoothing=args.label_smoothing)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_idx}")
                scaled_loss = loss * (valid_micro / valid_total)
                scaled_loss.backward()
                batch_loss += loss.item() * valid_micro / valid_total
            optimizer.step()
            loss_sum += batch_loss
            if batch_idx % 100 == 0:
                print(f"epoch={epoch} batch={batch_idx}/{len(train_data)} loss={batch_loss:.5f} gpu_gib={torch.cuda.memory_allocated()/2**30:.2f}", flush=True)
        val = evaluate(model, val_data, device, args.seed + 10001)
        improved = val["Recall@10"] > best_recall
        if improved:
            best_recall, best_epoch = val["Recall@10"], epoch
            atomic_save({"model": model.state_dict(), "epoch": epoch, "validation": val}, best_path)
        state = {
            "run_dir": str(run), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_recall": best_recall, "best_epoch": best_epoch,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        }
        atomic_save(state, latest)
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_data), "validation": val,
                  "best_epoch": best_epoch, "seconds": time.monotonic() - started}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"EARLY_STOP best_epoch={best_epoch} best_recall10={best_recall:.8f}", flush=True)
            break


if __name__ == "__main__":
    main()
