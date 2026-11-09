"""Train or evaluate Original and Ordered HyenaRec from a dataset configuration."""

import argparse
import contextlib
import json
import math
from pathlib import Path

import torch

from config import Config, parse_args
import models.hyena as kernels


@contextlib.contextmanager
def ordered_kernel_builder(beta, xlong=False):
    """Use the historical Ordered construction and restore the shared builder."""
    original = kernels.exposure_spectral_basis

    def build(profile, rank, difference_order=2, alpha=1.0, epsilon=1e-3):
        if difference_order != 2 or alpha != 1.0:
            raise ValueError("Ordered kernels require second differences and alpha=1")
        # The small-dataset historical entry point computes this endpoint in FP32.
        endpoint = beta if xlong else torch.log(torch.tensor(math.exp(beta)))
        weights = torch.exp(torch.linspace(0.0, endpoint, len(profile) - 2,
                                           dtype=torch.float64))
        synthetic = torch.ones(len(profile), dtype=torch.float64)
        synthetic[1:-1] = (1.0 + epsilon) / weights - epsilon
        if (synthetic[1:-1] <= 0).any():
            raise ValueError("This historical builder requires exp(beta) < (1+eps)/eps")
        return original(synthetic, rank, difference_order, alpha, epsilon)

    kernels.exposure_spectral_basis = build
    try:
        yield
    finally:
        kernels.exposure_spectral_basis = original


def make_config(parameters, method, output):
    args = parse_args([])
    for name, value in parameters.items():
        if not hasattr(args, name):
            raise ValueError(f"Unknown training parameter {name!r}")
        setattr(args, name, value)
    args.finite_history_mode = "hard" if method == "ordered" else "none"
    args.discrete_orthogonal = False
    args.min_val_recall10_for_test = 0.0
    args.checkpoint_path = str(output / "best.pth")
    args.latest_checkpoint_path = str(output / "latest.pth")
    config = Config.from_args(args)
    config.set_seed()
    return config


def evaluate_checkpoint(config, checkpoint, split):
    """Restore saved kernel buffers and use the training pipeline's evaluator."""
    from dataloader import dataloader_factory
    from models.checkpoints import compatible_state_dict
    from trainer.hyena import build_model, evaluate

    args = config.to_args()
    train_loader, validation_loader, test_loader = dataloader_factory(args)
    model = build_model(config, train_loader, args.num_items)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(compatible_state_dict(saved["model_state_dict"], model), strict=True)
    model.to(config.device)
    loader = validation_loader if split == "validation" else test_loader
    return evaluate(model, loader, torch.device(config.device))


def run_xlong(parameters, method, action, output, checkpoint, beta, data_root):
    from experiments import run_xlong_sampled

    settings = run_xlong_sampled.parse_args([])
    for name, value in parameters.items():
        if not hasattr(settings, name):
            raise ValueError(f"Unknown XLong training parameter {name!r}")
        setattr(settings, name, value)
    settings.mode = "exposure" if method == "ordered" else "baseline"
    settings.run_dir = output
    settings.latest_path = output / "latest.pth"
    settings.test_only = action == "evaluate"
    if action == "evaluate":
        if checkpoint.resolve() != (output / "best.pth").resolve():
            raise ValueError("XLong evaluation reads best.pth inside --output")
    cache = data_root / "preprocessed/xlong_min_rating0-min_uc5-min_sc5-leave_one_out/arrays_l1000"
    context = ordered_kernel_builder(beta, xlong=True) if method == "ordered" else contextlib.nullcontext()
    with context:
        run_xlong_sampled.main(settings, cache=cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=["original", "ordered"], default="ordered")
    parser.add_argument("--action", choices=["train", "evaluate"], default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    specification = json.loads(args.config.read_text(encoding="utf-8"))
    parameters = specification["parameters"]
    checkpoint = args.checkpoint or args.output / "best.pth"
    if args.action == "evaluate" and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.action == "train" and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory for a new training run")
    torch.set_num_threads(4)
    if specification["dataset"] == "xlong":
        if args.split != "test" and args.action == "evaluate":
            raise ValueError("The historical XLong evaluation entry point evaluates Test")
        run_xlong(parameters, args.method, args.action, args.output, checkpoint,
                  specification["beta"], args.data_root.resolve())
        return
    import datasets.base
    datasets.base.RAW_DATASET_ROOT_FOLDER = str(args.data_root.resolve())
    config = make_config(parameters, args.method, args.output)
    context = ordered_kernel_builder(specification["beta"]) if args.method == "ordered" else contextlib.nullcontext()
    with context:
        if args.action == "train":
            from trainer.hyena import train_model
            train_model(config)
        else:
            result = evaluate_checkpoint(config, checkpoint, args.split)
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
