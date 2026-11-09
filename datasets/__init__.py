import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path


def _load_dataset_module(filename, module_name):
    """Load legacy dataset files whose filenames contain hyphens."""
    path = _Path(__file__).with_name(filename)
    spec = _importlib_util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load dataset module: {path}")
    module = _importlib_util.module_from_spec(spec)
    _sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ML1MDataset = _load_dataset_module('ml-1m.py', 'datasets.ml_1m').ML1MDataset
from .video import VideoDataset
from .steam import SteamDataset

DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    VideoDataset.code(): VideoDataset,
    SteamDataset.code(): SteamDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
