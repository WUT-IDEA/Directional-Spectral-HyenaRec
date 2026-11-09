"""Dataset, optimization and kernel configuration for HyenaRec."""
import argparse
import json
import random
import numpy as np
import torch

RAW_DATASET_ROOT_FOLDER = "data"
STATE_DICT_KEY = "model_state_dict"
OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"

DEFAULTS = {'seed': 42,
 'device': 'cuda',
 'min_rating': 0,
 'min_uc': 5,
 'min_sc': 5,
 'split': 'leave_one_out',
 'bert_max_len': 50,
 'bert_mask_prob': 0.2,
 'sliding_window_size': 0.5,
 'num_epochs': 500,
 'num_workers': 4,
 'pin_memory': True,
 'early_stopping': True,
 'min_val_recall10_for_test': 0.13,
 'use_padding': True,
 'discrete_orthogonal': False,
 'filter_order': 64,
 'finite_history_mode': 'none',
 'finite_history_exposure': 'empirical',
 'finite_history_rank': 16,
 'finite_history_difference_order': 2,
 'finite_history_alpha': 1.0,
 'finite_history_epsilon': 0.001,
 'finite_history_profile': None,
 'label_smoothing': 0.0,
 'hidden_size': 64,
 'num_hidden_layers': 2,
 'hidden_dropout_prob': 0.6,
 'resid_dropout': 0.2,
 'embed_dropout': 0.2,
 'initializer_range': 0.02,
 'dataset_code': 'steam',
 'train_batch_size': 128,
 'val_batch_size': 256,
 'test_batch_size': 256,
 'learning_rate': 0.001,
 'lr_scheduler': 'none',
 'scheduler_tmax': 30,
 'min_learning_rate': 1e-05,
 'weight_decay': 0,
 'patience': 10,
 'checkpoint_path': 'best_model.pth',
 'latest_checkpoint_path': '',
 'skip_test': False,
 'resume_checkpoint': '',
 'start_epoch': 0,
 'resume_best_recall': 0.0,
 'resume_best_epoch': -1,
 'n_classes': 13078,
 'use_head': True,
 'loss_weighting': 'uniform'}
DEFAULTS["device"] = "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    """Resolve backbone dimensions from flat experiment settings."""
    def __init__(self, **parameters):
        unknown = parameters.keys() - DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown configuration parameters {sorted(unknown)}")
        self.__dict__.update(DEFAULTS)
        self.__dict__.update(parameters)
        self.d_inner = self.hidden_size * 4
        self.max_length = self.bert_max_len
        self.layer_cfg = {
            "d_model": self.hidden_size, "d_inner": self.d_inner,
            "dropout": self.hidden_dropout_prob, "l_max": self.max_length,
            "filter_order": self.filter_order,
            "discrete_orthogonal": self.discrete_orthogonal,
            "finite_history_mode": self.finite_history_mode,
            "finite_history_rank": self.finite_history_rank,
            "finite_history_difference_order": self.finite_history_difference_order,
            "finite_history_alpha": self.finite_history_alpha,
            "finite_history_epsilon": self.finite_history_epsilon,
            "finite_history_profile": self.finite_history_profile,
        }
        self.backbone_cfg = {
            "d_model": self.hidden_size, "n_layer": self.num_hidden_layers,
            "vocab_size": self.n_classes + 2, "max_len": self.max_length,
            "d_inner": self.d_inner, "resid_dropout": self.resid_dropout,
            "embed_dropout": self.embed_dropout, "layer": self.layer_cfg,
            "initializer_cfg": {"initializer_range": self.initializer_range},
        }

    @classmethod
    def from_args(cls, args):
        return cls(**vars(args))

    def to_args(self):
        return argparse.Namespace(**vars(self))

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument("--" + name, action="store_false" if default else "store_true", default=default)
        else:
            parser.add_argument("--" + name, type=json.loads if default is None else type(default), default=default)
    return parser.parse_args(argv)
