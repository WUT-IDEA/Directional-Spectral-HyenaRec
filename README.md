# Directional Spectral Kernel Spaces for Sequential Recommendation

Code for **Directional Spectral Kernel Spaces for Sequential Recommendation** (ACM TORS submission).

The implementation supports the original Legendre-based HyenaRec and Ordered Spectral HyenaRec. Ordered Spectral uses a fixed basis constructed from a lag-weighted second-order difference operator, with learnable coefficients for each convolution channel.

## Installation

Install a CUDA-enabled PyTorch build compatible with your GPU, then install the dependencies.

```bash
pip install -r requirements.txt
```

## Data

Place the preprocessed data at

```text
data/preprocessed/{dataset}_min_rating0-min_uc5-min_sc5-leave_one_out/dataset.pkl
```

Dataset names are `ml-1m`, `steam`, `video`, and `xlong`. Each pickle contains `train`, `val`, `test`, `umap`, and `smap`. Preprocessing code for the three smaller datasets is in `datasets/`.

For XLong, prepare the sequence arrays before training or evaluation.

```bash
python -m experiments.prepare_xlong_arrays
```

## Training

Dataset-specific hyperparameters are provided in `configs/`. Use an empty output directory for a new run.

```bash
python run.py --config configs/video.json --method ordered --output runs/video_ordered
python run.py --config configs/ml-1m.json --method ordered --output runs/ml1m_ordered
python run.py --config configs/steam.json --method ordered --output runs/steam_ordered
python run.py --config configs/xlong.json --method ordered --output runs/xlong_ordered
```

Set `--method original` to train the Legendre-based model. Configurations and checkpoints are selected according to validation Recall@10. The released Ordered configurations use beta=5 and seed=42. XLong uses training batch size 32, microbatch size 8 with gradient accumulation, and evaluation batch size 64.

## Pretrained Models and Evaluation

Download the four Ordered Spectral checkpoints from [Release v1.0](https://github.com/WUT-IDEA/Directional-Spectral-HyenaRec/releases/tag/v1.0) into `downloads/`. To download all assets using GitHub CLI, run the following command.

```bash
gh release download v1.0 --repo WUT-IDEA/Directional-Spectral-HyenaRec --dir downloads
```

Then install the checkpoints.

```bash
python scripts/install_checkpoints.py downloads
```

The installer merges the XLong parts and verifies the checkpoint checksums. File metadata is provided in `checkpoints/manifest.json`.

```bash
python run.py --config configs/video.json --action evaluate --output checkpoints/video
python run.py --config configs/ml-1m.json --action evaluate --output checkpoints/ml-1m
python run.py --config configs/steam.json --action evaluate --output checkpoints/steam
python run.py --config configs/xlong.json --action evaluate --output checkpoints/xlong
```

ML-1M, Steam, and Video use full-catalog ranking. XLong samples 10,000 item IDs uniformly with replacement from the non-padding vocabulary and additionally includes the held-out target as the designated positive. Historical items, the target, and duplicate sampled IDs are not explicitly filtered.

## Reported Results

| Dataset | NDCG@10 | NDCG@20 | Recall@10 | Recall@20 |
|---|---:|---:|---:|---:|
| ML-1M | 0.201941 | 0.229342 | 0.339570 | 0.448179 |
| Steam | 0.074719 | 0.091419 | 0.137508 | 0.203926 |
| Video | 0.066994 | 0.080923 | 0.121485 | 0.176790 |
| XLong | 0.456896 | 0.469391 | 0.564162 | 0.613508 |

`results.csv` maps each reported result to its configuration, checkpoint, and evaluation record. Public baseline implementations are linked in [BASELINES.md](BASELINES.md). Upstream implementation acknowledgments are provided in [NOTICE.md](NOTICE.md).
