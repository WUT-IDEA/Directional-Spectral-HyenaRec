import torch
import torch.nn.functional as F
import numpy as np

from config import Config, parse_args
from models import HyenaRec
from models.checkpoints import compatible_state_dict
from dataloader import dataloader_factory
from tqdm import tqdm
from pathlib import Path
import json
import os
import random


@torch.no_grad()
def calculate_metrics(batch, model, device, ks=[10, 20]):
    """
    计算 Recall@k, MRR@k 和 NDCG@k 等指标
    """
    seqs, labels = batch

    seqs = seqs.to(device)
    labels = labels.to(device)

    # 获取模型的预测结果
    scores = model(seqs)[:, -1, :]
    B, L = seqs.shape
    for i in range(L):
        scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
    scores[:, 0] = -1e9  # padding

    # 计算指标
    metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), ks)

    return metrics


@torch.no_grad()
def evaluate(model, loader, device, ks=(1, 5, 10, 20)):
    """User-weighted full-catalog metrics; retain the historical masking policy."""
    model.eval()
    totals = {}
    count = 0
    for batch in tqdm(loader):
        metrics = calculate_metrics(batch, model, device, ks)
        size = batch[0].size(0)
        count += size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + size * value
    if not count:
        raise ValueError('Cannot evaluate an empty dataset')
    return {key: value / count for key, value in totals.items()}


def absolute_recall_mrr_ndcg_for_ks(scores, labels, ks):
    """
    计算 Recall@k, MRR@k 和 NDCG@k
    """
    metrics = {}
    labels = F.one_hot(labels, num_classes=scores.size(1))  # One-hot 编码标签
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)  # 排名

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]  # 选取前 k 个推荐物品
        hits = labels_float.gather(1, cut)

        # Recall@k 计算
        metrics['Recall@%d' % k] = \
            (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

        # MRR@k 计算
        metrics['MRR@%d' % k] = \
            (hits / torch.arange(1, k + 1).unsqueeze(0).to(labels.device)).sum(1).mean().cpu().item()

        # NDCG@k 计算
        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg.cpu().item()

    return metrics


def train(model, device, train_loader, optimizer, epoch, log_interval=100,
          label_smoothing=0.0, loss_weighting='uniform', recent_loss_gamma=2.0,
          loss_tail_positions=10, finite_history_regularization=0.0):
    """单轮训练函数"""
    model.train()
    ce = torch.nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=label_smoothing,
        reduction='none' if loss_weighting in ('recent', 'last') else 'mean',
    )
    train_loss = 0
    train_tqdm_dataloader = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(train_tqdm_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        output = output.view(-1, output.size(-1))

        target = target.view(-1)

        loss = ce(output, target)
        if loss_weighting in ('recent', 'last'):
            loss = loss.view(data.size(0), data.size(1))
            valid = target.view(data.size(0), data.size(1)).ne(0)
            if loss_weighting == 'recent':
                pos = torch.linspace(0.0, 1.0, data.size(1), device=data.device)
                weights = torch.exp(recent_loss_gamma * pos).unsqueeze(0)
                weights = weights.expand_as(loss) * valid
            else:
                tail = min(loss_tail_positions, data.size(1))
                weights = torch.zeros_like(loss)
                weights[:, -tail:] = 1.0
                weights = weights * valid
            loss = (loss * weights).sum() / weights.sum().clamp_min(1.0)
        if finite_history_regularization > 0:
            loss = loss + finite_history_regularization * model.kernel_regularization_loss()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return train_loss / len(train_loader)


def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(compatible_state_dict(checkpoint['model_state_dict'], model), strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    rng_state = checkpoint.get('rng_state')
    if rng_state is not None:
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and rng_state['cuda'] is not None:
            torch.cuda.set_rng_state_all(rng_state['cuda'])
    return model


def save_latest_training_state(model, optimizer, path, epoch):
    """Atomically save a resumable epoch boundary without touching the best model."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'rng_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }, temporary)
    os.replace(temporary, target)


def build_model(config, train_loader, num_items):
    """Resolve the vocabulary and fixed geometry for training or evaluation."""
    # The item vocabulary is dataset-specific.  The dataloader resolves it
    # from the preprocessed ``smap``; use that value instead of the old
    # Steam-specific default in Config (otherwise other datasets can index
    # past the embedding table).
    config.n_classes = num_items
    config.backbone_cfg['vocab_size'] = config.n_classes + 2
    print(f"数据集实际 item 数：{config.n_classes}，模型词表大小：{config.backbone_cfg['vocab_size']}")

    if getattr(config, 'finite_history_mode', 'none') != 'none':
        if not hasattr(train_loader.dataset, 'lag_exposure_counts'):
            raise TypeError('Training dataset does not expose causal lag counts')
        exposure = train_loader.dataset.lag_exposure_counts().numpy()
        exposure_mode = getattr(config, 'finite_history_exposure', 'empirical')
        if exposure_mode == 'uniform':
            exposure = np.ones_like(exposure)
        elif exposure_mode == 'shuffled':
            exposure = np.random.default_rng(config.seed + 7919).permutation(exposure)
        elif exposure_mode != 'empirical':
            raise ValueError(f'Unknown finite-history exposure mode: {exposure_mode}')
        config.finite_history_profile = exposure.tolist()
        config.layer_cfg['finite_history_profile'] = config.finite_history_profile
        config.backbone_cfg['layer'] = config.layer_cfg
        print(
            'Finite-history exposure:', exposure_mode,
            f'head={exposure[0]:.0f}', f'tail={exposure[-1]:.0f}',
            f'nonzero={int(np.count_nonzero(exposure))}/{len(exposure)}',
        )

    # 初始化模型及优化器
    model = HyenaRec(**config.backbone_cfg, dataset_code=config.dataset_code, use_head=config.use_head, n_classes=config.n_classes)
    return model


def train_model(config):
    device = torch.device(config.device)
    print("使用设备：", device)

    args = config.to_args()
    train_loader, val_loader, test_loader = dataloader_factory(args)

    model = build_model(config, train_loader, args.num_items)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = None
    if getattr(config, 'lr_scheduler', 'none') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler_tmax,
            eta_min=config.min_learning_rate,
        )

    checkpoint_path = getattr(config, 'checkpoint_path', 'best_model.pth')
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_path + '.config.json').write_text(json.dumps(vars(config), ensure_ascii=False, indent=2), encoding='utf-8')
    start_epoch = getattr(config, 'start_epoch', 0)
    best_recall = getattr(config, 'resume_best_recall', 0.0)
    best_epoch = getattr(config, 'resume_best_epoch', -1)
    resume_checkpoint = getattr(config, 'resume_checkpoint', '')
    if resume_checkpoint:
        print(f'从 checkpoint 恢复：{resume_checkpoint}')
        load_model(model, optimizer, resume_checkpoint)
        if best_epoch < 0:
            best_epoch = start_epoch - 1
        print(f'续训起始轮次：{start_epoch}，已有最佳 Recall@10：{best_recall:.6f}（Epoch {best_epoch}）')
    elif best_epoch < 0:
        best_epoch = 0

    for epoch in range(start_epoch, config.num_epochs):
        avg_train_loss = train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            label_smoothing=config.label_smoothing,
            loss_weighting=getattr(config, 'loss_weighting', 'uniform'),
            recent_loss_gamma=getattr(config, 'recent_loss_gamma', 2.0),
            loss_tail_positions=getattr(config, 'loss_tail_positions', 10),
            finite_history_regularization=getattr(
                config, 'finite_history_regularization', 0.0
            ),
        )

        avg_metrics = evaluate(model, val_loader, device)

        print(f'Epoch: {epoch} 平均训练损失: {avg_train_loss:.4f}')
        if scheduler is not None:
            scheduler.step()
            print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.8g}')
        print(f'验证指标: {avg_metrics}')

        with open(checkpoint_path + '.metrics.jsonl', 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({'epoch': epoch, 'train_loss': avg_train_loss, 'validation': avg_metrics}) + '\n')

        # 如果验证指标提升，则保存模型
        if avg_metrics['Recall@10'] > best_recall:
            best_recall = avg_metrics['Recall@10']
            best_epoch = epoch
            save_model(model, optimizer, checkpoint_path)
            print(f'保存最佳模型，当前 Recall@10: {avg_metrics["Recall@10"]:.4f}')

        latest_checkpoint_path = getattr(config, 'latest_checkpoint_path', '')
        if latest_checkpoint_path:
            save_latest_training_state(model, optimizer, latest_checkpoint_path, epoch)

        # 提前停止机制
        if config.early_stopping and (epoch - best_epoch) >= config.patience:
            print(f'早停：连续{config.patience}轮无提升')
            break

    # Only evaluate on test after the validation gate is passed.  This keeps
    # large hyperparameter sweeps from spending time on runs that are already
    # below the user's validation criterion.
    val_gate = getattr(config, 'min_val_recall10_for_test', 0.13)
    if getattr(config, 'skip_test', False):
        print(f'Validation-only run: best Recall@10={best_recall:.8f}, epoch={best_epoch}')
        return model, None
    if best_recall < val_gate:
        print(f'验证 R@10 最高为 {best_recall:.5f}，低于测试门槛 {val_gate:.5f}，跳过 test')
        return model, None

    # 测试阶段：加载最佳模型后在测试集上评估
    model = load_model(model, optimizer, checkpoint_path)
    model.eval()
    test_avg_metrics = evaluate(model, test_loader, device)
    print(f'最终测试指标: {test_avg_metrics}')
    return model, test_avg_metrics


if __name__ == "__main__":
    args = parse_args()
    config = Config.from_args(args)
    train_model(config)

