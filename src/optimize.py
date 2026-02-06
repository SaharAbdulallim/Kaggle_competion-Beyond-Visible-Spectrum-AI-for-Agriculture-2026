import os

import joblib
import kornia.augmentation as K
import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.config import CFG
from src.train import WheatClassifier
from src.utils import WheatDataset, infer_hs_channels, make_df, seed_everything

torch.set_float32_matmul_precision('medium')


def objective(trial: optuna.Trial) -> float:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    cfg.VAL_DIR = "test"
    cfg.OUT_DIR = "./outputs"
    cfg.EPOCHS = 25
    cfg.WANDB_ENABLED = False

    cfg.LR = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    cfg.WD = trial.suggest_float("wd", 0.005, 0.1, log=True)
    cfg.DROPOUT = trial.suggest_float("dropout", 0.3, 0.6, step=0.1)
    cfg.LABEL_SMOOTHING = trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05)
    cfg.MIXUP_ALPHA = trial.suggest_float("mixup_alpha", 0.0, 0.4, step=0.1)
    cfg.BATCH_SIZE = trial.suggest_categorical("batch_size", [16, 32])

    seed_everything(cfg.SEED)

    train_df = make_df(cfg.ROOT, cfg.TRAIN_DIR)
    hs_ch = infer_hs_channels(train_df, cfg)

    pca_model, pca_n_features = None, None
    if cfg.PCA_COMPONENTS > 0 and cfg.USE_HS and os.path.exists(cfg.PCA_PATH):
        pca_data = joblib.load(cfg.PCA_PATH)
        pca_model = pca_data['model'] if isinstance(pca_data, dict) else pca_data
        pca_n_features = pca_data.get('n_features', hs_ch) if isinstance(pca_data, dict) else hs_ch
        hs_ch = cfg.PCA_COMPONENTS

    train_transforms = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5), K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=90.0, p=0.5),
        K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.3),
        K.RandomGaussianNoise(mean=0., std=0.03, p=0.2), data_keys=["image"]
    )

    full_dataset = WheatDataset(train_df, cfg, hs_ch, train_transforms, pca_model, pca_n_features)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.SEED)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        train_loader = DataLoader(
            Subset(full_dataset, tr_idx), cfg.BATCH_SIZE, True,
            num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_idx), cfg.BATCH_SIZE, False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True
        )

        model = WheatClassifier(cfg, hs_ch, 3)
        trainer = pl.Trainer(
            max_epochs=cfg.EPOCHS,
            accelerator='auto', devices=1,
            callbacks=[EarlyStopping(monitor='val_f1', patience=12, mode='max')],
            enable_progress_bar=False, enable_model_summary=False,
            logger=False, precision='16-mixed', deterministic=True
        )
        trainer.fit(model, train_loader, val_loader)
        val_f1 = trainer.callback_metrics.get('val_f1', torch.tensor(0.0)).item()
        fold_scores.append(val_f1)

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_f1 = float(np.mean(fold_scores))
    trial.set_user_attr("fold_scores", fold_scores)
    return mean_f1


def run_optimization(n_trials: int = 50):
    seed_everything(4433)
    os.makedirs("./outputs", exist_ok=True)

    study = optuna.create_study(
        direction='maximize',
        study_name='wheat-multimodal-optimization',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print(f"\nBest trial: {study.best_trial.number} | val_f1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()

    study = run_optimization(args.n_trials)

    best_config = {
        'val_f1': float(study.best_value),
        'trial': study.best_trial.number,
        'hyperparameters': study.best_params
    }
    with open("./outputs/best_params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    print("Saved to outputs/best_params.yaml")