import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import os
    import pandas as pd
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    import torch

    from src.config import CFG, ID2LBL
    from src.train import MultiModalClassifier

    from src.utils import WheatDataModule, seed_everything
    return (
        CFG,
        EarlyStopping,
        ModelCheckpoint,
        MultiModalClassifier,
        WheatDataModule,
        os,
        pl,
        seed_everything,
    )


@app.cell
def _(CFG, os, seed_everything):
    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    cfg.VAL_DIR = "test"
    cfg.OUT_DIR = "./outputs"
    cfg.EPOCHS = 50
    cfg.BATCH_SIZE = 64

    cfg.RGB_BACKBONE = "efficientnet_b0"
    cfg.MS_BACKBONE = "efficientnet_b0"
    cfg.HS_BACKBONE = "resnet18"
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    return (cfg,)


@app.cell
def _(WheatDataModule, cfg):
    dm = WheatDataModule(cfg)
    dm.setup()

    print(f"Mode:  'MULTIMODAL'")
    print(f"Channels: {dm.n_ch} | HS: {dm.hs_ch}")
    print(f"Train samples: {len(dm.train_ds)}")
    print(f"Val samples: {len(dm.val_ds)}")
    print(f"Test samples: {len(dm.test_ds)}")
    return (dm,)


@app.cell
def _(EarlyStopping, ModelCheckpoint, MultiModalClassifier, cfg, dm, pl):
    model = MultiModalClassifier(cfg, hs_channels=dm.hs_ch, num_classes=3)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.OUT_DIR,
        filename='best-{epoch:02d}-{val_f1:.4f}',
        monitor='val_f1',
        mode='max',
        save_top_k=1
    )

    early_stop_cb = EarlyStopping(monitor='val_f1', patience=20, mode='max')

    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        precision='16-mixed',
        deterministic=True
    )
    return checkpoint_cb, model, trainer


@app.cell
def _(dm, model, trainer):
    trainer.fit(model, dm)
    return


@app.cell
def _(checkpoint_cb, trainer):
    print(f"\nBest model: {checkpoint_cb.best_model_path}")
    print(f"Best validation F1: {checkpoint_cb.best_model_score:.4f}")

    metrics = trainer.callback_metrics
    print(f"\nFinal validation accuracy: {metrics['val_acc']:.4f}")
    print(f"Final validation F1: {metrics['val_f1']:.4f}")
    return


@app.cell
def _():
    # test_preds = trainer.predict(model, dm.test_dataloader())
    # preds = torch.cat([batch['preds'] for batch in test_preds]).cpu().numpy()

    # sub = pd.DataFrame({
    #     'Id': [os.path.basename(dm.test_df.iloc[i].get('hs') or dm.test_df.iloc[i].get('ms') or dm.test_df.iloc[i].get('rgb')) 
    #            for i in range(len(dm.test_df))],
    #     'Category': [ID2LBL[p] for p in preds]
    # })
    # sub.to_csv(os.path.join(cfg.OUT_DIR, 'submission.csv'), index=False)
    # print(f"\nSubmission saved: {os.path.join(cfg.OUT_DIR, 'submission.csv')}")
    # print(sub['Category'].value_counts())
    return


if __name__ == "__main__":
    app.run()
