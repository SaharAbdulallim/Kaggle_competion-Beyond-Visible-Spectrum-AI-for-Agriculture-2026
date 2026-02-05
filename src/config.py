from dataclasses import dataclass


@dataclass
class CFG:
    ROOT: str = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared"
    TRAIN_DIR: str = "train"
    VAL_DIR: str = "val"
    
    USE_RGB: bool = True
    USE_MS: bool = True
    USE_HS: bool = True
    
    IMG_SIZE: int = 64
    BATCH_SIZE: int = 128
    EPOCHS: int = 100
    LR: float = 2e-4
    WD: float = 0.05
    
    NUM_WORKERS: int = 4
    SEED: int = 3557
    
    RGB_BACKBONE: str = "efficientnet_b0"
    MS_BACKBONE: str = "efficientnet_b0"
    HS_BACKBONE: str = "resnet18"
    
    HS_DROP_FIRST: int = 10
    HS_DROP_LAST: int = 14
    
    OUT_DIR: str = "./outputs"


LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}