import os

import joblib
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.utils import (
    make_df,
    read_hs,
    read_ms,
    read_rgb,
    stratified_holdout,
)


def calculate_stats(cfg, verbose=True, fit_pca=False, pca_path="pca_hs.pkl"):
    train_df = make_df(cfg.ROOT, cfg.TRAIN_DIR)
    train_df, _ = stratified_holdout(train_df, frac=0.1, seed=cfg.SEED)
    
    if verbose:
        print(f"Calculating statistics from {len(train_df)} samples...")
    
    rgb_vals, ms_vals, hs_vals, hs_shapes = [], [], [], []
    
    for idx in tqdm(range(len(train_df)), desc="Loading", disable=not verbose):
        row = train_df.iloc[idx]
        
        if cfg.USE_RGB and 'rgb' in row and row['rgb']:
            rgb_vals.append(read_rgb(row['rgb']))
        
        if cfg.USE_MS and 'ms' in row and row['ms']:
            ms_vals.append(read_ms(row['ms'], cfg))
        
        if cfg.USE_HS and 'hs' in row and row['hs']:
            hs = read_hs(row['hs'], cfg.HS_DROP_FIRST, cfg.HS_DROP_LAST)
            hs_vals.append(hs)
            hs_shapes.append(hs.shape)
    
    stats = {}
    
    if rgb_vals:
        rgb_stacked = []
        for rgb in rgb_vals:
            C, H, W = rgb.shape
            rgb_stacked.append(rgb.reshape(C, -1))
        rgb_all = torch.cat(rgb_stacked, dim=1)
        stats['rgb_mean'] = tuple(rgb_all.mean(dim=1).tolist())
        stats['rgb_std'] = tuple(rgb_all.std(dim=1).tolist())
    
    if ms_vals:
        ms_stacked = []
        for ms in ms_vals:
            C, H, W = ms.shape
            ms_stacked.append(ms.reshape(C, -1))
        ms_all = torch.cat(ms_stacked, dim=1)
        stats['ms_mean'] = tuple(ms_all.mean(dim=1).tolist())
        stats['ms_std'] = tuple(ms_all.std(dim=1).tolist())
    
    if hs_vals:
        hs_max_ch = max(s[0] for s in hs_shapes)
        hs_stacked = []
        for hs in hs_vals:
            C, H, W = hs.shape
            if C < hs_max_ch:
                pad = torch.zeros(hs_max_ch - C, H, W)
                hs = torch.cat([hs, pad], 0)
            hs_stacked.append(hs.reshape(hs_max_ch, -1))
        hs_all = torch.cat(hs_stacked, dim=1)
        stats['hs_mean'] = tuple(hs_all.mean(dim=1).tolist())
        stats['hs_std'] = tuple(hs_all.std(dim=1).tolist())
        stats['hs_channels'] = hs_max_ch
        
        if fit_pca and hasattr(cfg, 'PCA_COMPONENTS') and cfg.PCA_COMPONENTS > 0:
            if verbose:
                print(f"\nFitting PCA: {hs_max_ch} channels → {cfg.PCA_COMPONENTS} components...")
            
            hs_flat_list = []
            for hs in hs_vals:
                C, H, W = hs.shape
                if C < hs_max_ch:
                    pad = torch.zeros(hs_max_ch - C, H, W)
                    hs = torch.cat([hs, pad], 0)
                hs_flat_list.append(hs.permute(1, 2, 0).reshape(-1, hs_max_ch))
            hs_flat = torch.cat(hs_flat_list, dim=0).numpy()
            
            pca = PCA(n_components=cfg.PCA_COMPONENTS)
            pca.n_features_expected = hs_max_ch
            pca.fit(hs_flat)
            
            explained_var = pca.explained_variance_ratio_.sum()
            if verbose:
                print(f"Explained variance: {explained_var:.2%}")
            
            hs_pca_list = []
            for hs in hs_vals:
                C, H, W = hs.shape
                if C < hs_max_ch:
                    pad = torch.zeros(hs_max_ch - C, H, W)
                    hs = torch.cat([hs, pad], 0)
                hs_flat_i = hs.permute(1, 2, 0).reshape(-1, hs_max_ch).numpy()
                hs_pca_i = pca.transform(hs_flat_i).reshape(H, W, cfg.PCA_COMPONENTS)
                hs_pca_tensor_i = torch.from_numpy(hs_pca_i).permute(2, 0, 1).float()
                C_pca, H_pca, W_pca = hs_pca_tensor_i.shape
                hs_pca_list.append(hs_pca_tensor_i.reshape(C_pca, -1))
            
            hs_pca_all = torch.cat(hs_pca_list, dim=1)
            hs_pca_mean = tuple(hs_pca_all.mean(dim=1).tolist())
            hs_pca_std = tuple(hs_pca_all.std(dim=1).tolist())
            
            os.makedirs(os.path.dirname(pca_path) if os.path.dirname(pca_path) else '.', exist_ok=True)
            pca_data = {
                'model': pca,
                'n_features': hs_max_ch,
                'ms_mean': stats.get('ms_mean'),
                'ms_std': stats.get('ms_std'),
                'hs_pca_mean': hs_pca_mean,
                'hs_pca_std': hs_pca_std,
                'pca_explained_variance': float(explained_var)
            }
            joblib.dump(pca_data, pca_path)
            if verbose:
                print(f"PCA saved: {pca_path} | Variance: {explained_var:.1%}")
            
            stats['hs_pca_mean'] = hs_pca_mean
            stats['hs_pca_std'] = hs_pca_std
            stats['pca_explained_variance'] = float(explained_var)
    
    return stats


if __name__ == "__main__":
    from src.config import CFG
    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    cfg.PCA_COMPONENTS = 20
    calculate_stats(cfg, fit_pca=True, pca_path="./outputs/pca_hs.pkl")
