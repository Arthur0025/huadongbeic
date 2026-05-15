import numpy as np
import os
import sys
# Ensure local workspace directory is on sys.path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_generation import build_features_from_ohlcv

def generate_factors(dataset_name: str, data: np.ndarray) -> np.ndarray:
    """平台入口函数：接收 dataset_name 和 OHLCV 数据，返回 float32 的特征矩阵。"""
    _ = dataset_name
    ohlcv = np.asarray(data, dtype=float)
    X, idxs = build_features_from_ohlcv(ohlcv)
    N = len(ohlcv)
    D = X.shape[1] if X.ndim == 2 else 0
    full = np.full((N, D), np.nan, dtype=np.float32)
    if X.size:
        full[idxs, :] = X.astype(np.float32, copy=False)
    return full

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', required=True)
    args = parser.parse_args()
    arr = np.load(args.infile)
    feats = generate_factors('cli', arr)
    np.save(args.outfile, feats)
