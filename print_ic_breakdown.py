import glob
import json
import os
import re
import numpy as np


def safe_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.nan_to_num(np.asarray(preds, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.nan_to_num(np.asarray(labels, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if preds.size < 2 or labels.size < 2:
        return 0.0
    if np.std(preds) < 1e-12 or np.std(labels) < 1e-12:
        return 0.0
    corr = np.corrcoef(preds, labels)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


def shard_index(path: str) -> int:
    m = re.search(r"shard(\d+)\.npy$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def build_shard_name_map(manifest_path: str) -> dict:
    """读取 manifest.json，建立 shard序号(1-based) → dataset名称 的映射。"""
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    name_map = {}
    for i, shard_meta in enumerate(manifest.get('shards', []), 1):
        name_map[i] = shard_meta.get('dataset', f'dataset{i - 1}')
    return name_map


def load_target_rows(prefix: str, target: str):
    pred_files = sorted(
        glob.glob(f"{prefix}.{target}.oof_pred.shard*.npy"),
        key=shard_index,
    )
    rows = {}
    for pf in pred_files:
        idx = shard_index(pf)
        lf = f"{prefix}.{target}.oof_label.shard{idx:03d}.npy"
        ef = f"{prefix}.{target}.oof_is_extreme.shard{idx:03d}.npy"
        if not (os.path.exists(lf) and os.path.exists(ef)):
            continue
        pred = np.load(pf, allow_pickle=False)
        label = np.load(lf, allow_pickle=False)
        is_ext = np.load(ef, allow_pickle=False).astype(bool)
        normal_mask = ~is_ext
        extreme_mask = is_ext
        rows[idx] = {
            "normal_ic": safe_ic(pred[normal_mask], label[normal_mask]),
            "extreme_ic": safe_ic(pred[extreme_mask], label[extreme_mask]),
        }

    # Fallback: merged OOF arrays
    if not rows:
        pred_path = f"{prefix}.{target}.oof_pred.npy"
        label_path = f"{prefix}.{target}.oof_label.npy"
        extreme_path = f"{prefix}.{target}.oof_is_extreme.npy"
        if os.path.exists(pred_path) and os.path.exists(label_path) and os.path.exists(extreme_path):
            pred = np.load(pred_path, allow_pickle=False)
            label = np.load(label_path, allow_pickle=False)
            is_ext = np.load(extreme_path, allow_pickle=False).astype(bool)
            normal_mask = ~is_ext
            extreme_mask = is_ext
            rows[0] = {
                "normal_ic": safe_ic(pred[normal_mask], label[normal_mask]),
                "extreme_ic": safe_ic(pred[extreme_mask], label[extreme_mask]),
            }
    return rows


def aggregate_ic(rows: dict) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    normal_vals = [item.get("normal_ic", 0.0) for item in rows.values()]
    extreme_vals = [item.get("extreme_ic", 0.0) for item in rows.values()]
    normal_ic = float(np.mean(normal_vals)) if normal_vals else 0.0
    extreme_ic = float(np.mean(extreme_vals)) if extreme_vals else 0.0
    return normal_ic, extreme_ic


def main():
    prefix = "lgb_model.joblib"
    manifest_path = "train_shards/manifest.json"
    name_map = build_shard_name_map(manifest_path)

    ret5 = load_target_rows(prefix, "ret5")
    ret60 = load_target_rows(prefix, "ret60")

    shard_ids = sorted(set(ret5.keys()) | set(ret60.keys()))
    ret5_normal_ic, ret5_extreme_ic = aggregate_ic(ret5)
    ret60_normal_ic, ret60_extreme_ic = aggregate_ic(ret60)

    print("Per-dataset IC breakdown")
    if shard_ids == [0]:
        print("(no per-shard OOF files found; showing merged OOF IC instead)")
    print("Normal IC (non-extreme intervals)")
    print("Dataset\tRet5\tRet60")
    for sid in shard_ids:
        dname = "merged_oof" if sid == 0 else name_map.get(sid, f"dataset{sid - 1}")
        r5 = ret5.get(sid, {}).get("normal_ic", 0.0)
        r60 = ret60.get(sid, {}).get("normal_ic", 0.0)
        print(f"{dname}\t{r5:.6f}\t{r60:.6f}")

    print("Extreme IC (extreme intervals only)")
    print("Dataset\tRet5\tRet60")
    for sid in shard_ids:
        dname = "merged_oof" if sid == 0 else name_map.get(sid, f"dataset{sid - 1}")
        r5 = ret5.get(sid, {}).get("extreme_ic", 0.0)
        r60 = ret60.get(sid, {}).get("extreme_ic", 0.0)
        print(f"{dname}\t{r5:.6f}\t{r60:.6f}")

    print("Aggregate IC over local training shards")
    print(f"Ret5\tnormal_ic={ret5_normal_ic:.6f}\textreme_ic={ret5_extreme_ic:.6f}")
    print(f"Ret60\tnormal_ic={ret60_normal_ic:.6f}\textreme_ic={ret60_extreme_ic:.6f}")


if __name__ == "__main__":
    main()
