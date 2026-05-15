import json
import joblib
import os
import random
import gc
import numpy as np
import lightgbm as lgb
from feature_generation import apply_preprocessing, generate_dataset_shards_from_folder

import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def _load_shard(shard_path):
    """Load shard from npz file, returning X, y, w, is_extreme."""
    data = np.load(shard_path, allow_pickle=False)
    return (
        data['X'].astype(np.float32, copy=False),
        data['y'].astype(np.float32, copy=False),
        data['w'].astype(np.float32, copy=False),
        data['is_extreme'].astype(bool, copy=False),
    )


def load_params(path='model_params.json'):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _winsorize_labels(y: np.ndarray, limit: float = 0.005) -> np.ndarray:
    """Clip extreme label values to reduce impact of outliers on MSE training.
    limit is the fraction to clip from each tail (default 0.5%)."""
    y = np.asarray(y, dtype=float)
    lower = np.percentile(y, limit * 100)
    upper = np.percentile(y, (1.0 - limit) * 100)
    return np.clip(y, lower, upper)


def _train_valid_split_indices(n_samples, valid_ratio=0.15, purge=30):
    """Temporal split with purge gap to avoid label autocorrelation leakage."""
    if n_samples < purge + 10:
        return None, None
    split = int(n_samples * (1.0 - valid_ratio))
    split = max(purge, min(split, n_samples - purge - 1))
    train_idx = np.arange(0, split - purge, dtype=int)
    valid_idx = np.arange(split, n_samples, dtype=int)
    if train_idx.size < 10 or valid_idx.size < 10:
        return None, None
    return train_idx, valid_idx

def feval_ic(preds, dataset):
    preds = np.nan_to_num(np.asarray(preds, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    labels = dataset.get_label()
    labels = np.nan_to_num(np.asarray(labels, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(labels) < 2:
        return 'ic', 0.0, True
    corr = np.corrcoef(preds, labels)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    return 'ic', float(corr), True

def safe_ic(preds, labels):
    preds = np.nan_to_num(np.asarray(preds, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.nan_to_num(np.asarray(labels, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if preds.size < 2 or labels.size < 2:
        return 0.0
    corr = np.corrcoef(preds, labels)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)

def evaluate_fold_ic(preds, labels, is_extreme):
    is_extreme = np.asarray(is_extreme, dtype=bool)
    labels = np.asarray(labels, dtype=float)
    preds = np.asarray(preds, dtype=float)
    normal_mask = ~is_extreme
    extreme_mask = is_extreme
    return {
        'normal_ic': safe_ic(preds[normal_mask], labels[normal_mask]),
        'extreme_ic': safe_ic(preds[extreme_mask], labels[extreme_mask]),
    }

def get_feature_names(n_features):
    return [f'feat_{idx}' for idx in range(n_features)]

def save_feature_importance(booster, feature_names, out_model, target_name):
    importance_gain = booster.feature_importance(importance_type='gain')
    importance_split = booster.feature_importance(importance_type='split')
    feature_table = []
    for feature_name, gain_value, split_value in zip(feature_names, importance_gain, importance_split):
        feature_table.append({
            'feature': feature_name,
            'gain': float(gain_value),
            'split': int(split_value),
        })
    feature_table.sort(key=lambda item: item['gain'], reverse=True)
    importance_bundle = {
        'target': target_name,
        'features': feature_table,
    }
    importance_path = f'{out_model}.{target_name}.importance.json'
    with open(importance_path, 'w', encoding='utf-8') as f:
        json.dump(importance_bundle, f, ensure_ascii=False, indent=2)
    top_lines = [f"{item['feature']},{item['gain']:.12f},{item['split']}" for item in feature_table]
    with open(f'{out_model}.{target_name}.importance.csv', 'w', encoding='utf-8') as f:
        f.write('feature,gain,split\n')
        f.write('\n'.join(top_lines))
    return importance_bundle

def _plot_base_path(out_model):
    return os.path.splitext(out_model)[0]

def _save_if_possible(fig, out_path):
    if plt is None or fig is None:
        return
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def save_target_visualizations(out_model, target_name, importance_table, oof_preds, oof_labels, oof_extreme, summary_dict):
    if plt is None:
        return

    base = _plot_base_path(out_model)
    vis_dir = f'{base}.visuals'
    os.makedirs(vis_dir, exist_ok=True)

    # 1) Feature importance top-20
    top_features = importance_table[:20]
    if top_features:
        names = [item['feature'] for item in top_features][::-1]
        gains = [item['gain'] for item in top_features][::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, gains, color='#2a6fdb')
        ax.set_title(f'{target_name} feature importance (top 20)')
        ax.set_xlabel('gain')
        ax.grid(axis='x', alpha=0.2)
        _save_if_possible(fig, os.path.join(vis_dir, f'{target_name}.feature_importance.png'))

    # 2) Prediction vs label (subsample to keep the figure light)
    if oof_preds.size and oof_labels.size:
        rng = np.random.default_rng(SEED)
        sample_n = min(5000, oof_preds.size)
        if oof_preds.size > sample_n:
            idx = rng.choice(oof_preds.size, sample_n, replace=False)
            preds_plot = np.asarray(oof_preds)[idx]
            labels_plot = np.asarray(oof_labels)[idx]
            extreme_plot = np.asarray(oof_extreme)[idx]
        else:
            preds_plot = np.asarray(oof_preds)
            labels_plot = np.asarray(oof_labels)
            extreme_plot = np.asarray(oof_extreme)

        fig, ax = plt.subplots(figsize=(7, 7))
        normal_mask = ~extreme_plot
        if normal_mask.any():
            ax.scatter(labels_plot[normal_mask], preds_plot[normal_mask], s=8, alpha=0.35, c='#1f77b4', label='normal')
        if extreme_plot.any():
            ax.scatter(labels_plot[extreme_plot], preds_plot[extreme_plot], s=8, alpha=0.35, c='#d62728', label='extreme')
        min_v = float(min(labels_plot.min(), preds_plot.min()))
        max_v = float(max(labels_plot.max(), preds_plot.max()))
        ax.plot([min_v, max_v], [min_v, max_v], linestyle='--', color='gray', linewidth=1)
        ax.set_title(f'{target_name} OOF prediction vs label')
        ax.set_xlabel('label')
        ax.set_ylabel('prediction')
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        _save_if_possible(fig, os.path.join(vis_dir, f'{target_name}.oof_scatter.png'))

    # 3) Summary bar for normal/extreme IC
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['normal_ic', 'extreme_ic']
    values = [float(summary_dict.get('normal_ic', 0.0)), float(summary_dict.get('extreme_ic', 0.0))]
    colors = ['#1f77b4', '#d62728']
    ax.bar(labels, values, color=colors)
    ax.set_title(f'{target_name} IC summary')
    ax.set_ylim(min(-1.0, min(values) - 0.05), max(1.0, max(values) + 0.05))
    ax.grid(axis='y', alpha=0.2)
    for idx, value in enumerate(values):
        ax.text(idx, value, f'{value:.4f}', ha='center', va='bottom' if value >= 0 else 'top')
    _save_if_possible(fig, os.path.join(vis_dir, f'{target_name}.ic_summary.png'))

def save_classifier_visualization(out_model, oof_labels, oof_preds, summary_dict):
    if plt is None or not np.asarray(oof_preds).size:
        return

    base = _plot_base_path(out_model)
    vis_dir = f'{base}.visuals'
    os.makedirs(vis_dir, exist_ok=True)

    labels = np.asarray(oof_labels).astype(int, copy=False)
    preds = np.asarray(oof_preds, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 4))
    if (labels == 0).any():
        ax.hist(preds[labels == 0], bins=40, alpha=0.6, color='#1f77b4', label='normal')
    if (labels == 1).any():
        ax.hist(preds[labels == 1], bins=40, alpha=0.6, color='#d62728', label='extreme')
    ax.set_title('Extreme classifier OOF probability distribution')
    ax.set_xlabel('predicted probability')
    ax.set_ylabel('count')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.2)
    _save_if_possible(fig, os.path.join(vis_dir, 'classifier.oof_probability_hist.png'))

    fig, ax = plt.subplots(figsize=(6, 4))
    auc_value = float(summary_dict.get('valid_auc', 0.0))
    ax.bar(['valid_auc'], [auc_value], color='#2a6fdb')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Extreme classifier validation AUC')
    ax.text(0, auc_value, f'{auc_value:.4f}', ha='center', va='bottom')
    ax.grid(axis='y', alpha=0.2)
    _save_if_possible(fig, os.path.join(vis_dir, 'classifier.valid_auc.png'))

def train_one_target(shards, params, target_index, target_name, out_model):
    fold_metrics = []
    best_rounds = []
    all_valid_preds = []
    all_valid_labels = []
    all_valid_extreme = []
    boosters = []
    print(f'Starting shard-wise training for {target_name} with {len(shards)} shards...')
    for shard_index, shard_meta in enumerate(shards, 1):
        shard_path = shard_meta['path']
        print(f'  -> [{shard_index}/{len(shards)}] {target_name} loading {shard_path}')
        X, shard_y_all, w, is_extreme = _load_shard(shard_path)
        y = shard_y_all[:, target_index]

        train_idx, valid_idx = _train_valid_split_indices(len(y), valid_ratio=0.2)
        if train_idx is None:
            print(f'     -> skipped shard (too few samples): {len(y)}')
            continue

        train_set = lgb.Dataset(X[train_idx], label=y[train_idx], weight=w[train_idx])
        valid_set = lgb.Dataset(X[valid_idx], label=y[valid_idx], weight=w[valid_idx], reference=train_set)
        # Call LightGBM training with compatibility fallbacks.
        # Some LightGBM builds may not accept `early_stopping_rounds` or `verbose_eval` kwarg,
        # so try the common call first and fall back to using callbacks or a minimal call.
        try:
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=300,
                valid_sets=[valid_set],
                feval=feval_ic,
                early_stopping_rounds=20,
                verbose_eval=50,
            )  
        except TypeError as e:
            err = str(e) 
            # Prepare a basic kwargs dict for fallback calls
            train_kwargs = dict(
                params=params, 
                train_set=train_set,
                num_boost_round=300,
                valid_sets=[valid_set],
                feval=feval_ic,
            )
            # If LightGBM exposes callback helpers, use them for early stopping and logging
            callbacks = []
            if hasattr(lgb, 'early_stopping'):
                try:
                    callbacks.append(lgb.early_stopping(50))
                except Exception:
                    pass
            if hasattr(lgb, 'log_evaluation'):
                try:
                    callbacks.append(lgb.log_evaluation(50))
                except Exception:
                    pass

            if callbacks:
                booster = lgb.train(**train_kwargs, callbacks=callbacks)
            else:
                # Last resort: train without early stopping/logging args
                booster = lgb.train(**train_kwargs)
        best_iteration = booster.best_iteration or 300
        best_rounds.append(best_iteration)
        boosters.append(booster)
        fold_preds = booster.predict(X[valid_idx], num_iteration=best_iteration)
        ic_metrics = evaluate_fold_ic(fold_preds, y[valid_idx], is_extreme[valid_idx])
        fold_metrics.append(ic_metrics)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        all_valid_preds.append(np.asarray(fold_preds, dtype=np.float32)) 
        all_valid_labels.append(np.asarray(y[valid_idx], dtype=np.float32))
        all_valid_extreme.append(np.asarray(is_extreme[valid_idx], dtype=bool))
        np.save(f'{out_model}.{target_name}.oof_pred.shard{shard_index:03d}.npy', np.asarray(fold_preds, dtype=np.float32))
        np.save(f'{out_model}.{target_name}.oof_label.shard{shard_index:03d}.npy', np.asarray(y[valid_idx], dtype=np.float32))
        np.save(f'{out_model}.{target_name}.oof_is_extreme.shard{shard_index:03d}.npy', np.asarray(is_extreme[valid_idx], dtype=bool))
        print(
            f'{target_name} shard {shard_index}: '
            f'normal_ic={ic_metrics["normal_ic"]:.6f}, '
            f'extreme_ic={ic_metrics["extreme_ic"]:.6f}, '
            f'best_round={best_iteration}'
        )
        del X, y_all, y, w, is_extreme, train_set, valid_set
        gc.collect()

    if not boosters:
        raise ValueError(f'No valid shards to train {target_name}.')

    avg_best_round = int(round(float(np.mean(best_rounds)))) if best_rounds else 1000
    oof_preds = np.concatenate(all_valid_preds) if all_valid_preds else np.zeros((0,), dtype=np.float32)
    oof_labels = np.concatenate(all_valid_labels) if all_valid_labels else np.zeros((0,), dtype=np.float32)
    oof_extreme = np.concatenate(all_valid_extreme) if all_valid_extreme else np.zeros((0,), dtype=bool)
    overall_ic = safe_ic(oof_preds, oof_labels)
    overall_normal_ic = safe_ic(oof_preds[~oof_extreme], oof_labels[~oof_extreme]) if oof_preds.size else 0.0
    overall_extreme_ic = safe_ic(oof_preds[oof_extreme], oof_labels[oof_extreme]) if oof_preds.size else 0.0
    print(
        f'{target_name} CV summary: overall_ic={overall_ic:.6f}, '
        f'normal_ic={overall_normal_ic:.6f}, extreme_ic={overall_extreme_ic:.6f}, '
        f'avg_best_round={avg_best_round}'
    )
    return boosters, avg_best_round, {
        'overall_ic': overall_ic,
        'normal_ic': overall_normal_ic,
        'extreme_ic': overall_extreme_ic,
        'fold_metrics': fold_metrics,
    }, oof_preds, oof_labels, oof_extreme


def train_extreme_classifier(shards, params, out_model, preprocessing_params):
    """Train a binary classifier to predict probability of extreme行情.
    Returns (booster, best_round, summary_dict).
    """
    # To avoid OOM when datasets are huge, train classifier incrementally per-shard.
    # Configurable knobs (can be set in model_params.json):
    #  - 'classifier_rounds_per_shard' (int): num_boost_round used on each shard (default 200)
    #  - 'classifier_subsample_frac' (float): fraction of rows to sample from each shard (default 1.0)
    rounds_per_shard = int(params.get('classifier_rounds_per_shard', 200))
    subsample_frac = float(params.get('classifier_subsample_frac', 1.0))

    clf_params = dict(params)
    clf_params['objective'] = 'binary'
    clf_params['metric'] = 'auc'
    # Moderate regularization for classifier
    clf_params['num_leaves'] = 15
    clf_params['max_depth'] = 4
    clf_params['min_data_in_leaf'] = 200
    clf_params['lambda_l1'] = 3.0
    clf_params['lambda_l2'] = 5.0
    clf_params['feature_fraction'] = 0.4
    clf_params['bagging_fraction'] = 0.5
    clf_params.setdefault('verbose', -1)

    bst = None
    best_rounds = []
    all_valid_preds = []
    all_valid_labels = []

    for shard_index, shard_meta in enumerate(shards, 1):
        shard_path = shard_meta['path']
        print(f'  -> [{shard_index}/{len(shards)}] classifier loading {shard_path}')
        X, _, _, is_extreme = _load_shard(shard_path)
        X = apply_preprocessing(X, preprocessing_params, target_name='ret5')

        n = X.shape[0]
        if n < 3:
            print(f'     -> skipped shard (too few samples): {n}')
            continue

        # optional subsample to reduce memory/time per shard
        if subsample_frac < 1.0:
            keep_n = max(3, int(n * subsample_frac))
            idx = np.random.choice(n, keep_n, replace=False)
            X = X[idx]
            is_extreme = is_extreme[idx]
            n = X.shape[0]

        train_idx, valid_idx = _train_valid_split_indices(n, valid_ratio=0.2)
        if train_idx is None:
            print(f'     -> skipped shard after split (too few samples): {n}')
            continue

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = is_extreme[train_idx].astype(np.int8), is_extreme[valid_idx].astype(np.int8)

        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

        try:
            bst = lgb.train(
                clf_params,
                train_set,
                num_boost_round=rounds_per_shard,
                valid_sets=[valid_set],
                early_stopping_rounds=20,
                verbose_eval=50,
                init_model=bst,
            )
        except TypeError:
            # fallback to callback style, pass init_model via train_kwargs if supported
            train_kwargs = dict(
                params=clf_params,
                train_set=train_set,
                num_boost_round=rounds_per_shard,
                valid_sets=[valid_set],
            )
            callbacks = []
            if hasattr(lgb, 'early_stopping'):
                try:
                    callbacks.append(lgb.early_stopping(50))
                except Exception:
                    pass
            if hasattr(lgb, 'log_evaluation'):
                try:
                    callbacks.append(lgb.log_evaluation(50))
                except Exception:
                    pass
            # try to include init_model if possible
            try:
                bst = lgb.train(**train_kwargs, callbacks=callbacks, init_model=bst)
            except TypeError:
                # last resort: ignore init_model
                if callbacks:
                    bst = lgb.train(**train_kwargs, callbacks=callbacks)
                else:
                    bst = lgb.train(**train_kwargs)

        best_iter = int(bst.best_iteration or rounds_per_shard)
        best_rounds.append(best_iter)

        # OOF preds for this shard
        preds_valid = bst.predict(X_valid, num_iteration=best_iter)
        all_valid_preds.append(np.asarray(preds_valid, dtype=np.float32))
        all_valid_labels.append(np.asarray(y_valid, dtype=np.int8))
        np.save(f'{out_model}.extreme.oof_pred.shard{shard_index:03d}.npy', np.asarray(preds_valid, dtype=np.float32))
        np.save(f'{out_model}.extreme.oof_label.shard{shard_index:03d}.npy', np.asarray(y_valid, dtype=np.int8))

        print(f'    shard {shard_index}: best_round={best_iter}, valid_pos_frac={(y_valid.sum()/len(y_valid)):.4f}')

        # free shard memory
        del X, is_extreme, X_train, X_valid, y_train, y_valid, train_set, valid_set
        gc.collect()

    if bst is None:
        raise ValueError('No data available to train extreme classifier')

    # aggregate OOF preds for overall validation AUC
    oof_preds = np.concatenate(all_valid_preds) if all_valid_preds else np.zeros((0,), dtype=np.float32)
    oof_labels = np.concatenate(all_valid_labels) if all_valid_labels else np.zeros((0,), dtype=np.int8)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(oof_labels, oof_preds)) if oof_labels.size else 0.0
    except Exception:
        auc = 0.0

    avg_best_round = int(round(float(np.mean(best_rounds)))) if best_rounds else rounds_per_shard
    summary = {'valid_auc': auc, 'best_round': int(avg_best_round)}
    print(f'Extreme classifier CV (agg): valid_auc={auc:.6f}, avg_best_round={avg_best_round}')
    save_classifier_visualization(out_model, oof_labels, oof_preds, summary)
    return bst, avg_best_round, summary


def train_split_regressors(shards, params, target_index, target_name, out_model, preprocessing_params, classifier_booster=None):
    """Train normal/extreme regressors shard-wise to preserve temporal split semantics.
    For ret60, use milder split with soft fusion.
    OOF fusion uses classifier probability (or fixed fallback), not true regime routing.
    """
    # ret60: train on full data (weaker signal, needs all samples)
    # ret5: split by regime (stronger signal, benefits from specialization)
    use_split = (target_name != 'ret60')

    if use_split:
        booster_groups = {'normal': [], 'extreme': []}
        best_round_lists = {'normal': [], 'extreme': []}
    else:
        booster_groups = {'full': []}
        best_round_lists = {'full': []}

    oof_preds_list = []
    oof_labels_list = []
    oof_extreme_list = []

    huber_alpha = 0.9 if target_name == 'ret5' else 0.85

    for shard_index, shard_meta in enumerate(shards, 1):
        shard_path = shard_meta['path']
        X, shard_y_all, w, is_extreme = _load_shard(shard_path)
        y = shard_y_all[:, target_index]
        X = apply_preprocessing(X, preprocessing_params, target_name=target_name)

        train_idx, valid_idx = _train_valid_split_indices(len(y), valid_ratio=0.2)
        if train_idx is None:
            print(f'  -> [{shard_index}/{len(shards)}] skip {target_name} (too few samples)')
            continue

        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train_orig, y_valid = y[train_idx], y[valid_idx]
        w_train, w_valid = w[train_idx], w[valid_idx]
        is_ext_train, is_ext_valid = is_extreme[train_idx], is_extreme[valid_idx]

        y_train = _winsorize_labels(y_train_orig, limit=0.005)

        if not use_split:
            # ret60: unified model on full data
            if X_train.shape[0] < 3 or X_valid.shape[0] < 1:
                continue

            tr_set = lgb.Dataset(X_train, label=y_train, weight=w_train)
            va_set = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=tr_set)

            reg_params = dict(params)
            reg_params['objective'] = 'huber'
            reg_params['alpha'] = huber_alpha
            reg_params.setdefault('verbose', -1)

            try:
                bst = lgb.train(reg_params, tr_set, num_boost_round=300,
                                valid_sets=[va_set], feval=feval_ic,
                                early_stopping_rounds=20, verbose_eval=50)
            except TypeError:
                train_kwargs = dict(params=reg_params, train_set=tr_set, num_boost_round=300,
                                    valid_sets=[va_set], feval=feval_ic)
                callbacks = []
                if hasattr(lgb, 'early_stopping'):
                    try: callbacks.append(lgb.early_stopping(50))
                    except Exception: pass
                if hasattr(lgb, 'log_evaluation'):
                    try: callbacks.append(lgb.log_evaluation(50))
                    except Exception: pass
                bst = lgb.train(**train_kwargs, callbacks=callbacks) if callbacks else lgb.train(**train_kwargs)

            best_iteration = int(bst.best_iteration or 300)
            booster_groups['full'].append(bst)
            best_round_lists['full'].append(best_iteration)
            fused_pred = bst.predict(X_valid, num_iteration=best_iteration)

        else:
            # ret5: split by regime
            shard_models = {'normal': None, 'extreme': None}
            shard_rounds = {'normal': 0, 'extreme': 0}

            for split_name, mask_train, mask_valid in (
                ('normal', ~is_ext_train, ~is_ext_valid),
                ('extreme', is_ext_train, is_ext_valid),
            ):
                if mask_train.sum() < 3 or mask_valid.sum() < 1:
                    continue

                tr_set = lgb.Dataset(X_train[mask_train], label=y_train[mask_train], weight=w_train[mask_train])
                va_set = lgb.Dataset(X_valid[mask_valid], label=y_valid[mask_valid], weight=w_valid[mask_valid], reference=tr_set)

                reg_params = dict(params)
                reg_params['objective'] = 'huber'
                reg_params['alpha'] = huber_alpha
                reg_params.setdefault('verbose', -1)

                try:
                    bst = lgb.train(reg_params, tr_set, num_boost_round=300,
                                    valid_sets=[va_set], feval=feval_ic,
                                    early_stopping_rounds=20, verbose_eval=50)
                except TypeError:
                    train_kwargs = dict(params=reg_params, train_set=tr_set, num_boost_round=300,
                                        valid_sets=[va_set], feval=feval_ic)
                    callbacks = []
                    if hasattr(lgb, 'early_stopping'):
                        try: callbacks.append(lgb.early_stopping(50))
                        except Exception: pass
                    if hasattr(lgb, 'log_evaluation'):
                        try: callbacks.append(lgb.log_evaluation(50))
                        except Exception: pass
                    bst = lgb.train(**train_kwargs, callbacks=callbacks) if callbacks else lgb.train(**train_kwargs)

                best_iteration = int(bst.best_iteration or 300)
                shard_models[split_name] = bst
                shard_rounds[split_name] = best_iteration
                booster_groups[split_name].append(bst)
                best_round_lists[split_name].append(best_iteration)

            normal_pred = None
            extreme_pred = None
            if shard_models['normal'] is not None:
                normal_pred = shard_models['normal'].predict(X_valid, num_iteration=shard_rounds['normal'])
            if shard_models['extreme'] is not None:
                extreme_pred = shard_models['extreme'].predict(X_valid, num_iteration=shard_rounds['extreme'])

            if normal_pred is not None and extreme_pred is not None:
                if classifier_booster is not None:
                    p_ext = classifier_booster.predict(X_valid)
                    p_ext = np.clip(p_ext, 0.02, 0.98)
                    fused_pred = normal_pred * (1.0 - p_ext) + extreme_pred * p_ext
                else:
                    fused_pred = 0.5 * normal_pred + 0.5 * extreme_pred
            elif extreme_pred is not None:
                fused_pred = extreme_pred
            elif normal_pred is not None:
                fused_pred = normal_pred
            else:
                continue

        shard_metrics = evaluate_fold_ic(fused_pred, y_valid, is_ext_valid)
        print(
            f'  -> [{shard_index}/{len(shards)}] {target_name} '
            f'{"fused" if use_split else "full"}: '
            f'normal_ic={shard_metrics["normal_ic"]:.6f}, '
            f'extreme_ic={shard_metrics["extreme_ic"]:.6f}'
        )

        np.save(f'{out_model}.{target_name}.oof_pred.shard{shard_index:03d}.npy',
                np.asarray(fused_pred, dtype=np.float32))
        np.save(f'{out_model}.{target_name}.oof_label.shard{shard_index:03d}.npy',
                np.asarray(y_valid, dtype=np.float32))
        np.save(f'{out_model}.{target_name}.oof_is_extreme.shard{shard_index:03d}.npy',
                np.asarray(is_ext_valid, dtype=bool))

        oof_preds_list.append(np.asarray(fused_pred, dtype=np.float32))
        oof_labels_list.append(np.asarray(y_valid, dtype=np.float32))
        oof_extreme_list.append(np.asarray(is_ext_valid, dtype=bool))

        del X, shard_y_all, y, is_extreme
        gc.collect()

    if not use_split:
        if not booster_groups.get('full', []):
            raise ValueError(f'No valid models trained for {target_name}')
        boost_summary = {
            'full': {'n_models': len(booster_groups['full']),
                     'avg_best_round': int(round(float(np.mean(best_round_lists['full'])))) if best_round_lists['full'] else 0}
        }
    else:
        split_names = ('normal', 'extreme')
        if not any(booster_groups.get(sn, []) for sn in split_names):
            raise ValueError(f'No valid split models trained for {target_name}')
        boost_summary = {
            sn: {'n_models': len(booster_groups[sn]),
                 'avg_best_round': int(round(float(np.mean(best_round_lists[sn])))) if best_round_lists[sn] else 0}
            for sn in split_names
        }

    oof_preds = np.concatenate(oof_preds_list) if oof_preds_list else np.zeros((0,), dtype=np.float32)
    oof_labels = np.concatenate(oof_labels_list) if oof_labels_list else np.zeros((0,), dtype=np.float32)
    oof_extreme = np.concatenate(oof_extreme_list) if oof_extreme_list else np.zeros((0,), dtype=bool)

    summary = {
        'overall_ic': safe_ic(oof_preds, oof_labels),
        'normal_ic': safe_ic(oof_preds[~oof_extreme], oof_labels[~oof_extreme]) if oof_preds.size else 0.0,
        'extreme_ic': safe_ic(oof_preds[oof_extreme], oof_labels[oof_extreme]) if oof_preds.size else 0.0,
        'split_models': boost_summary,
    }
    print(
        f'{target_name} split CV summary: overall_ic={summary["overall_ic"]:.6f}, '
        f'normal_ic={summary["normal_ic"]:.6f}, extreme_ic={summary["extreme_ic"]:.6f}'
    )

    representative = {}
    best_rounds = {}
    if not use_split:
        representative['full'] = booster_groups.get('full', [])[-1] if booster_groups.get('full', []) else None
        if 'full' in boost_summary:
            best_rounds['full'] = boost_summary['full'].get('avg_best_round', 0)
    else:
        for split_name in ('normal', 'extreme'):
            representative[split_name] = booster_groups.get(split_name, [])[-1] if booster_groups.get(split_name, []) else None
            if split_name in boost_summary:
                best_rounds[split_name] = boost_summary[split_name].get('avg_best_round', 0)
    return representative, best_rounds, summary, oof_preds, oof_labels, oof_extreme

def train_and_save(data_folder='train_dataset', out_model='lgb_model.joblib', window=60, shard_dir='train_shards'):
    import json as _json
    manifest_path = os.path.join(shard_dir, 'manifest.json')
    if os.path.exists(manifest_path):
        print(f'Loading existing manifest from {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = _json.load(f)
    else:
        print(f'No manifest found at {manifest_path}, generating shards...')
        manifest = generate_dataset_shards_from_folder(data_folder, shard_dir=shard_dir, window=window)
    shards = manifest.get('shards', [])
    if not shards:
        print('No training data found.')
        return
    preprocessing_params = manifest.get('preprocessing_params', {})
    feature_count = int(manifest.get('feature_count', preprocessing_params.get('feature_count', 0)))
    print(f"Shard summary: shards={len(shards)}, total_samples={manifest.get('total_samples', 0)}, feature_count={feature_count}")

    params = load_params()
    params.setdefault('seed', SEED)
    params.setdefault('bagging_seed', SEED)
    params.setdefault('feature_fraction_seed', SEED)
    params.setdefault('data_random_seed', SEED)
    params.setdefault('drop_seed', SEED)
    params.setdefault('deterministic', True)
    params['metric'] = 'None'

    # Train extreme classifier first
    print('Training extreme/normal classifier...')
    clf_bst, clf_best_round, clf_summary = train_extreme_classifier(shards, params, out_model, preprocessing_params)
    classifier_str = clf_bst.model_to_string()

    target_names = ['ret5', 'ret60']
    boosters = {}
    best_rounds = {}
    validation_summary = {}
    importance_summary = {}
    oof_summary = {}
    feature_names = get_feature_names(feature_count)
    for target_index, target_name in enumerate(target_names):
        print(f'Training split regressors for {target_name}...')
        bst_dict, best_rounds_dict, summary_dict, oof_preds, oof_labels, oof_extreme = train_split_regressors(
            shards,
            params,
            target_index,
            target_name,
            out_model,
            preprocessing_params,
            classifier_booster=clf_bst,
        )
        # store model strings if present
        models_entry = {}
        importance_rows = []
        split_names = tuple(bst_dict.keys()) if isinstance(bst_dict, dict) else ('normal', 'extreme')
        for split_name in split_names:
            bst = bst_dict.get(split_name)
            if bst is not None:
                models_entry[split_name] = bst.model_to_string()
                gain = bst.feature_importance(importance_type='gain')
                split = bst.feature_importance(importance_type='split')
                importance_rows.append((gain, split))
            else:
                models_entry[split_name] = None

        boosters[target_name] = models_entry
        best_rounds[target_name] = best_rounds_dict
        validation_summary[target_name] = summary_dict

        avg_gain = np.mean([row[0] for row in importance_rows], axis=0) if importance_rows else np.zeros((feature_count,), dtype=float)
        avg_split = np.mean([row[1] for row in importance_rows], axis=0) if importance_rows else np.zeros((feature_count,), dtype=float)
        merged_table = [
            {'feature': name, 'gain': float(g), 'split': int(round(s))}
            for name, g, s in zip(feature_names, avg_gain, avg_split)
        ]
        merged_table.sort(key=lambda item: item['gain'], reverse=True)
        importance_bundle = {'target': target_name, 'features': merged_table}
        with open(f'{out_model}.{target_name}.importance.json', 'w', encoding='utf-8') as f:
            json.dump(importance_bundle, f, ensure_ascii=False, indent=2)
        with open(f'{out_model}.{target_name}.importance.csv', 'w', encoding='utf-8') as f:
            f.write('feature,gain,split\n')
            for item in merged_table:
                f.write(f"{item['feature']},{item['gain']:.12f},{item['split']}\n")
        save_target_visualizations(out_model, target_name, merged_table, oof_preds, oof_labels, oof_extreme, summary_dict)
        importance_summary[target_name] = importance_bundle
        oof_summary[target_name] = {
            'predictions': oof_preds,
            'labels': oof_labels,
            'is_extreme': oof_extreme,
        }
        np.save(f'{out_model}.{target_name}.oof_pred.npy', oof_preds)
        np.save(f'{out_model}.{target_name}.oof_label.npy', oof_labels)
        np.save(f'{out_model}.{target_name}.oof_is_extreme.npy', oof_extreme)

    bundle = {
        'n_features': feature_count,
        'target_names': target_names,
        'best_rounds': best_rounds,
        'models': boosters,
        'extreme_classifier': classifier_str,
        'preprocessing_params': preprocessing_params,
        'validation_summary': validation_summary,
        'feature_importance': importance_summary,
        'oof_predictions': oof_summary,
        'shard_manifest': manifest,
    }
    joblib.dump(bundle, out_model)
    # 保存特征维度信息
    joblib.dump({'n_features': feature_count}, out_model + '.meta.pkl')
    print('Model saved to', out_model)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', default='train_dataset')
    parser.add_argument('--out-model', default='lgb_model.joblib')
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--shard-dir', default='train_shards')
    args = parser.parse_args()
    train_and_save(args.data_folder, args.out_model, window=args.window, shard_dir=args.shard_dir)
