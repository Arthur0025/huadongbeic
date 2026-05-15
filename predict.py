import os
import sys
import numpy as np
import joblib
import lightgbm as lgb
# Ensure local workspace directory is on sys.path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_generation import apply_preprocessing
from factor import generate_factors

MODEL_PATH = '/workspace/submission/lgb_model.joblib'


def _resolve_model_path() -> str:
    candidates = []

    env_path = os.environ.get('MODEL_PATH', '').strip()
    if env_path:
        candidates.append(env_path)

    if MODEL_PATH:
        candidates.append(MODEL_PATH)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, 'lgb_model.joblib'))
    candidates.append(os.path.abspath('lgb_model.joblib'))

    seen = set()
    for path in candidates:
        if not path:
            continue
        norm_path = os.path.normpath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        if os.path.exists(norm_path):
            return norm_path

    raise FileNotFoundError(
        'model file not found. tried paths: ' + ', '.join(seen)
    )


def generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray:
    """平台入口：接收 dataset_name 和特征矩阵 factors，返回预测结果。"""
    _ = dataset_name
    X = np.asarray(factors, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    model_path = _resolve_model_path()

    model_obj = joblib.load(model_path)
    preprocessing_params = model_obj.get('preprocessing_params') if isinstance(model_obj, dict) else None

    if not (isinstance(model_obj, dict) and 'models' in model_obj):
        bst = lgb.Booster(model_file=model_path)
        preds = bst.predict(X)
        preds = np.asarray(preds, dtype=np.float32)
        if preds.ndim == 1:
            return np.column_stack([preds, np.zeros_like(preds)]).astype(np.float32, copy=False)
        if preds.ndim == 2 and preds.shape[1] == 2:
            return preds.astype(np.float32, copy=False)
        raise ValueError(f'unexpected prediction shape from single model: {preds.shape}')

    models = model_obj['models']

    classifier_booster = None
    extreme_classifier = model_obj.get('extreme_classifier')
    if isinstance(extreme_classifier, str) and extreme_classifier:
        try:
            classifier_booster = lgb.Booster(model_str=extreme_classifier)
        except Exception:
            classifier_booster = None

    def _predict_model_string(model_str, X_input):
        if not isinstance(model_str, str) or not model_str:
            return None
        return lgb.Booster(model_str=model_str).predict(X_input)

    def _blend_split_predictions(normal_pred, extreme_pred, X_input, target_name='ret5'):
        if normal_pred is None:
            return extreme_pred
        if extreme_pred is None:
            return normal_pred
        if classifier_booster is not None:
            p_ext = classifier_booster.predict(X_input)
            p_ext = np.clip(p_ext, 0.02, 0.98)
            return normal_pred * (1.0 - p_ext) + extreme_pred * p_ext
        return 0.5 * normal_pred + 0.5 * extreme_pred

    def _predict_target(target_name):
        if target_name not in models:
            return None
        target_model = models[target_name]
        X_target = apply_preprocessing(X, preprocessing_params, target_name=target_name)
        if isinstance(target_model, dict):
            full_pred = _predict_model_string(target_model.get('full'), X_target)
            if full_pred is not None:
                return full_pred
            normal_pred = _predict_model_string(target_model.get('normal'), X_target)
            extreme_pred = _predict_model_string(target_model.get('extreme'), X_target)
            return _blend_split_predictions(normal_pred, extreme_pred, X_target, target_name)
        if isinstance(target_model, list):
            if not target_model:
                return None
            preds = []
            for model_str in target_model:
                pred = _predict_model_string(model_str, X_target)
                if pred is not None:
                    preds.append(pred)
            if not preds:
                return None
            return np.mean(np.vstack(preds), axis=0)
        return _predict_model_string(target_model, X_target)

    pred_ret5 = _predict_target('ret5')
    pred_ret60 = _predict_target('ret60')

    if pred_ret5 is not None and pred_ret60 is not None:
        return np.column_stack([pred_ret5, pred_ret60]).astype(np.float32, copy=False)

    if pred_ret5 is not None and pred_ret60 is None:
        ret5_arr = np.asarray(pred_ret5, dtype=np.float32)
        return np.column_stack([ret5_arr, np.zeros_like(ret5_arr)]).astype(np.float32, copy=False)

    if pred_ret60 is not None and pred_ret5 is None:
        ret60_arr = np.asarray(pred_ret60, dtype=np.float32)
        return np.column_stack([np.zeros_like(ret60_arr), ret60_arr]).astype(np.float32, copy=False)

    raise ValueError('no valid model found in bundle')


def predict_from_ohlcv(ohlcv: np.ndarray, dataset_name: str = 'cli') -> np.ndarray:
    factors = generate_factors(dataset_name, ohlcv)
    return generate_signals(dataset_name, factors)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True, help='Input OHLCV npy file')
    parser.add_argument('--model', dest='model_path', default='lgb_model.joblib', help='Model file path')
    parser.add_argument('--out', dest='outfile', required=True, help='Output prediction npy file')
    parser.add_argument('--dataset-name', default='cli', help='Dataset name passed to platform entry')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    arr = np.load(args.infile, allow_pickle=False)
    preds = predict_from_ohlcv(arr, dataset_name=args.dataset_name)
    np.save(args.outfile, np.asarray(preds, dtype=np.float32))
    print('saved predictions to', args.outfile, 'shape=', np.asarray(preds).shape)
