import os
import gc
import json
import numpy as np
import pandas as pd

EPS = 1e-9
FEATURE_PERIODS = (5, 10, 20, 30, 60, 120, 240)


def _safe_div(numerator, denominator):
    return numerator / (denominator + EPS)


def _moment_skew_kurt(values):
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return 0.0, 0.0
    centered = values - values.mean()
    variance = np.mean(centered * centered)
    if variance < EPS:
        return 0.0, 0.0
    skew = np.mean(centered ** 3) / ((variance ** 1.5) + EPS)
    kurt = np.mean(centered ** 4) / ((variance ** 2) + EPS) - 3.0
    return float(skew), float(kurt)


def _corr_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _ema_last(values, period):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    period = max(1, int(period))
    alpha = 2.0 / (period + 1.0)
    ema = float(values[0])
    for value in values[1:]:
        ema = alpha * float(value) + (1.0 - alpha) * ema
    return float(ema)


def _ema_series(values, period):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros((0,), dtype=float)
    period = max(1, int(period))
    alpha = 2.0 / (period + 1.0)
    ema = np.empty_like(values, dtype=float)
    ema[0] = values[0]
    for idx in range(1, values.size):
        ema[idx] = alpha * values[idx] + (1.0 - alpha) * ema[idx - 1]
    return ema


def _rsi_last(close, period):
    close = np.asarray(close, dtype=float)
    if close.size < 2:
        return 50.0
    period = max(2, min(int(period), close.size - 1))
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean() if gains.size else 0.0
    avg_loss = losses[-period:].mean() if losses.size else 0.0
    if avg_loss < EPS:
        return 100.0 if avg_gain > EPS else 50.0
    rs = avg_gain / (avg_loss + EPS)
    return float(100.0 - 100.0 / (1.0 + rs))


def _macd_last(close, fast=12, slow=26, signal=9):
    close = np.asarray(close, dtype=float)
    if close.size < 2:
        return 0.0, 0.0, 0.0
    fast_ema = _ema_series(close, min(fast, close.size))
    slow_ema = _ema_series(close, min(slow, close.size))
    macd_line = fast_ema - slow_ema
    signal_line = _ema_series(macd_line, min(signal, macd_line.size))
    macd_value = float(macd_line[-1])
    signal_value = float(signal_line[-1])
    hist_value = macd_value - signal_value
    return macd_value, signal_value, hist_value


def _kdj_last(high, low, close, period):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if close.size == 0:
        return 50.0, 50.0, 50.0
    period = max(2, min(int(period), close.size))
    k_value = 50.0
    d_value = 50.0
    for idx in range(close.size):
        start = max(0, idx - period + 1)
        highest_high = np.max(high[start:idx + 1])
        lowest_low = np.min(low[start:idx + 1])
        spread = highest_high - lowest_low
        rsv = 50.0 if spread < EPS else (close[idx] - lowest_low) / (spread + EPS) * 100.0
        k_value = (2.0 / 3.0) * k_value + (1.0 / 3.0) * rsv
        d_value = (2.0 / 3.0) * d_value + (1.0 / 3.0) * k_value
    j_value = 3.0 * k_value - 2.0 * d_value
    return float(k_value), float(d_value), float(j_value)


def _bollinger_last(close, period, num_std=2.0):
    close = np.asarray(close, dtype=float)
    if close.size == 0:
        return 0.0, 0.0, 0.0
    period = max(2, min(int(period), close.size))
    tail = close[-period:]
    mean = float(tail.mean())
    std = float(tail.std())
    upper = mean + num_std * std
    lower = mean - num_std * std
    width = _safe_div(upper - lower, abs(mean))
    pct_b = _safe_div(tail[-1] - lower, upper - lower)
    zscore = _safe_div(tail[-1] - mean, std)
    return float(width), float(pct_b), float(zscore)


def _atr_last(high, low, close, period):
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if close.size == 0:
        return 0.0, 0.0
    period = max(2, min(int(period), close.size))
    high_tail = high[-period:]
    low_tail = low[-period:]
    close_tail = close[-period:]
    prev_close = np.concatenate(([close_tail[0]], close_tail[:-1]))
    true_range = np.maximum.reduce([
        high_tail - low_tail,
        np.abs(high_tail - prev_close),
        np.abs(low_tail - prev_close),
    ])
    return float(true_range.mean()), float(true_range.std())


def _window_feature_block(open_hist, high_hist, low_hist, close_hist, vol_hist, period):
    period = int(period)
    usable = min(period, close_hist.size)
    if usable < 2:
        return np.zeros((0,), dtype=np.float32)

    open_tail = open_hist[-usable:]
    high_tail = high_hist[-usable:]
    low_tail = low_hist[-usable:]
    close_tail = close_hist[-usable:]
    vol_tail = vol_hist[-usable:]

    returns = np.diff(close_tail) / (close_tail[:-1] + EPS)
    price_last = float(close_tail[-1])
    price_first = float(close_tail[0])
    price_mean = float(close_tail.mean())
    price_std = float(close_tail.std())
    price_min = float(close_tail.min())
    price_max = float(close_tail.max())
    price_median = float(np.median(close_tail))
    price_range_pct = _safe_div(price_max - price_min, abs(price_min))
    price_z = _safe_div(price_last - price_mean, price_std)
    price_trend = _safe_div(price_last - price_first, usable - 1)
    price_skew, price_kurt = _moment_skew_kurt(close_tail)
    pos_return_ratio = float(np.mean(returns > 0.0)) if returns.size else 0.0
    realized_vol = float(returns.std()) if returns.size else 0.0
    downside = returns[returns < 0.0]
    upside = returns[returns > 0.0]
    downside_vol = float(downside.std()) if downside.size else 0.0
    upside_vol = float(upside.std()) if upside.size else 0.0
    autocorr1 = _corr_safe(returns[1:], returns[:-1]) if returns.size > 2 else 0.0

    vol_mean = float(vol_tail.mean())
    vol_std = float(vol_tail.std())
    vol_ratio_last = _safe_div(float(vol_tail[-1]), vol_mean)
    vol_z = _safe_div(float(vol_tail[-1]) - vol_mean, vol_std)
    vol_skew, vol_kurt = _moment_skew_kurt(vol_tail)
    vol_change = _safe_div(float(vol_tail[-1]) - float(vol_tail[0]), float(vol_tail[0]))
    vol_spike_ratio = _safe_div(float(vol_tail[-1]), np.percentile(vol_tail, 90))

    hl_range = high_tail - low_tail
    prev_close = np.concatenate(([close_tail[0]], close_tail[:-1]))
    true_range = np.maximum.reduce([
        hl_range,
        np.abs(high_tail - prev_close),
        np.abs(low_tail - prev_close),
    ])
    hl_mean = float(hl_range.mean())
    hl_std = float(hl_range.std())
    atr_mean = float(true_range.mean())
    atr_std = float(true_range.std())
    upper_wick = high_tail - np.maximum(open_tail, close_tail)
    lower_wick = np.minimum(open_tail, close_tail) - low_tail
    upper_wick_mean = float(upper_wick.mean())
    lower_wick_mean = float(lower_wick.mean())
    close_pos_range_last = _safe_div(float(close_tail[-1] - low_tail[-1]), float(high_tail[-1] - low_tail[-1]))
    body_ratio_last = _safe_div(abs(float(close_tail[-1] - open_tail[-1])), float(high_tail[-1] - low_tail[-1]))

    sma_ratio = _safe_div(price_last, price_mean) - 1.0
    ema_value = _ema_last(close_tail, min(period, close_tail.size))
    ema_ratio = _safe_div(price_last, ema_value) - 1.0
    rsi_value = _rsi_last(close_tail, period)
    macd_value, macd_signal, macd_hist = _macd_last(close_tail)
    k_value, d_value, j_value = _kdj_last(high_tail, low_tail, close_tail, period)
    boll_width, boll_pct_b, boll_z = _bollinger_last(close_tail, period)

    breakout_up = _safe_div(price_last, float(high_tail.max())) - 1.0
    breakout_down = _safe_div(float(low_tail.min()), price_last) - 1.0
    gap = _safe_div(float(open_tail[-1]), float(close_tail[-2])) - 1.0
    gap_abs = abs(gap)
    abs_ret_q90_ratio = _safe_div(abs(float(returns[-1])), np.percentile(np.abs(returns), 90)) if returns.size else 0.0
    vol_q90_ratio = _safe_div(float(vol_tail[-1]), np.percentile(vol_tail, 90))
    price_vol_corr = _corr_safe(returns, vol_tail[1:]) if vol_tail.size > 1 else 0.0
    tail_ratio = float(np.mean(np.abs(returns) > (2.0 * realized_vol))) if returns.size else 0.0
    vwap = _safe_div(float(np.sum(close_tail * vol_tail)), float(np.sum(vol_tail)))
    vwap_ratio = _safe_div(price_last, vwap) - 1.0
    vwap_z = _safe_div(price_last - vwap, price_std)

    return [
        price_last / (price_first + EPS) - 1.0,
        np.log((price_last + EPS) / (price_first + EPS)),
        price_mean,
        price_std,
        price_min,
        price_max,
        price_range_pct,
        price_median,
        price_z,
        price_trend,
        price_skew,
        price_kurt,
        pos_return_ratio,
        realized_vol,
        downside_vol,
        upside_vol,
        autocorr1,
        vol_mean,
        vol_std,
        vol_ratio_last,
        vol_z,
        vol_skew,
        vol_kurt,
        vol_change,
        vol_spike_ratio,
        hl_mean,
        hl_std,
        atr_mean,
        atr_std,
        upper_wick_mean,
        lower_wick_mean,
        close_pos_range_last,
        body_ratio_last,
        sma_ratio,
        ema_ratio,
        rsi_value,
        macd_value,
        macd_signal,
        macd_hist,
        k_value,
        d_value,
        j_value,
        boll_width,
        boll_pct_b,
        boll_z,
        breakout_up,
        breakout_down,
        gap,
        gap_abs,
        abs_ret_q90_ratio,
        vol_q90_ratio,
        price_vol_corr,
        tail_ratio,
        vwap_ratio,
        vwap_z,
    ]

def intervals_to_bool(intervals, length):
    """把区间数组转换为逐点布尔数组。
    intervals: array-like of shape (M,2) or (...,)
    length: 目标长度
    返回 dtype=bool 的数组
    """
    mask = np.zeros(length, dtype=bool)
    if intervals is None:
        return mask
    arr = np.asarray(intervals)
    if arr.size == 0:
        return mask
    if arr.ndim == 1 and arr.shape[0] == 2:
        starts = [int(arr[0])]
        ends = [int(arr[1])]
    else:
        starts = arr[:, 0].astype(int)
        ends = arr[:, 1].astype(int)
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        s_clamped = max(0, s)
        e_clamped = min(length, e)
        mask[s_clamped:e_clamped] = True
    return mask

def build_features_from_ohlcv(ohlcv, window=60, progress_label=None, progress_every=None):
    """基于 OHLCV （N x 5） 矩阵生成滚动特征。返回 (N-window) x D 特征矩阵。
    支持缺失或短序列的鲁棒处理。
    自动检测 8 列 / 7 列 / 5 列格式。
    """
    ohlcv = np.nan_to_num(np.asarray(ohlcv, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if ohlcv.ndim != 2 or ohlcv.shape[1] < 5:
        raise ValueError("ohlcv must be 2D array with at least 5 columns (O H L C V)")

    open_arr, high_arr, low_arr, close_arr, vol_arr, _, _, _ = _auto_slice_ohlcv(ohlcv)
    open_s = pd.Series(open_arr)
    high_s = pd.Series(high_arr)
    low_s = pd.Series(low_arr)
    close_s = pd.Series(close_arr)
    vol_s = pd.Series(vol_arr)

    N = len(close_s)
    if N < window:
        return np.zeros((0, 0)), np.arange(N)

    feature_blocks = []
    start = window - 1
    total_steps = N - window + 1
    if progress_every is None:
        progress_every = max(1, total_steps // 10)
    if progress_label:
        print(f'  -> {progress_label}: building rolling features ({total_steps} steps)')
    close_np = close_s.to_numpy(dtype=float)
    open_np = open_s.to_numpy(dtype=float)

    ret1 = np.zeros(N, dtype=float)
    if N > 1:
        ret1[1:] = (close_np[1:] - close_np[:-1]) / (np.abs(close_np[:-1]) + EPS)
    ret1_s = pd.Series(ret1)

    def build_period_block(period):
        period = max(2, min(int(period), N))
        ret_window = max(2, period - 1)

        price_roll = close_s.rolling(period, min_periods=period)
        price_mean = price_roll.mean()
        price_std = price_roll.std(ddof=0)
        price_min = price_roll.min()
        price_max = price_roll.max()
        price_median = price_roll.median()
        price_skew = price_roll.skew().fillna(0.0)
        price_kurt = price_roll.kurt().fillna(0.0)
        price_first = close_s.shift(period - 1)
        price_last = close_s
        price_range_pct = _safe_div(price_max - price_min, price_min.abs())
        price_z = _safe_div(price_last - price_mean, price_std.replace(0.0, np.nan))
        price_trend = _safe_div(price_last - price_first, period - 1)

        returns_roll = ret1_s.rolling(ret_window, min_periods=ret_window)
        pos_return_ratio = (ret1_s > 0.0).rolling(ret_window, min_periods=ret_window).mean().fillna(0.0)
        realized_vol = returns_roll.std(ddof=0).fillna(0.0)
        downside_vol = ret1_s.where(ret1_s < 0.0).rolling(ret_window, min_periods=1).std(ddof=0).fillna(0.0)
        upside_vol = ret1_s.where(ret1_s > 0.0).rolling(ret_window, min_periods=1).std(ddof=0).fillna(0.0)
        autocorr1 = ret1_s.rolling(ret_window, min_periods=2).corr(ret1_s.shift(1)).fillna(0.0)
        abs_ret_q90 = ret1_s.abs().rolling(ret_window, min_periods=ret_window).quantile(0.9)
        abs_ret_q90_ratio = _safe_div(ret1_s.abs(), abs_ret_q90)

        vol_roll = vol_s.rolling(period, min_periods=period)
        vol_mean = vol_roll.mean()
        vol_std = vol_roll.std(ddof=0)
        vol_ratio_last = _safe_div(vol_s, vol_mean)
        vol_z = _safe_div(vol_s - vol_mean, vol_std.replace(0.0, np.nan))
        vol_skew = vol_roll.skew().fillna(0.0)
        vol_kurt = vol_roll.kurt().fillna(0.0)
        vol_change = _safe_div(vol_s - vol_s.shift(period - 1), vol_s.shift(period - 1).abs())
        vol_q90 = vol_roll.quantile(0.9)
        vol_spike_ratio = _safe_div(vol_s, vol_q90)
        vol_q90_ratio = vol_spike_ratio

        hl_range = high_s - low_s
        prev_close = close_s.shift(1).fillna(close_s.iloc[0])
        true_range = pd.concat([
            hl_range,
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ], axis=1).max(axis=1)
        hl_mean = hl_range.rolling(period, min_periods=period).mean()
        hl_std = hl_range.rolling(period, min_periods=period).std(ddof=0).fillna(0.0)
        atr_mean = true_range.rolling(period, min_periods=period).mean()
        atr_std = true_range.rolling(period, min_periods=period).std(ddof=0).fillna(0.0)
        upper_wick = high_s - np.maximum(open_np, close_np)
        lower_wick = np.minimum(open_np, close_np) - low_s
        upper_wick_mean = pd.Series(upper_wick).rolling(period, min_periods=period).mean()
        lower_wick_mean = pd.Series(lower_wick).rolling(period, min_periods=period).mean()
        close_pos_range_last = _safe_div(close_s - low_s, (high_s - low_s).replace(0.0, np.nan))
        body_ratio_last = _safe_div((close_s - open_s).abs(), (high_s - low_s).replace(0.0, np.nan))

        sma_ratio = _safe_div(price_last, price_mean) - 1.0
        ema_value = close_s.ewm(span=period, adjust=False).mean()
        ema_ratio = _safe_div(price_last, ema_value) - 1.0

        delta = close_s.diff().fillna(0.0)
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        avg_gain = gains.rolling(ret_window, min_periods=ret_window).mean()
        avg_loss = losses.rolling(ret_window, min_periods=ret_window).mean()
        rs = _safe_div(avg_gain, avg_loss)
        rsi_value = 100.0 - 100.0 / (1.0 + rs)
        rsi_value = rsi_value.where(avg_loss >= EPS, np.where(avg_gain > EPS, 100.0, 50.0))

        fast_ema = close_s.ewm(span=12, adjust=False).mean()
        slow_ema = close_s.ewm(span=26, adjust=False).mean()
        macd_value = fast_ema - slow_ema
        macd_signal = macd_value.ewm(span=9, adjust=False).mean()
        macd_hist = macd_value - macd_signal

        highest_high = high_s.rolling(period, min_periods=period).max()
        lowest_low = low_s.rolling(period, min_periods=period).min()
        spread = highest_high - lowest_low
        rsv = _safe_div(close_s - lowest_low, spread) * 100.0
        rsv = rsv.where(spread >= EPS, 50.0)
        k_value = rsv.ewm(alpha=1.0 / 3.0, adjust=False).mean()
        d_value = k_value.ewm(alpha=1.0 / 3.0, adjust=False).mean()
        j_value = 3.0 * k_value - 2.0 * d_value

        boll_mid = price_mean
        boll_std = price_std.replace(0.0, np.nan)
        boll_upper = boll_mid + 2.0 * boll_std
        boll_lower = boll_mid - 2.0 * boll_std
        boll_width = _safe_div(boll_upper - boll_lower, boll_mid.abs())
        boll_pct_b = _safe_div(close_s - boll_lower, boll_upper - boll_lower)
        boll_z = _safe_div(close_s - boll_mid, boll_std)

        breakout_up = _safe_div(close_s, highest_high) - 1.0
        breakout_down = _safe_div(lowest_low, close_s) - 1.0
        gap = _safe_div(open_s, prev_close) - 1.0
        gap_abs = gap.abs()
        price_vol_corr = ret1_s.rolling(ret_window, min_periods=2).corr(vol_s).fillna(0.0)
        tail_indicator = (ret1_s.abs() > (2.0 * realized_vol)).astype(float)
        tail_ratio = tail_indicator.rolling(ret_window, min_periods=ret_window).mean().fillna(0.0)
        vwap = _safe_div((close_s * vol_s).rolling(period, min_periods=period).sum(), vol_s.rolling(period, min_periods=period).sum())
        vwap_ratio = _safe_div(price_last, vwap) - 1.0
        vwap_z = _safe_div(price_last - vwap, price_std.replace(0.0, np.nan))

        # --- Enhanced features (all vectorized, no .apply()) ---
        # Parkinson volatility: sigma = sqrt(mean(ln(H/L)^2) / (4*ln(2)))
        parkinson_sq = np.log(_safe_div(high_s, low_s)) ** 2
        parkinson_vol = np.sqrt(parkinson_sq.rolling(period, min_periods=period).mean().fillna(0.0) / (4.0 * np.log(2.0)))
        parkinson_long_mean = parkinson_vol.replace(0.0, np.nan).rolling(int(period * 2), min_periods=period).mean().fillna(0.0)
        parkinson_vol_ratio = _safe_div(parkinson_vol, parkinson_long_mean)

        # Return acceleration (2nd derivative)
        ret_accel = ret1_s.diff().rolling(ret_window, min_periods=ret_window).mean().fillna(0.0)

        # Upside/downside volatility asymmetry
        vol_asymmetry = _safe_div(upside_vol - downside_vol, upside_vol + downside_vol)

        # Consecutive up/down ratio
        up_streak = (ret1_s > 0.0).astype(float)
        dn_streak = (ret1_s < 0.0).astype(float)
        up_streak_sum = up_streak.rolling(ret_window, min_periods=1).sum().fillna(0.0)
        dn_streak_sum = dn_streak.rolling(ret_window, min_periods=1).sum().fillna(0.0)
        up_dn_ratio = _safe_div(up_streak_sum, up_streak_sum + dn_streak_sum)

        # Volume momentum
        vol_mom = _safe_div(vol_s - vol_s.shift(int(period // 2)),
                            vol_s.shift(int(period // 2)).abs()).fillna(0.0)

        # Price drawdown from period high and run-up from period low
        drawdown = _safe_div(close_s - highest_high, highest_high)
        runup = _safe_div(close_s - lowest_low, lowest_low)

        # Garman-Klass volatility (fully vectorized)
        gk_term1 = 0.5 * (np.log(_safe_div(high_s, low_s))) ** 2
        gk_term2 = (2.0 * np.log(2.0) - 1.0) * (np.log(_safe_div(close_s, open_s))) ** 2
        gk_comp = (gk_term1 - gk_term2).rolling(period, min_periods=period).mean().fillna(0.0)
        gk_vol = np.sqrt(np.maximum(gk_comp, 0.0))

        # Return density: fraction of returns exceeding 1 std
        ret_density = (ret1_s.abs() > realized_vol.replace(0.0, np.nan)).astype(float) \
            .rolling(ret_window, min_periods=ret_window).mean().fillna(0.0)

        # Short-term vs long-term volatility ratio
        st_vol = ret1_s.abs().rolling(max(2, period // 4), min_periods=2).mean().fillna(0.0)
        lt_vol = ret1_s.abs().rolling(period, min_periods=period).mean().fillna(0.0)
        vol_term_ratio = _safe_div(st_vol, lt_vol)

        # Volume-price trend divergence (endpoint slope, fully vectorized)
        half_period = max(2, period // 2)
        vol_trend_slope = _safe_div(vol_s - vol_s.shift(period - 1), vol_s.shift(period - 1).abs().rolling(period, min_periods=period).mean().fillna(EPS))
        price_trend_slope = _safe_div(close_s - close_s.shift(period - 1), close_s.shift(period - 1).abs().rolling(period, min_periods=period).mean().fillna(EPS))
        vol_price_div = (vol_trend_slope - price_trend_slope).fillna(0.0)

        # Convexity: E[r^2] / E[|r|]^2 proxy
        ret_sq = (ret1_s * ret1_s).rolling(ret_window, min_periods=ret_window).mean().fillna(0.0)
        convexity = _safe_div(ret_sq, realized_vol.replace(0.0, np.nan) ** 2 + EPS)

        # Return skewness over short window
        ret_skew_short = ret1_s.rolling(max(2, period // 2), min_periods=2).skew().fillna(0.0)

        # Close position within recent short range
        range_recent_hi = high_s.rolling(max(2, period // 2), min_periods=2).max()
        range_recent_lo = low_s.rolling(max(2, period // 2), min_periods=2).min()
        close_pos_short = _safe_div(close_s - range_recent_lo, range_recent_hi - range_recent_lo)

        block = np.column_stack([
            (price_last / (price_first + EPS) - 1.0).to_numpy(),
            np.log((price_last + EPS) / (price_first + EPS)).to_numpy(),
            price_mean.to_numpy(),
            price_std.to_numpy(),
            price_min.to_numpy(),
            price_max.to_numpy(),
            price_range_pct.to_numpy(),
            price_median.to_numpy(),
            price_z.to_numpy(),
            price_trend.to_numpy(),
            price_skew.to_numpy(),
            price_kurt.to_numpy(),
            pos_return_ratio.to_numpy(),
            realized_vol.to_numpy(),
            downside_vol.to_numpy(),
            upside_vol.to_numpy(),
            autocorr1.to_numpy(),
            vol_mean.to_numpy(),
            vol_std.to_numpy(),
            vol_ratio_last.to_numpy(),
            vol_z.to_numpy(),
            vol_skew.to_numpy(),
            vol_kurt.to_numpy(),
            vol_change.to_numpy(),
            vol_spike_ratio.to_numpy(),
            hl_mean.to_numpy(),
            hl_std.to_numpy(),
            atr_mean.to_numpy(),
            atr_std.to_numpy(),
            upper_wick_mean.to_numpy(),
            lower_wick_mean.to_numpy(),
            close_pos_range_last.to_numpy(),
            body_ratio_last.to_numpy(),
            sma_ratio.to_numpy(),
            ema_ratio.to_numpy(),
            rsi_value.to_numpy(),
            macd_value.to_numpy(),
            macd_signal.to_numpy(),
            macd_hist.to_numpy(),
            k_value.to_numpy(),
            d_value.to_numpy(),
            j_value.to_numpy(),
            boll_width.to_numpy(),
            boll_pct_b.to_numpy(),
            boll_z.to_numpy(),
            breakout_up.to_numpy(),
            breakout_down.to_numpy(),
            gap.to_numpy(),
            gap_abs.to_numpy(),
            abs_ret_q90_ratio.to_numpy(),
            vol_q90_ratio.to_numpy(),
            price_vol_corr.to_numpy(),
            tail_ratio.to_numpy(),
            vwap_ratio.to_numpy(),
            vwap_z.to_numpy(),
            parkinson_vol.to_numpy(),
            parkinson_vol_ratio.to_numpy(),
            ret_accel.to_numpy(),
            vol_asymmetry.to_numpy(),
            up_dn_ratio.to_numpy(),
            vol_mom.to_numpy(),
            drawdown.to_numpy(),
            runup.to_numpy(),
            gk_vol.to_numpy(),
            ret_density.to_numpy(),
            vol_term_ratio.to_numpy(),
            vol_price_div.to_numpy(),
            convexity.to_numpy(),
            ret_skew_short.to_numpy(),
            close_pos_short.to_numpy(),
        ]).astype(np.float32, copy=False)
        return np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    for period in FEATURE_PERIODS:
        if progress_label:
            print(f'     -> computing vectorized block for period={period}')
        block = build_period_block(period)
        feature_blocks.append(block[start:].astype(np.float32, copy=False))

    X = np.hstack(feature_blocks).astype(np.float32, copy=False) if feature_blocks else np.zeros((N - window + 1, 0), dtype=np.float32)
    idxs = np.arange(window - 1, N, dtype=int)
    return X, idxs

def _auto_slice_ohlcv(ohlcv):
    """Detect column layout and return (O, H, L, C, V, ret5, ret60, has_labels).

    Training data: 8 cols [index, O, H, L, C, V, ret5, ret60]
    Platform input: 5 cols [O, H, L, C, V]
    Also handles 7-col [O, H, L, C, V, ret5, ret60]
    """
    ohlcv = np.asarray(ohlcv, dtype=float)
    ncols = ohlcv.shape[1]

    # Quick heuristic: if first col is sequential integers starting from 0, it's an index col
    if ncols >= 8:
        first_col = ohlcv[:min(5, len(ohlcv)), 0]
        expected = np.arange(len(first_col), dtype=float)
        if np.allclose(first_col, expected):
            return (ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4], ohlcv[:, 5],
                    ohlcv[:, 6], ohlcv[:, 7], True)

    if ncols >= 7:
        return (ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4],
                ohlcv[:, 5], ohlcv[:, 6], True)

    return (ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4],
            None, None, False)


def extract_targets_from_ohlcv(ohlcv):
    ohlcv = np.asarray(ohlcv)
    if ohlcv.ndim != 2 or ohlcv.shape[1] < 4:
        raise ValueError('ohlcv must be 2D array with at least 4 columns')

    _, _, _, close, _, ret5, ret60, has_labels = _auto_slice_ohlcv(ohlcv)
    if has_labels:
        return np.asarray(ret5, dtype=float), np.asarray(ret60, dtype=float)

    close = np.asarray(close, dtype=float)
    ret5 = np.full(len(close), np.nan, dtype=float)
    ret60 = np.full(len(close), np.nan, dtype=float)
    for i in range(len(close)):
        if i + 5 < len(close):
            ret5[i] = (close[i + 5] - close[i]) / (close[i] + 1e-9)
        if i + 60 < len(close):
            ret60[i] = (close[i + 60] - close[i]) / (close[i] + 1e-9)
    return np.asarray(ret5, dtype=float), np.asarray(ret60, dtype=float)

def fit_preprocessing_params(X):
    X = np.asarray(X, dtype=float)
    feature_count = X.shape[1] if X.ndim == 2 else 0
    return {
        'method': 'rolling_zscore',
        'window': 240,
        'min_history': 1,
        'eps': 1e-9,
        'feature_count': int(feature_count),
    }

def _apply_diff_preprocessing(X, preprocessing_params):
    """Apply causal first-order difference standardization to preserve ret60 trend information."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.size == 0:
        return X
    n_samples, n_features = X.shape
    if n_samples == 0:
        return X

    window = int(preprocessing_params.get('window', 240)) if preprocessing_params else 240
    min_history = int(preprocessing_params.get('min_history', 1)) if preprocessing_params else 1
    eps = float(preprocessing_params.get('eps', 1e-9)) if preprocessing_params else 1e-9

    X_out = np.zeros((n_samples, n_features), dtype=np.float32)
    rolling_sum = np.zeros((n_features,), dtype=np.float32)
    rolling_sumsq = np.zeros((n_features,), dtype=np.float32)
    history_buffer = np.zeros((window, n_features), dtype=np.float32)

    prev_row = X[0]
    for idx in range(n_samples):
        current_row = X[idx]
        if idx == 0:
            diff_row = np.zeros((n_features,), dtype=np.float32)
        else:
            diff_row = (current_row - prev_row).astype(np.float32, copy=False)

        history_count = min(idx, window)
        if history_count < min_history:
            buffer_slot = idx % window
            if idx >= window:
                outgoing = history_buffer[buffer_slot]
                rolling_sum -= outgoing
                rolling_sumsq -= outgoing * outgoing
            history_buffer[buffer_slot] = diff_row
            rolling_sum += diff_row
            rolling_sumsq += diff_row * diff_row
            prev_row = current_row
            continue

        mean = rolling_sum / history_count
        variance = rolling_sumsq / history_count - mean * mean
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std < eps, 1.0, std)
        X_out[idx] = ((diff_row - mean) / std).astype(np.float32, copy=False)

        buffer_slot = idx % window
        if idx >= window:
            outgoing = history_buffer[buffer_slot]
            rolling_sum -= outgoing
            rolling_sumsq -= outgoing * outgoing
        history_buffer[buffer_slot] = diff_row
        rolling_sum += diff_row
        rolling_sumsq += diff_row * diff_row
        prev_row = current_row

    return X_out

def apply_preprocessing(X, preprocessing_params, target_name=None):
    X = np.asarray(X, dtype=np.float32)
    if preprocessing_params is None:
        return X
    if X.size == 0:
        return X
    
    method = preprocessing_params.get('method', 'rolling_zscore')
    if method != 'rolling_zscore':
        return X
    window = int(preprocessing_params.get('window', 240))
    min_history = int(preprocessing_params.get('min_history', 1))
    eps = float(preprocessing_params.get('eps', 1e-9))
    if X.ndim != 2:
        return X
    n_samples, n_features = X.shape
    X_out = np.zeros((n_samples, n_features), dtype=np.float32)
    if n_samples == 0:
        return X_out
    # Upstream feature generation already sanitizes NaN/Inf, so avoid nan_to_num here
    # to prevent creating another full-size temporary bool mask array.
    clean_X = X.astype(np.float32, copy=False)
    # 使用流式滚动窗口，只保留最近 window 行的统计量，避免为全量样本分配前缀和大数组。
    # 第 idx 行只用 [max(0, idx-window), idx) 的历史数据做标准化，不包含当前行。
    rolling_sum = np.zeros((n_features,), dtype=np.float32)
    rolling_sumsq = np.zeros((n_features,), dtype=np.float32)
    history_buffer = np.zeros((window, n_features), dtype=np.float32)

    for idx in range(n_samples):
        history_count = min(idx, window)
        if history_count < min_history:
            row = clean_X[idx]
            buffer_slot = idx % window
            if idx >= window:
                outgoing = history_buffer[buffer_slot]
                rolling_sum -= outgoing
                rolling_sumsq -= outgoing * outgoing
            history_buffer[buffer_slot] = row
            rolling_sum += row
            rolling_sumsq += row * row
            continue
        mean = rolling_sum / history_count
        variance = rolling_sumsq / history_count - mean * mean
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std < eps, 1.0, std)
        X_out[idx] = ((clean_X[idx] - mean) / std).astype(np.float32, copy=False)

        row = clean_X[idx]
        buffer_slot = idx % window
        if idx >= window:
            outgoing = history_buffer[buffer_slot]
            rolling_sum -= outgoing
            rolling_sumsq -= outgoing * outgoing
        history_buffer[buffer_slot] = row
        rolling_sum += row
        rolling_sumsq += row * row
    return X_out

def generate_dataset_shards_from_folder(folder, shard_dir='train_shards', window=60, label_type='Ret5'):
    """按标的生成训练分片，避免全量拼接导致内存峰值过高。"""
    files = os.listdir(folder)
    ohlcv_files = [f for f in files if f.endswith('_train_ohlcv.npy')]
    # Natural sort by numeric part to get dataset0, dataset1, dataset2, ... not 0, 10, 11...
    ohlcv_files = sorted(ohlcv_files, key=lambda f: int(''.join(ch for ch in f if ch.isdigit()) or 0))
    print(f'Found {len(ohlcv_files)} ohlcv files in {folder}')
    os.makedirs(shard_dir, exist_ok=True)
    preprocessing_params = {
        'method': 'rolling_zscore',
        'window': 240,
        'min_history': 1,
        'eps': 1e-9,
        'feature_count': 0,
    }
    manifest = {
        'folder': os.path.abspath(folder),
        'shard_dir': os.path.abspath(shard_dir),
        'window': int(window),
        'label_type': label_type,
        'preprocessing_params': preprocessing_params,
        'feature_count': 0,
        'total_samples': 0,
        'shards': [],
    }

    for i, of in enumerate(ohlcv_files, 1):
        print(f'[{i}/{len(ohlcv_files)}] Processing {of} ...')
        base = of.replace('_train_ohlcv.npy', '')
        group_token = ''.join(ch for ch in base if ch.isdigit())
        group_id = int(group_token) if group_token else i - 1
        ohlcv = np.load(os.path.join(folder, of))
        # 对应的极端区间文件
        ext_file = base + '_train_extreme_intervals.npy'
        ext_path = os.path.join(folder, ext_file)
        intervals = None
        if os.path.exists(ext_path):
            intervals = np.load(ext_path, allow_pickle=True)
        N = len(ohlcv)
        mask_ext = intervals_to_bool(intervals, N)
        print(f'  -> building features for {of}')
        X, idxs = build_features_from_ohlcv(ohlcv, window=window, progress_label=of)
        print(f'  -> generated features shape: {X.shape}')
        if X.shape[0] == 0:
            print('  -> skipped (no samples)')
            continue
        ret5_all, ret60_all = extract_targets_from_ohlcv(ohlcv)
        y = np.column_stack([ret5_all[idxs], ret60_all[idxs]])
        # 丢弃末尾含 NaN 的样本（因标签对齐问题）
        valid = ~np.isnan(y).any(axis=1)
        X = X[valid].astype(np.float32, copy=False)
        y = y[valid]
        idxs_valid = idxs[valid]
        # 样本权重：极端样本加权 3 倍
        weights = np.where(mask_ext[idxs_valid], 3.0, 1.0).astype(np.float32, copy=False)
        if preprocessing_params['feature_count'] == 0:
            preprocessing_params['feature_count'] = int(X.shape[1])
            manifest['feature_count'] = int(X.shape[1])
        print(f'  -> saving raw features for {of}; preprocessing will be applied during training/prediction')

        shard_path = os.path.join(shard_dir, f'{base}_shard.npz')
        shard_group_ids = np.full(X.shape[0], group_id, dtype=np.int32)
        shard_time_indices = idxs_valid.astype(np.int32, copy=False)
        shard_is_extreme = mask_ext[idxs_valid].astype(bool, copy=False)
        np.savez_compressed(
            shard_path,
            X=X.astype(np.float32, copy=False),
            y=y.astype(np.float32, copy=False),
            w=weights,
            group_ids=shard_group_ids,
            time_indices=shard_time_indices,
            is_extreme=shard_is_extreme,
        )
        manifest['shards'].append({
            'path': os.path.abspath(shard_path),
            'dataset': base,
            'group_id': int(group_id),
            'n_samples': int(X.shape[0]),
        })
        manifest['total_samples'] += int(X.shape[0])

        print(f'  -> kept {X.shape[0]} samples, weights mean {weights.mean():.2f}')
        print(f'  -> saved shard: {shard_path}')
        del ohlcv, intervals, X, idxs, ret5_all, ret60_all, y, idxs_valid, weights, mask_ext
        gc.collect()

    manifest_path = os.path.join(shard_dir, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'Shard manifest saved to {manifest_path}')
    return manifest


def generate_dataset_from_folder(folder, window=60, label_type='Ret5'):
    """兼容旧调用：默认改为分片落盘并返回 manifest。"""
    default_shard_dir = os.path.join(folder, 'train_shards')
    return generate_dataset_shards_from_folder(folder, shard_dir=default_shard_dir, window=window, label_type=label_type)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', default='train_dataset')
    parser.add_argument('--shard-dir', default='train_shards')
    parser.add_argument('--window', type=int, default=60)
    args = parser.parse_args()
    manifest = generate_dataset_shards_from_folder(args.data_folder, shard_dir=args.shard_dir, window=args.window)
    print(f"Generated {len(manifest['shards'])} shards, total samples: {manifest['total_samples']}")
