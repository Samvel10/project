import json
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import BayesianRidge
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None  # type: ignore
    RandomForestRegressor = None  # type: ignore
    ExtraTreesRegressor = None  # type: ignore
    MultiOutputRegressor = None  # type: ignore
    MLPRegressor = None  # type: ignore
    BayesianRidge = None  # type: ignore


_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _ROOT / "data" / "sltp_ensemble.pkl"
_DECISIONS_PATH = _ROOT / "data" / "sl_tp_decision_snapshots.jsonl"
_ANALYSIS_PATH = _ROOT / "data" / "sl_tp_post_analysis.jsonl"

_MIN_SL = 0.3
_MAX_SL = 7.0
_MIN_TP = 0.5
_MAX_TP = 25.0

_MODEL_CACHE = None
_MODEL_CACHE_MTIME: Optional[float] = None


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _build_feature_vector(
    symbol: str,
    side: str,
    entry_price: float,
    base_sl_pct: float,
    base_tp_pcts: List[float],
    features: Dict[str, float],
) -> Tuple[List[float], Dict[str, float]]:
    ep = max(_safe_float(entry_price), 1e-12)
    side_u = str(side).upper()
    side_sign = 1.0 if side_u in ("BUY", "LONG") else -1.0

    last_price = _safe_float(features.get("last_price"), ep)
    candle_range_pct = _safe_float(features.get("candle_range_pct"))
    trend_strength = _safe_float(features.get("trend_strength"))
    structure_level = _safe_float(features.get("structure_level"))

    atr = _safe_float(features.get("atr"))
    atr_pct = abs(atr / ep) * 100.0 if atr > 0 else 0.0
    vol_1m = _safe_float(features.get("volatility_1m"))
    vol_5m = _safe_float(features.get("volatility_5m"))
    vol_1h = _safe_float(features.get("volatility_1h"))
    range_exp = _safe_float(features.get("range_expansion"))

    rsi = _safe_float(features.get("rsi"), 50.0)
    momentum = _safe_float(features.get("momentum"))
    accel = _safe_float(features.get("acceleration"))

    confidence = _safe_float(features.get("confidence"))
    signal_strength = _safe_float(features.get("signal_strength"), confidence)
    volume = _safe_float(features.get("volume"))

    symbol_win_rate = _safe_float(features.get("symbol_win_rate"), 0.5)
    symbol_avg_pnl = _safe_float(features.get("symbol_avg_pnl"), 0.0)
    symbol_avg_drawdown = _safe_float(features.get("symbol_avg_drawdown"), 0.0)

    # Activity score for ACTIVE/PASSIVE gate.
    score = (
        0.18 * _clamp(vol_1m / 2.0, 0.0, 1.0)
        + 0.20 * _clamp(vol_5m / 3.0, 0.0, 1.0)
        + 0.15 * _clamp(vol_1h / 6.0, 0.0, 1.0)
        + 0.15 * _clamp(atr_pct / 4.0, 0.0, 1.0)
        + 0.10 * _clamp(candle_range_pct / 3.0, 0.0, 1.0)
        + 0.10 * _clamp(abs(momentum) / 3.0, 0.0, 1.0)
        + 0.06 * _clamp(abs(accel) / 2.0, 0.0, 1.0)
        + 0.06 * _clamp(range_exp / 2.0, 0.0, 1.0)
    )
    market_state = "ACTIVE" if score >= 0.45 else "PASSIVE"
    trade_allowed = market_state == "ACTIVE"

    base_tp1 = _safe_float(base_tp_pcts[0] if len(base_tp_pcts) > 0 else 1.0, 1.0)
    base_tp2 = _safe_float(base_tp_pcts[1] if len(base_tp_pcts) > 1 else base_tp1 * 1.5, base_tp1 * 1.5)
    base_tp3 = _safe_float(base_tp_pcts[2] if len(base_tp_pcts) > 2 else base_tp2 * 1.3, base_tp2 * 1.3)

    # Temporal features (simple recent-outcome proxies) to satisfy temporal expert input.
    recent_pnl_ema = _safe_float(features.get("recent_pnl_ema"), symbol_avg_pnl)
    recent_hit_rate = _safe_float(features.get("recent_hit_rate"), symbol_win_rate)

    x = [
        last_price,
        ep,
        side_sign,
        base_sl_pct,
        base_tp1,
        base_tp2,
        base_tp3,
        atr,
        atr_pct,
        vol_1m,
        vol_5m,
        vol_1h,
        candle_range_pct,
        range_exp,
        volume,
        rsi,
        momentum,
        accel,
        trend_strength,
        structure_level,
        confidence,
        signal_strength,
        symbol_win_rate,
        symbol_avg_pnl,
        symbol_avg_drawdown,
        recent_pnl_ema,
        recent_hit_rate,
        score,
    ]

    meta = {
        "activity_score": float(score),
        "market_state": market_state,
        "trade_allowed": bool(trade_allowed),
        "atr_pct": float(atr_pct),
        "vol_1m": float(vol_1m),
        "vol_5m": float(vol_5m),
        "vol_1h": float(vol_1h),
        "confidence": float(confidence),
    }
    return x, meta


def _load_ensemble():
    global _MODEL_CACHE, _MODEL_CACHE_MTIME
    try:
        if not _MODEL_PATH.exists():
            _MODEL_CACHE = None
            _MODEL_CACHE_MTIME = None
            return None
        mtime = float(_MODEL_PATH.stat().st_mtime)
        if _MODEL_CACHE is not None and _MODEL_CACHE_MTIME == mtime:
            return _MODEL_CACHE
        with _MODEL_PATH.open("rb") as f:
            obj = pickle.load(f)
        _MODEL_CACHE = obj
        _MODEL_CACHE_MTIME = mtime
        return obj
    except Exception:
        _MODEL_CACHE = None
        _MODEL_CACHE_MTIME = None
        return None


def decide_sltp(
    symbol: str,
    side: str,
    entry_price: float,
    base_sl_pct: float,
    base_tp_pcts: List[float],
    features: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    feats = dict(features or {})
    x, meta = _build_feature_vector(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        base_sl_pct=base_sl_pct,
        base_tp_pcts=base_tp_pcts,
        features=feats,
    )

    # PASSIVE market: do not allow new entries.
    if not bool(meta["trade_allowed"]):
        return {
            "trade_allowed": False,
            "market_state": meta["market_state"],
            "sl_pct": _clamp(float(base_sl_pct), _MIN_SL, _MAX_SL),
            "tp_pcts": [_clamp(float(v), _MIN_TP, _MAX_TP) for v in list(base_tp_pcts)[:3]],
            "activity_score": float(meta["activity_score"]),
        }

    # Heuristic baseline for dynamic SL/TP.
    atr_pct = float(meta.get("atr_pct", 0.0))
    conf = float(meta.get("confidence", 0.0))
    vol_mix = 0.5 * float(meta.get("vol_5m", 0.0)) + 0.5 * float(meta.get("vol_1h", 0.0))
    trend_strength = _safe_float(feats.get("trend_strength"), 0.0)

    conf_scale = 0.85 if conf < 0.6 else (1.0 if conf < 0.75 else 1.15)
    vol_scale = 0.9 if vol_mix < 1.0 else (1.0 if vol_mix < 2.5 else 1.15)
    trend_scale = 0.95 if trend_strength < 0.4 else (1.0 if trend_strength < 0.8 else 1.08)
    sl_h = _clamp(float(base_sl_pct) * conf_scale * (2.0 - vol_scale), _MIN_SL, _MAX_SL)
    tp_base = [float(v) for v in (list(base_tp_pcts)[:3] or [1.0, 1.5, 2.0])]
    while len(tp_base) < 3:
        tp_base.append(tp_base[-1] * 1.25)
    tp_h = [
        _clamp(tp_base[0] * vol_scale * trend_scale, _MIN_TP, _MAX_TP),
        _clamp(tp_base[1] * vol_scale * trend_scale, _MIN_TP, _MAX_TP),
        _clamp(tp_base[2] * vol_scale * trend_scale, _MIN_TP, _MAX_TP),
    ]

    ens = _load_ensemble()
    sl_pred = float(sl_h)
    tp_pred = list(tp_h)
    if ens and np is not None:
        try:
            arr = np.array([x], dtype=float)
            preds = []
            weights = []
            models = ens.get("models", {})
            model_weights = ens.get("weights", {})
            for name, model in models.items():
                try:
                    y = model.predict(arr)
                    if isinstance(y, list):
                        row = y[0]
                    else:
                        row = y[0]
                    if len(row) < 4:
                        continue
                    preds.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                    w = float(model_weights.get(name, 1.0))
                    weights.append(max(w, 1e-6))
                except Exception:
                    continue
            if preds and weights:
                p = np.array(preds, dtype=float)
                w = np.array(weights, dtype=float)
                w = w / max(w.sum(), 1e-12)
                yhat = (p * w[:, None]).sum(axis=0)
                sl_pred = float(yhat[0])
                tp_pred = [float(yhat[1]), float(yhat[2]), float(yhat[3])]
        except Exception:
            pass

    # Blend heuristic + ensemble for stability.
    alpha = 0.65
    sl = _clamp(alpha * sl_pred + (1.0 - alpha) * sl_h, _MIN_SL, _MAX_SL)
    tp1 = _clamp(alpha * tp_pred[0] + (1.0 - alpha) * tp_h[0], _MIN_TP, _MAX_TP)
    tp2 = _clamp(alpha * tp_pred[1] + (1.0 - alpha) * tp_h[1], _MIN_TP, _MAX_TP)
    tp3 = _clamp(alpha * tp_pred[2] + (1.0 - alpha) * tp_h[2], _MIN_TP, _MAX_TP)

    # Enforce monotonic TP ladder.
    if tp2 <= tp1:
        tp2 = _clamp(tp1 * 1.15, _MIN_TP, _MAX_TP)
    if tp3 <= tp2:
        tp3 = _clamp(tp2 * 1.15, _MIN_TP, _MAX_TP)

    return {
        "trade_allowed": True,
        "market_state": "ACTIVE",
        "sl_pct": float(sl),
        "tp_pcts": [float(tp1), float(tp2), float(tp3)],
        "activity_score": float(meta["activity_score"]),
    }


def log_decision_snapshot(
    symbol: str,
    side: str,
    entry_price: float,
    decision: Dict[str, object],
    features: Dict[str, float],
) -> None:
    try:
        _DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": f"{int(time.time() * 1000)}:{symbol}:{side}",
            "ts": int(time.time()),
            "symbol": symbol,
            "side": side,
            "entry_price": float(entry_price),
            "decision": decision,
            "features": features,
            "status": "open",
        }
        with _DECISIONS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def analyze_due_snapshots(lookback_minutes: int = 180) -> int:
    """Run approximate post-trade analysis for snapshots older than ~2h."""
    if not _DECISIONS_PATH.exists():
        return 0
    now_ts = int(time.time())
    rows: List[dict] = []
    try:
        with _DECISIONS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return 0

    processed = 0
    remaining: List[dict] = []
    try:
        from data.historical_loader import load_klines
    except Exception:
        load_klines = None  # type: ignore

    for row in rows:
        try:
            status = str(row.get("status") or "open")
            ts = int(row.get("ts") or 0)
        except Exception:
            remaining.append(row)
            continue
        if status != "open" or ts <= 0:
            remaining.append(row)
            continue
        if (now_ts - ts) < 2 * 3600:
            remaining.append(row)
            continue

        symbol = str(row.get("symbol") or "")
        side = str(row.get("side") or "LONG").upper()
        entry = _safe_float(row.get("entry_price"), 0.0)
        dec = row.get("decision") or {}
        sl_pct = _safe_float((dec or {}).get("sl_pct"), 0.0)
        tps = list((dec or {}).get("tp_pcts") or [])
        if not symbol or entry <= 0 or load_klines is None:
            row["status"] = "analyzed"
            remaining.append(row)
            processed += 1
            continue

        try:
            candles = load_klines(symbol=symbol, interval="1m", limit=max(120, int(lookback_minutes)))
        except Exception:
            candles = []
        highs = [float(c.get("high")) for c in candles if isinstance(c, dict) and c.get("high") is not None]
        lows = [float(c.get("low")) for c in candles if isinstance(c, dict) and c.get("low") is not None]
        if not highs or not lows:
            row["status"] = "analyzed"
            remaining.append(row)
            processed += 1
            continue

        max_h = max(highs)
        min_l = min(lows)
        sl_hit = False
        tp_hits = 0
        if side in ("BUY", "LONG"):
            sl_px = entry * (1.0 - sl_pct / 100.0)
            sl_hit = min_l <= sl_px
            for tp in tps[:3]:
                try:
                    tp_px = entry * (1.0 + float(tp) / 100.0)
                except Exception:
                    continue
                if max_h >= tp_px:
                    tp_hits += 1
        else:
            sl_px = entry * (1.0 + sl_pct / 100.0)
            sl_hit = max_h >= sl_px
            for tp in tps[:3]:
                try:
                    tp_px = entry * (1.0 - float(tp) / 100.0)
                except Exception:
                    continue
                if min_l <= tp_px:
                    tp_hits += 1

        analysis = {
            "id": row.get("id"),
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "tp_hits_2h": int(tp_hits),
            "sl_hit_2h": bool(sl_hit),
            "max_high": float(max_h),
            "min_low": float(min_l),
            "ts_analyzed": now_ts,
        }
        try:
            _ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _ANALYSIS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(analysis, ensure_ascii=False) + "\n")
        except Exception:
            pass
        row["status"] = "analyzed"
        remaining.append(row)
        processed += 1

    try:
        with _DECISIONS_PATH.open("w", encoding="utf-8") as f:
            for row in remaining:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return processed


def train_ensemble_from_log(log_path: Path, min_samples: int = 80) -> Dict[str, float]:
    """Train 5-model SL/TP ensemble from sl_tp_nn_log.csv pairs."""
    if np is None:
        return {"trained": 0.0, "samples": 0.0}
    if (
        GradientBoostingRegressor is None
        or RandomForestRegressor is None
        or ExtraTreesRegressor is None
        or MLPRegressor is None
        or MultiOutputRegressor is None
        or BayesianRidge is None
    ):
        return {"trained": 0.0, "samples": 0.0}
    if not log_path.exists():
        return {"trained": 0.0, "samples": 0.0}

    rows = []
    try:
        import csv
        with log_path.open("r", newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)]
    except Exception:
        rows = []
    if not rows:
        return {"trained": 0.0, "samples": 0.0}

    # Build suggestion->exit pairs by symbol+side.
    rows.sort(key=lambda r: int(_safe_float(r.get("timestamp_ms"), 0.0)))
    last_sugg: Dict[Tuple[str, str], dict] = {}
    pairs: List[Tuple[dict, float, str]] = []
    for r in rows:
        symbol = str(r.get("symbol") or "").strip()
        side_raw = str(r.get("side") or "").strip().upper()
        if side_raw in ("BUY", "LONG"):
            side = "LONG"
        elif side_raw in ("SELL", "SHORT"):
            side = "SHORT"
        else:
            side = side_raw
        if not symbol or not side:
            continue
        key = (symbol, side)
        if str(r.get("ai_sl_pct") or "").strip() != "":
            last_sugg[key] = r
        pnl_raw = r.get("pnl_pct")
        if pnl_raw in (None, "", " "):
            continue
        try:
            pnl = float(pnl_raw)
        except Exception:
            continue
        s = last_sugg.get(key)
        if s is None:
            continue
        pairs.append((s, pnl, str(r.get("reason") or "")))
    if len(pairs) < int(min_samples):
        return {"trained": 0.0, "samples": float(len(pairs))}

    X: List[List[float]] = []
    Y: List[List[float]] = []
    for sugg, pnl, reason in pairs:
        symbol = str(sugg.get("symbol") or "")
        side = str(sugg.get("side") or "LONG")
        ep = _safe_float(sugg.get("entry_price"), 0.0)
        base_sl = _safe_float(sugg.get("base_sl_pct"), 2.0)
        ai_sl = _safe_float(sugg.get("ai_sl_pct"), base_sl)
        tp_raw = str(sugg.get("ai_tp_pcts") or "")
        tp_vals = []
        for t in tp_raw.split(";"):
            tv = _safe_float(t, 0.0)
            if tv > 0:
                tp_vals.append(tv)
        while len(tp_vals) < 3:
            tp_vals.append(tp_vals[-1] * 1.2 if tp_vals else 1.0)

        feat = {}
        try:
            feat = json.loads(str(sugg.get("reason") or "{}"))
            if not isinstance(feat, dict):
                feat = {}
        except Exception:
            feat = {}
        x, _ = _build_feature_vector(symbol, side, ep, base_sl, tp_vals, feat)
        X.append(x)

        # Pseudo-target improvement from realized PnL (safe bounded adaptation).
        pnl_scale = max(min(abs(float(pnl)) / 10.0, 0.25), 0.0)
        if float(pnl) >= 0:
            y_sl = ai_sl * (1.0 + 0.20 * pnl_scale)
            y_tp1 = tp_vals[0] * (1.0 + 0.40 * pnl_scale)
            y_tp2 = tp_vals[1] * (1.0 + 0.45 * pnl_scale)
            y_tp3 = tp_vals[2] * (1.0 + 0.50 * pnl_scale)
        else:
            y_sl = ai_sl * (1.0 - 0.45 * pnl_scale)
            y_tp1 = tp_vals[0] * (1.0 - 0.25 * pnl_scale)
            y_tp2 = tp_vals[1] * (1.0 - 0.30 * pnl_scale)
            y_tp3 = tp_vals[2] * (1.0 - 0.35 * pnl_scale)

        # Exit reason aware correction
        rsn = str(reason).upper()
        if "SL" in rsn:
            y_sl *= 0.92
            y_tp1 *= 0.95
            y_tp2 *= 0.95
            y_tp3 *= 0.95
        elif "TP" in rsn:
            y_tp1 *= 1.02
            y_tp2 *= 1.03
            y_tp3 *= 1.04

        Y.append([
            _clamp(float(y_sl), _MIN_SL, _MAX_SL),
            _clamp(float(y_tp1), _MIN_TP, _MAX_TP),
            _clamp(float(y_tp2), _MIN_TP, _MAX_TP),
            _clamp(float(y_tp3), _MIN_TP, _MAX_TP),
        ])

    if len(X) < int(min_samples):
        return {"trained": 0.0, "samples": float(len(X))}

    X_arr = np.array(X, dtype=float)
    Y_arr = np.array(Y, dtype=float)
    n = X_arr.shape[0]
    split = int(n * 0.8)
    if split <= 20 or split >= n:
        split = max(20, n // 2)
    xtr, xva = X_arr[:split], X_arr[split:]
    ytr, yva = Y_arr[:split], Y_arr[split:]

    models = {
        "gbr": MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
        "rf": RandomForestRegressor(n_estimators=240, max_depth=8, n_jobs=-1, random_state=42),
        "mlp": MLPRegressor(hidden_layer_sizes=(96, 48), max_iter=350, random_state=42),
        "temporal_et": ExtraTreesRegressor(n_estimators=220, max_depth=10, n_jobs=-1, random_state=42),
        "prob_bayes": MultiOutputRegressor(BayesianRidge()),
    }

    fitted = {}
    maes = {}
    for name, m in models.items():
        try:
            m.fit(xtr, ytr)
            pred = m.predict(xva)
            mae = float(np.mean(np.abs(pred - yva))) if yva.size else 1.0
            fitted[name] = m
            maes[name] = max(mae, 1e-6)
        except Exception:
            continue
    if not fitted:
        return {"trained": 0.0, "samples": float(len(X))}

    # Inverse-MAE weighting for ensemble aggregation.
    inv = {k: 1.0 / v for k, v in maes.items()}
    s = sum(inv.values()) if inv else 0.0
    weights = {k: (inv[k] / s if s > 0 else 1.0 / float(len(inv))) for k in inv}

    out = {
        "models": fitted,
        "weights": weights,
        "meta": {
            "trained_at": int(time.time()),
            "samples": int(len(X)),
            "val_mae": maes,
        },
    }
    try:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _MODEL_PATH.open("wb") as f:
            pickle.dump(out, f)
    except Exception:
        return {"trained": 0.0, "samples": float(len(X))}

    return {"trained": 1.0, "samples": float(len(X))}

