import json
import math
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from execution.sl_tp_ai_engine import decide_sltp, log_decision_snapshot


_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "data" / "sl_tp_nn_weights.json"
_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "sl_tp_nn_log.csv"
_SYMBOL_STATS_PATH = Path(__file__).resolve().parents[1] / "data" / "sl_tp_nn_symbol_stats.json"
_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "sl_tp_nn_model.pkl"
_LOCK = threading.Lock()

# In-memory cache for per-symbol/per-side performance statistics produced by
# ml/train_sl_tp_nn.py. Lazily loaded on first use.
_SYMBOL_STATS_CACHE: Optional[Dict[str, Dict[str, float]]] = None

# In-memory cache for the supervised SL/TP model (RandomForestRegressor) trained
# offline by ml/train_sl_tp_nn.py. Reloaded when the underlying pickle file
# changes on disk so that periodic training immediately affects new signals.
_SUPERVISED_MODEL = None
_SUPERVISED_MODEL_MTIME: Optional[float] = None

# Hard safety guardrails for AI-generated SL/TP percentages. These bounds are
# deliberately conservative so that, even if future training logic or weights
# misbehave, SL/TP distances remain within a sane and tradable range.
_MIN_SL_PCT = 0.3
_MAX_SL_PCT = 7.0
_MIN_TP_PCT = 0.5
_MAX_TP_PCT = 25.0
_MAX_TP_LEVELS = 3


def _ensure_files() -> None:
    if not _WEIGHTS_PATH.parent.exists():
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _LOG_PATH.exists():
        with _LOG_PATH.open("w", encoding="utf-8") as f:
            f.write(
                "timestamp_ms,symbol,side,entry_price,base_sl_pct,base_tp_pcts,ai_sl_pct,ai_tp_pcts,pnl_pct,reason\n"
            )


def _load_weights() -> Dict[str, float]:
    _ensure_files()
    if not _WEIGHTS_PATH.exists():
        return {}
    try:
        raw = _WEIGHTS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _load_symbol_stats() -> Dict[str, Dict[str, float]]:
    """Load per-symbol/per-side stats from JSON, caching in memory.

    The JSON file is produced offline by ml/train_sl_tp_nn.py and contains
    keys of the form "SYMBOL:SIDE" (e.g. "BTCUSDT:LONG"). When unavailable
    or malformed, we simply return an empty dict so that online behaviour
    gracefully falls back to global SL/TP settings.
    """

    global _SYMBOL_STATS_CACHE
    if _SYMBOL_STATS_CACHE is not None:
        return _SYMBOL_STATS_CACHE

    stats: Dict[str, Dict[str, float]] = {}
    try:
        if _SYMBOL_STATS_PATH.exists():
            raw = _SYMBOL_STATS_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(k, str) or not isinstance(v, dict):
                        continue
                    stats[k] = v
    except Exception:
        stats = {}

    _SYMBOL_STATS_CACHE = stats
    return stats


def _load_supervised_model():
    """Load the supervised SL/TP model (if available) with mtime-based cache.

    When the model file or required libraries are missing, this returns None so
    that online behaviour gracefully falls back to the lightweight NN layer and
    per-symbol/regime heuristics.
    """

    global _SUPERVISED_MODEL, _SUPERVISED_MODEL_MTIME

    try:
        if not _MODEL_PATH.exists():
            _SUPERVISED_MODEL = None
            _SUPERVISED_MODEL_MTIME = None
            return None

        mtime = float(_MODEL_PATH.stat().st_mtime)
        if _SUPERVISED_MODEL is not None and _SUPERVISED_MODEL_MTIME == mtime:
            return _SUPERVISED_MODEL

        with _MODEL_PATH.open("rb") as f:
            model = pickle.load(f)

        _SUPERVISED_MODEL = model
        _SUPERVISED_MODEL_MTIME = mtime
        return model
    except Exception:
        _SUPERVISED_MODEL = None
        _SUPERVISED_MODEL_MTIME = None
        return None


def _save_weights(weights: Dict[str, float]) -> None:
    _ensure_files()
    try:
        _WEIGHTS_PATH.write_text(json.dumps(weights, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


def _log_row(row: List[str]) -> None:
    # Skip CSV logging during backtest to avoid file I/O bottleneck
    if os.environ.get("BACKTEST_MODE"):
        return
    _ensure_files()
    line = ",".join(row) + "\n"
    with _LOCK:
        try:
            with _LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            return


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _forward(base_sl_pct: float, base_tp_pcts: List[float]) -> Tuple[float, List[float]]:
    """Very small NN-like layer over base SL/TP percents.

    For now this behaves as an identity-ish transform but is parameterised so
    that future online updates can gradually adjust distances.
    """

    # Load shared weights once per call; guarded by a global lock for safety.
    with _LOCK:
        w = _load_weights()

    # Bias terms (default 0 -> identity).
    w_sl_bias = float(w.get("sl_bias", 0.0))
    w_tp_scale = float(w.get("tp_scale", 1.0))

    # Apply lightweight parametric layer.
    sl_adj = float(base_sl_pct) + w_sl_bias
    tp_adj = [float(p) * w_tp_scale for p in base_tp_pcts]

    # Enforce stricter safety bounds so SL/TP never become absurdly tight or
    # wide, even if weights drift. We both clamp absolute values and cap the
    # number of TP levels that downstream logic will see.
    if not math.isfinite(sl_adj):
        sl_adj = float(base_sl_pct)
    sl_adj = max(sl_adj, _MIN_SL_PCT)
    sl_adj = min(sl_adj, _MAX_SL_PCT)

    safe_tps: List[float] = []
    for v in tp_adj:
        if not math.isfinite(v):
            continue
        v = max(v, _MIN_TP_PCT)
        v = min(v, _MAX_TP_PCT)
        safe_tps.append(v)

    if not safe_tps:
        # Fallback: if scaling produced no valid TP levels, revert to base
        # config clamped to the same safety envelope.
        for p in base_tp_pcts:
            try:
                v = float(p)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            v = max(v, _MIN_TP_PCT)
            v = min(v, _MAX_TP_PCT)
            safe_tps.append(v)

    return sl_adj, safe_tps[:_MAX_TP_LEVELS]


def suggest_sl_tp(
    symbol: str,
    side: str,
    entry_price: float,
    base_sl_pct: float,
    base_tp_pcts: List[float],
    features: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Return AI-style SL/TP percentages based on base config and features.

    This is intentionally lightweight and safe for now: it wraps base
    percent-based SL/TP distances with a tiny parametric layer that can be
    tuned over time. It never changes core trading logic; it only affects
    what is shown in Telegram and what is logged for future learning.
    """

    try:
        ep = float(entry_price)
    except (TypeError, ValueError):
        ep = 0.0

    if ep <= 0 or base_sl_pct is None or not base_tp_pcts:
        return {
            "sl_pct": base_sl_pct,
            "tp_pcts": list(base_tp_pcts) if base_tp_pcts else [],
        }

    if features is None:
        features = {}
    else:
        features = dict(features)

    # Inject historical symbol outcomes so decision engine can use them.
    try:
        stats_map = _load_symbol_stats()
        side_key = str(side).upper()
        if side_key in ("BUY", "LONG"):
            dir_label = "LONG"
        elif side_key in ("SELL", "SHORT"):
            dir_label = "SHORT"
        else:
            dir_label = side_key or "LONG"
        sym_stats = stats_map.get(f"{symbol}:{dir_label}") if isinstance(stats_map, dict) else None
        if isinstance(sym_stats, dict):
            features["symbol_win_rate"] = _safe_float(sym_stats.get("win_rate"), 0.5)
            features["symbol_avg_pnl"] = _safe_float(sym_stats.get("mean_pnl"), 0.0)
            # drawdown proxy until dedicated per-symbol DD series is available.
            features["symbol_avg_drawdown"] = min(0.0, _safe_float(sym_stats.get("mean_pnl"), 0.0))
    except Exception:
        pass

    # Extract a compact JSON payload of core features for offline training.
    feat_payload = ""
    try:
        try:
            atr_val = float(features.get("atr", 0.0))
        except (TypeError, ValueError):
            atr_val = 0.0
        try:
            conf_val = float(features.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf_val = 0.0

        # Extended feature payload used by AI SL/TP ensemble training.
        payload = {
            "atr": atr_val,
            "confidence": conf_val,
            "last_price": _safe_float(features.get("last_price", ep), ep),
            "candle_range_pct": _safe_float(features.get("candle_range_pct", 0.0), 0.0),
            "trend_strength": _safe_float(features.get("trend_strength", 0.0), 0.0),
            "structure_level": _safe_float(features.get("structure_level", 0.0), 0.0),
            "volatility_1m": _safe_float(features.get("volatility_1m", 0.0), 0.0),
            "volatility_5m": _safe_float(features.get("volatility_5m", 0.0), 0.0),
            "volatility_1h": _safe_float(features.get("volatility_1h", 0.0), 0.0),
            "range_expansion": _safe_float(features.get("range_expansion", 0.0), 0.0),
            "volume": _safe_float(features.get("volume", 0.0), 0.0),
            "rsi": _safe_float(features.get("rsi", 50.0), 50.0),
            "momentum": _safe_float(features.get("momentum", 0.0), 0.0),
            "acceleration": _safe_float(features.get("acceleration", 0.0), 0.0),
            "signal_strength": _safe_float(features.get("signal_strength", conf_val), conf_val),
            "symbol_win_rate": _safe_float(features.get("symbol_win_rate", 0.5), 0.5),
            "symbol_avg_pnl": _safe_float(features.get("symbol_avg_pnl", 0.0), 0.0),
            "symbol_avg_drawdown": _safe_float(features.get("symbol_avg_drawdown", 0.0), 0.0),
            "recent_pnl_ema": _safe_float(features.get("recent_pnl_ema", 0.0), 0.0),
            "recent_hit_rate": _safe_float(features.get("recent_hit_rate", 0.5), 0.5),
        }
        feat_payload = json.dumps(payload, ensure_ascii=False)
    except Exception:
        feat_payload = ""

    # New AI decision engine (ensemble + activity detector).
    ai_decision = None
    try:
        ai_decision = decide_sltp(
            symbol=symbol,
            side=side,
            entry_price=ep,
            base_sl_pct=float(base_sl_pct),
            base_tp_pcts=list(base_tp_pcts),
            features=dict(features or {}),
        )
    except Exception:
        ai_decision = None

    # Fallback to legacy tiny wrapper if AI decision engine fails.
    sl_pct_ai, tp_pcts_ai = _forward(float(base_sl_pct), list(base_tp_pcts))
    trade_allowed = True
    market_state = "ACTIVE"
    if isinstance(ai_decision, dict):
        try:
            trade_allowed = bool(ai_decision.get("trade_allowed", True))
        except Exception:
            trade_allowed = True
        try:
            market_state = str(ai_decision.get("market_state", "ACTIVE")).upper()
        except Exception:
            market_state = "ACTIVE"
        try:
            sl_pct_ai = float(ai_decision.get("sl_pct", sl_pct_ai))
        except Exception:
            pass
        try:
            _tps = ai_decision.get("tp_pcts") or tp_pcts_ai
            tp_pcts_ai = [float(v) for v in list(_tps)[:3]]
        except Exception:
            pass

    # --- Per-symbol/per-regime adaptive tweaks (kept intentionally mild) ---
    sl_adj = float(sl_pct_ai)
    tp_adj = list(tp_pcts_ai)

    # 1) Symbol-level performance adjustment: for symbols/directions with a
    # sufficiently long and strong track record, allow slightly wider TPs; for
    # weak ones, tighten both SL and TPs a bit. Guardrails later ensure all
    # values stay within [_MIN_*, _MAX_*].
    try:
        stats_map = _load_symbol_stats()
        side_key = side.upper()
        if side_key in ("BUY", "LONG"):
            dir_label = "LONG"
        elif side_key in ("SELL", "SHORT"):
            dir_label = "SHORT"
        else:
            dir_label = side_key or "LONG"

        sym_key = f"{symbol}:{dir_label}"
        sym_stats = stats_map.get(sym_key)
        if isinstance(sym_stats, dict):
            try:
                n_trades = float(sym_stats.get("n", 0.0))
            except (TypeError, ValueError):
                n_trades = 0.0
            try:
                win_rate = float(sym_stats.get("win_rate", 0.0))
            except (TypeError, ValueError):
                win_rate = 0.0
            try:
                mean_pnl = float(sym_stats.get("mean_pnl", 0.0))
            except (TypeError, ValueError):
                mean_pnl = 0.0

            # Only adapt for symbols with a reasonable history.
            if n_trades >= 20.0:
                sl_mult = 1.0
                tp_mult = 1.0

                if win_rate >= 0.60 and mean_pnl > 0.0:
                    # Strong symbol/direction: reward a bit more.
                    tp_mult = 1.10
                elif win_rate <= 0.40 or mean_pnl < 0.0:
                    # Weak symbol/direction: cut risk slightly.
                    sl_mult = 0.90
                    tp_mult = 0.90

                sl_adj *= sl_mult
                tp_adj = [p * tp_mult for p in tp_adj]
    except Exception:
        sl_adj = float(sl_pct_ai)
        tp_adj = list(tp_pcts_ai)

    # 2) Simple regime heuristic using ATR/entry_price as a volatility proxy.
    #    Higher volatility → more trending → allow a touch wider TPs; very low
    #    volatility → likely ranging → keep things tighter.
    try:
        atr_val = float(features.get("atr", 0.0)) if features is not None else 0.0
    except (TypeError, ValueError):
        atr_val = 0.0

    vol_pct = 0.0
    if ep > 0.0 and atr_val > 0.0:
        vol_pct = abs(atr_val / ep) * 100.0

    # Use soft boundaries for a 3-regime split: LOW / NORMAL / HIGH volatility.
    if vol_pct >= 4.0:
        # Trending/high-vol regime: slightly wider TPs.
        tp_adj = [p * 1.05 for p in tp_adj]
    elif vol_pct <= 1.0 and vol_pct > 0.0:
        # Very quiet/range-bound: slightly tighter SL/TP.
        sl_adj *= 0.95
        tp_adj = [p * 0.95 for p in tp_adj]

    # 3) Supervised model adjustment: use the offline-trained RandomForest
    #    regressor as a mild value-function over the current SL/TP proposal.
    #    Predictions remain advisory only: effect is capped to +/-5% scaling.
    try:
        model = _load_supervised_model()
        if model is not None and ep > 0.0:
            # Build feature vector compatible with ml/train_sl_tp_nn._build_dataset_from_pairs.
            try:
                conf_val = float(features.get("confidence", 0.0)) if features is not None else 0.0
            except (TypeError, ValueError):
                conf_val = 0.0

            side_key = side.upper()
            if side_key in ("BUY", "LONG"):
                side_sign = 1.0
            elif side_key in ("SELL", "SHORT"):
                side_sign = -1.0
            else:
                side_sign = 0.0

            base_tp_mean = 0.0
            base_tp_max = 0.0
            if base_tp_pcts:
                try:
                    vals = [float(p) for p in base_tp_pcts if float(p) > 0]
                except Exception:
                    vals = []
                if vals:
                    base_tp_mean = sum(vals) / float(len(vals))
                    base_tp_max = max(vals)

            ai_tp_mean = 0.0
            ai_tp_max = 0.0
            if tp_adj:
                try:
                    vals_ai = [float(p) for p in tp_adj if float(p) > 0]
                except Exception:
                    vals_ai = []
                if vals_ai:
                    ai_tp_mean = sum(vals_ai) / float(len(vals_ai))
                    ai_tp_max = max(vals_ai)
            else:
                ai_tp_mean = base_tp_mean
                ai_tp_max = base_tp_max

            feats = [
                float(atr_val),
                float(conf_val),
                float(side_sign),
                float(base_sl_pct),
                float(sl_adj),
                float(base_tp_mean),
                float(ai_tp_mean),
                float(base_tp_max),
                float(ai_tp_max),
            ]

            try:
                pred = model.predict([feats])
                if isinstance(pred, (list, tuple)):
                    pnl_pred = float(pred[0])
                else:
                    pnl_pred = float(pred)
            except Exception:
                pnl_pred = 0.0

            # Map predicted pnl into a small scaling band. Strongly positive
            # predictions widen TPs slightly; strongly negative predictions
            # tighten both SL and TPs slightly.
            if pnl_pred >= 3.0:
                tp_adj = [p * 1.05 for p in tp_adj]
            elif pnl_pred <= -3.0:
                sl_adj *= 0.95
                tp_adj = [p * 0.95 for p in tp_adj]
    except Exception:
        # Any supervised-model issue should never affect live behaviour.
        pass

    # Final safety clamp on adjusted values.
    if not math.isfinite(sl_adj):
        sl_adj = float(sl_pct_ai)
    sl_adj = max(sl_adj, _MIN_SL_PCT)
    sl_adj = min(sl_adj, _MAX_SL_PCT)

    safe_tp_final: List[float] = []
    for v in tp_adj:
        if not math.isfinite(v):
            continue
        v = max(v, _MIN_TP_PCT)
        v = min(v, _MAX_TP_PCT)
        safe_tp_final.append(v)
    if not safe_tp_final:
        safe_tp_final = list(tp_pcts_ai)


    # Log suggestion for future training.
    ts_ms = int(time.time() * 1000)
    row = [
        str(ts_ms),
        symbol,
        side,
        f"{ep:.8f}",
        f"{float(base_sl_pct):.6f}",
        ";".join(f"{float(p):.6f}" for p in base_tp_pcts),
        f"{sl_adj:.6f}",
        ";".join(f"{float(p):.6f}" for p in safe_tp_final),
        "",  # pnl_pct placeholder, filled on exit
        feat_payload,
    ]
    _log_row(row)

    out = {
        "trade_allowed": bool(trade_allowed),
        "market_state": str(market_state),
        "sl_pct": sl_adj,
        "tp_pcts": safe_tp_final,
    }
    try:
        if bool(trade_allowed):
            log_decision_snapshot(
                symbol=symbol,
                side=side,
                entry_price=ep,
                decision=out,
                features=dict(features or {}),
            )
    except Exception:
        pass
    return out


def on_trade_exit(
    account_index: int,
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    reason: str,
) -> None:
    """Hook called when a real position hits TP/SL/timeout.

    For now this only logs realised PnL and reason into the same CSV so that
    an offline trainer can later correlate suggestions with outcomes and
    update weights. It does not change any live trading decisions.
    """

    try:
        ep = float(entry_price)
        xp = float(exit_price)
    except (TypeError, ValueError):
        return

    try:
        pnl_val = float(pnl_pct)
    except (TypeError, ValueError):
        pnl_val = 0.0

    ts_ms = int(time.time() * 1000)
    row = [
        str(ts_ms),
        symbol,
        side,
        f"{ep:.8f}",
        "",  # base_sl_pct unknown here
        "",  # base_tp_pcts unknown here
        "",  # ai_sl_pct unknown here
        "",  # ai_tp_pcts unknown here
        f"{pnl_val:.6f}",
        reason,
    ]
    _log_row(row)

    # Placeholder for future online learning; intentionally a no-op for now to
    # avoid touching live behaviour. An offline trainer can read the CSV and
    # adjust weights periodically.
    return
