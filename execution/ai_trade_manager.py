import json
import random
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
except Exception:  # pragma: no cover
    RandomForestClassifier = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    ExtraTreesClassifier = None  # type: ignore
    MLPClassifier = None  # type: ignore
    GaussianNB = None  # type: ignore

try:  # optional advanced experts
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore

torch = None  # type: ignore  # imported lazily on first use to avoid CUDA init hang at startup
nn = None  # type: ignore

from data.live_stream import fetch_candles
from config import settings as app_settings
from execution.binance_futures import (
    _get_clients,
    cancel_order_by_id,
    cancel_symbol_open_orders,
    close_position_market,
    get_open_positions_snapshot,
    place_exchange_stop_order,
    place_order,
)
from features.feature_store import build_features
from monitoring.logger import log


_ROOT = Path(__file__).resolve().parents[1]
_MODEL_PATH = _ROOT / "data" / "ai_trade_manager_model.pkl"
_DECISION_LOG_PATH = _ROOT / "data" / "ai_trade_manager_decisions.jsonl"
_OUTCOME_LOG_PATH = _ROOT / "data" / "ai_trade_manager_outcomes.jsonl"
_CREDIT_LOG_PATH = _ROOT / "data" / "ai_trade_manager_credit.jsonl"
_LIFECYCLE_LOG_PATH = _ROOT / "data" / "ai_trade_manager_lifecycle.jsonl"
_CALIBRATION_LOG_PATH = _ROOT / "data" / "ai_trade_manager_calibration.jsonl"
_NORM_STATS_PATH = _ROOT / "data" / "ai_trade_manager_symbol_norm_stats.json"

_ACTIONS = ("HOLD", "MOVE_SL", "MOVE_TP", "SCALE_IN", "CLOSE_TRADE", "REMOVE_TP")
_RECHECK_SEC = 60.0
_CACHE_TTL_SEC = 2 * 3600.0

_MODEL = None
_MODEL_MTIME: Optional[float] = None


if nn is not None:
    class _LSTMNet(nn.Module):
        def __init__(self, input_dim: int, out_dim: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=int(input_dim), hidden_size=48, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(48, 32),
                nn.ReLU(),
                nn.Linear(32, int(out_dim)),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            h = out[:, -1, :]
            return self.head(h)


class _TorchLSTMActionModel:
    """Optional temporal/LSTM expert for action classification.

    Uses a tiny LSTM over sequence length 1..N. In inference we support
    single-step vectors as a safe fallback.
    """

    def __init__(self, input_dim: int, actions: List[str]):
        self.input_dim = int(input_dim)
        self.actions = list(actions)
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(self.actions)}
        self.model = None

    def _build(self):
        if torch is None or nn is None:
            return None
        self.model = _LSTMNet(self.input_dim, len(self.actions))
        return self.model

    def fit(self, X, y):
        global torch, nn
        if torch is None:
            try:
                import torch as _torch, torch.nn as _nn
                torch = _torch; nn = _nn  # type: ignore
                if not hasattr(_LSTMNet, '__bases__'):
                    # Re-register base class now that nn is available
                    pass
            except Exception:
                pass
        if torch is None or nn is None or np is None:
            return self
        try:
            arr_x = np.asarray(X, dtype=float)
            if arr_x.ndim == 2:
                arr_x = arr_x[:, None, :]
            arr_y = np.asarray([self.action_to_idx.get(str(v), 0) for v in y], dtype=np.int64)
            if arr_x.shape[0] < 80:
                return self
            model = self._build()
            if model is None:
                return self
            device = "cpu"
            x_t = torch.tensor(arr_x, dtype=torch.float32, device=device)
            y_t = torch.tensor(arr_y, dtype=torch.long, device=device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()
            model.train()
            bs = 64
            n = x_t.shape[0]
            for _ in range(12):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    xb = x_t[idx]
                    yb = y_t[idx]
                    opt.zero_grad()
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
        except Exception:
            return self
        return self

    def predict_proba(self, X):
        if torch is None or self.model is None or np is None:
            return np.zeros((len(X), len(self.actions)), dtype=float) if np is not None else []
        try:
            arr_x = np.asarray(X, dtype=float)
            if arr_x.ndim == 2:
                arr_x = arr_x[:, None, :]
            x_t = torch.tensor(arr_x, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs
        except Exception:
            return np.zeros((len(X), len(self.actions)), dtype=float)

    def predict(self, X):
        probs = self.predict_proba(X)
        try:
            idx = np.argmax(probs, axis=1)
            return np.array([self.idx_to_action.get(int(i), "HOLD") for i in idx], dtype=object)
        except Exception:
            return np.array(["HOLD"] * len(X), dtype=object)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_model():
    global _MODEL, _MODEL_MTIME
    try:
        if not _MODEL_PATH.exists():
            _MODEL = None
            _MODEL_MTIME = None
            return None
        mt = float(_MODEL_PATH.stat().st_mtime)
        if _MODEL is not None and _MODEL_MTIME == mt:
            return _MODEL
        import pickle
        with _MODEL_PATH.open("rb") as f:
            m = pickle.load(f)
        _MODEL = m
        _MODEL_MTIME = mt
        return m
    except Exception:
        _MODEL = None
        _MODEL_MTIME = None
        return None


def _append_jsonl(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def _build_management_features(
    symbol: str,
    side: str,
    entry_price: float,
    current_price: float,
    unreal_pnl_pct: float,
    time_since_entry_sec: float,
    candles: List[dict],
    features: Dict[str, object],
    ctx: Dict[str, object],
) -> List[float]:
    s_up = str(side).upper()
    side_sign = 1.0 if s_up in ("BUY", "LONG") else -1.0
    atr = _safe_float(features.get("atr"))
    ep = max(_safe_float(entry_price), 1e-12)
    atr_pct = abs(atr / ep) * 100.0 if atr > 0 else 0.0
    momentum = _safe_float(features.get("momentum"))
    accel = _safe_float(features.get("acceleration"))
    rsi = _safe_float(features.get("rsi"), 50.0)
    volume = 0.0
    candle_range_pct = 0.0
    if candles:
        last = candles[-1]
        volume = _safe_float(last.get("volume"))
        h = _safe_float(last.get("high"))
        l = _safe_float(last.get("low"))
        c = _safe_float(last.get("close"), current_price)
        if c > 0:
            candle_range_pct = abs(h - l) / c * 100.0

    # Volatility windows from returns.
    closes = [_safe_float(c.get("close")) for c in candles if isinstance(c, dict)]

    def _vol(win: int) -> float:
        if len(closes) < win + 1:
            return 0.0
        rs = []
        sub = closes[-(win + 1):]
        for i in range(1, len(sub)):
            p0, p1 = sub[i - 1], sub[i]
            if p0 > 0:
                rs.append((p1 - p0) / p0 * 100.0)
        if not rs:
            return 0.0
        mu = sum(rs) / float(len(rs))
        var = sum((x - mu) ** 2 for x in rs) / float(max(1, len(rs)))
        return var ** 0.5

    vol_1m = abs((closes[-1] - closes[-2]) / closes[-2] * 100.0) if len(closes) >= 2 and closes[-2] > 0 else 0.0
    vol_5m = _vol(5)
    vol_1h = _vol(60)

    market_state = str(ctx.get("market_state") or "ACTIVE").upper()
    market_state_code = 1.0 if market_state == "ACTIVE" else 0.0

    return [
        side_sign,
        float(current_price),
        float(entry_price),
        float(unreal_pnl_pct),
        float(time_since_entry_sec / 60.0),
        float(atr_pct),
        float(vol_1m),
        float(vol_5m),
        float(vol_1h),
        float(candle_range_pct),
        float(volume),
        float(rsi),
        float(momentum),
        float(accel),
        float(market_state_code),
    ]


class AITradeManager:
    """Isolated AI trade manager for already-open positions only."""

    def __init__(self, get_accounts_cfg_live):
        self._get_accounts_cfg_live = get_accounts_cfg_live
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._ctx: Dict[Tuple[int, str], dict] = {}
        self._portfolio_ctx: Dict[int, dict] = {}
        self._symbol_schedule: Dict[Tuple[int, str], dict] = {}
        self._symbol_locks: Dict[Tuple[int, str], threading.Lock] = {}
        self._lifecycle: Dict[Tuple[int, str], dict] = {}
        self._pending_credit: Dict[Tuple[int, str], List[dict]] = {}
        self._symbol_expert_perf: Dict[str, Dict[str, dict]] = {}
        self._conf_calibration: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._metrics: Dict[str, float] = {
            "hold_count": 0.0,
            "move_sl_count": 0.0,
            "close_count": 0.0,
            "total_decisions": 0.0,
            "last_kpi_log_ts": 0.0,
            "giveback_events": 0.0,
            "giveback_no_sl_events": 0.0,
            "orphan_cancel_count": 0.0,
        }
        self._last_close_ts: Dict[Tuple[int, str], float] = {}
        self._perf_window: deque = deque(maxlen=50)
        self._perf_stats: Dict[str, float] = {
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "last_log_ts": 0.0,
        }
        self._train_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai_tm_train")
        self._io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ai_tm_io")
        self._train_future = None
        self._last_train_ts = 0.0
        self._norm_cache: Dict[str, dict] = {}
        self._last_norm_flush_ts = 0.0
        self._open_orders_cache: Dict[Tuple[int, str], dict] = {}
        self._order_advisory_cache: Dict[Tuple[int, str], dict] = {}
        self._load_norm_cache()

    def stop(self) -> None:
        self._stop.set()
        self._flush_norm_cache(force=True)
        try:
            self._io_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self._train_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def _symbol_interval_target_sec(self, settings: dict) -> float:
        if not self._flag_enabled(settings, "ai_tm_fast_loop_enabled", False):
            return float(_RECHECK_SEC)
        base = _safe_float((settings or {}).get("ai_tm_symbol_interval_sec"), 3.0)
        return min(8.0, max(2.0, base))

    def _symbol_interval_jittered_sec(self, settings: dict) -> float:
        base = self._symbol_interval_target_sec(settings or {})
        jitter_pct = _safe_float((settings or {}).get("ai_tm_symbol_interval_jitter_pct"), 0.15)
        jitter_pct = min(0.50, max(0.0, jitter_pct))
        delta = base * jitter_pct
        return min(8.0, max(2.0, base + random.uniform(-delta, delta)))

    def _flag_enabled(self, settings: dict, key: str, default: bool = False) -> bool:
        global_map = {
            "ai_tm_ml_first_enabled": "AITM_ML_FIRST_ENABLED",
            "ai_tm_symbol_normalization_enabled": "AITM_SYMBOL_NORMALIZATION_ENABLED",
            "ai_tm_fast_loop_enabled": "AITM_FAST_LOOP_ENABLED",
            "ai_tm_credit_linkage_enabled": "AITM_INTERMEDIATE_CREDIT_LOG_ENABLED",
            "ai_tm_mode_policy_enabled": "AITM_DYNAMIC_MODE_POLICY_ENABLED",
            "ai_tm_override_transparency_enabled": "AITM_OVERRIDE_TRANSPARENCY_ENABLED",
            "ai_tm_breakeven_lock_enabled": "AITM_BREAKEVEN_LOCK_ENABLED",
            "ai_tm_tranche_tp_enabled": "AITM_TRANCHE_TP_ENABLED",
            "ai_tm_orphan_cleanup_enabled": "AITM_ORPHAN_CLEANUP_ENABLED",
            "ai_tm_ml_strict_mode": "AITM_ML_STRICT_MODE",
        }
        if key in global_map:
            gv = getattr(app_settings, global_map[key], default)
            v_local = (settings or {}).get(key)
            if v_local is None:
                return bool(gv)
            return bool(v_local)
        v = (settings or {}).get(key)
        if v is None:
            return bool(default)
        return bool(v)

    def _emit_latency_warnings(self, action: str, acc_idx: int, symbol: str, elapsed_sec: float, settings: dict) -> None:
        """Emit two-level latency warnings for execution-critical operations."""
        try:
            sec = float(max(elapsed_sec, 0.0))
            if sec > 5.0:
                log(
                    f"[AI-TM][LATENCY][WARN][CRITICAL] account={acc_idx} symbol={symbol} "
                    f"action={action} elapsed={sec:.2f}s threshold=5.00s"
                )
            elif sec > 3.0:
                log(
                    f"[AI-TM][LATENCY][WARN] account={acc_idx} symbol={symbol} "
                    f"action={action} elapsed={sec:.2f}s threshold=3.00s"
                )
            elif sec > min(5.0, max(3.0, _safe_float(settings.get("ai_tm_latency_warn_sec"), 5.0))):
                # Backward-compatible custom threshold support.
                warn_sec = min(5.0, max(3.0, _safe_float(settings.get("ai_tm_latency_warn_sec"), 5.0)))
                log(
                    f"[AI-TM][LATENCY][WARN] account={acc_idx} symbol={symbol} "
                    f"action={action} elapsed={sec:.2f}s threshold={warn_sec:.2f}s"
                )
        except Exception:
            return

    def _detect_market_regime(self, volatility_pct: float, trend_score: float) -> str:
        """Classify short-term market regime for adaptive thresholds."""
        v = abs(_safe_float(volatility_pct, 0.0))
        t = abs(_safe_float(trend_score, 0.0))
        trend_cut = max(0.35, _safe_float(getattr(app_settings, "AI_TM_TRENDING_REGIME_CUT", 0.45), 0.45))
        if t >= trend_cut:
            return "TRENDING"
        if v < 0.5:
            return "LOW_VOL"
        if v <= 1.5:
            return "NORMAL"
        return "HIGH_VOL"

    def _adaptive_thresholds(self, regime: str) -> dict:
        """Adaptive risk/trend thresholds by detected regime."""
        r = str(regime or "NORMAL").upper()
        if r == "LOW_VOL":
            return {"risk_protect": 0.60, "trend_recover": 0.25}
        if r == "HIGH_VOL":
            return {"risk_protect": 0.55, "trend_recover": 0.45}
        if r == "TRENDING":
            return {"risk_protect": 0.75, "trend_recover": 0.30}
        return {"risk_protect": 0.70, "trend_recover": 0.35}

    def _update_perf_stats(self, pnl_pct: float) -> None:
        try:
            v = float(pnl_pct)
        except Exception:
            return
        self._perf_window.append(v)
        if not self._perf_window:
            return
        arr = list(self._perf_window)
        wins = [x for x in arr if x > 0.0]
        losses = [x for x in arr if x < 0.0]
        win_rate = (len(wins) / float(len(arr))) * 100.0 if arr else 0.0
        avg_profit = (sum(wins) / float(len(wins))) if wins else 0.0
        avg_loss = (sum(losses) / float(len(losses))) if losses else 0.0
        # max drawdown over rolling equity proxy
        eq = 0.0
        peak = 0.0
        mdd = 0.0
        for x in arr:
            eq += float(x)
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
        self._perf_stats.update(
            {
                "win_rate": float(win_rate),
                "avg_profit": float(avg_profit),
                "avg_loss": float(avg_loss),
                "max_drawdown": float(mdd),
            }
        )
        now = time.time()
        if (now - _safe_float(self._perf_stats.get("last_log_ts"), 0.0)) >= 30.0:
            self._perf_stats["last_log_ts"] = float(now)
            log(
                f"[AI-TM][PERF] win_rate_50={win_rate:.2f}% avg_profit={avg_profit:.4f}% "
                f"avg_loss={avg_loss:.4f}% max_drawdown_50={mdd:.4f}% trades={len(arr)}"
            )

    def _load_norm_cache(self) -> None:
        try:
            if not _NORM_STATS_PATH.exists():
                self._norm_cache = {}
                return
            with _NORM_STATS_PATH.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self._norm_cache = payload if isinstance(payload, dict) else {}
        except Exception:
            self._norm_cache = {}

    def _flush_norm_cache(self, force: bool = False) -> None:
        try:
            now = time.time()
            if (not force) and (now - self._last_norm_flush_ts) < 30.0:
                return
            _NORM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _NORM_STATS_PATH.open("w", encoding="utf-8") as f:
                json.dump(self._norm_cache, f, ensure_ascii=False)
            self._last_norm_flush_ts = now
        except Exception:
            return

    def _rolling_norm(
        self,
        symbol: str,
        field: str,
        value: float,
        window: int,
        warmup: int,
        fallback_entry: float,
    ) -> Tuple[float, dict]:
        s = str(symbol).upper()
        with self._state_lock:
            s_row = self._norm_cache.setdefault(s, {})
            f_row = s_row.setdefault(field, {"hist": []})
            hist = list(f_row.get("hist") or [])
            hist.append(float(value))
            if len(hist) > int(window):
                hist = hist[-int(window):]
            f_row["hist"] = hist
            s_row[field] = f_row
            self._norm_cache[s] = s_row
        n = len(hist)
        meta = {"symbol": s, "field": field, "count": int(n), "method": "none"}
        if n < int(warmup):
            # Cold start safe relative transform for price-like features.
            if field in ("current_price", "entry_price") and fallback_entry > 0:
                out = float(value / fallback_entry - 1.0)
            else:
                out = float(value)
            meta["method"] = "warmup_relative" if field in ("current_price", "entry_price") else "warmup_raw"
            return out, meta
        arr = hist
        mn = min(arr) if arr else float(value)
        mx = max(arr) if arr else float(value)
        mean = sum(arr) / max(float(n), 1.0)
        var = sum((float(x) - mean) ** 2 for x in arr) / max(float(n), 1.0)
        std = var ** 0.5
        if std >= 1e-9:
            z = (float(value) - mean) / (std + 1e-9)
            out = float(min(5.0, max(-5.0, z)))
            meta.update({"method": "zscore", "mean": float(mean), "std": float(std)})
            return out, meta
        # Min-max fallback to [-1, 1].
        rng = max(float(mx - mn), 1e-9)
        mm = ((float(value) - float(mn)) / rng) * 2.0 - 1.0
        out = float(min(1.0, max(-1.0, mm)))
        meta.update({"method": "minmax", "min": float(mn), "max": float(mx)})
        return out, meta

    def _apply_symbol_normalization(
        self,
        symbol: str,
        x: List[float],
        entry: float,
        settings: dict,
    ) -> Tuple[List[float], dict]:
        # x indices:
        # 1=current_price,2=entry_price,10=volume,12=momentum,13=accel.
        if not self._flag_enabled(settings, "ai_tm_symbol_normalization_enabled", False):
            return list(x), {"enabled": False}
        out = list(x)
        window = min(1000, max(300, _safe_int(settings.get("ai_tm_norm_window"), 500)))
        warmup = min(200, max(50, _safe_int(settings.get("ai_tm_norm_warmup_min_obs"), 50)))
        idx_map = {
            1: "current_price",
            2: "entry_price",
            10: "volume",
            12: "momentum",
            13: "acceleration",
        }
        metas = []
        for idx, fname in idx_map.items():
            if idx >= len(out):
                continue
            norm_v, meta = self._rolling_norm(
                symbol=symbol,
                field=fname,
                value=_safe_float(out[idx], 0.0),
                window=window,
                warmup=warmup,
                fallback_entry=max(entry, 1e-12),
            )
            out[idx] = float(norm_v)
            metas.append(meta)
        self._flush_norm_cache(force=False)
        return out, {"enabled": True, "symbol": str(symbol).upper(), "window": int(window), "warmup": int(warmup), "fields": metas}

    def _symbol_cluster(self, symbol: str) -> str:
        s = str(symbol or "").upper()
        if s.endswith("USDT"):
            return "USDT"
        if s.endswith("BUSD"):
            return "BUSD"
        return "OTHER"

    def _confidence_bin(self, conf: float) -> str:
        c = min(0.999, max(0.0, float(conf)))
        b = int(c * 10.0)
        return f"{b/10.0:.1f}-{(b+1)/10.0:.1f}"

    def _calibrate_confidence(self, symbol: str, action: str, raw_conf: float, min_samples: int = 30) -> float:
        cluster = self._symbol_cluster(symbol)
        b = self._confidence_bin(raw_conf)
        act_map = self._conf_calibration.get(cluster) or {}
        row = (act_map.get(action) or {}).get(b) or {}
        total = _safe_float(row.get("total"), 0.0)
        wins = _safe_float(row.get("wins"), 0.0)
        if total < float(min_samples):
            return float(raw_conf)
        empirical = wins / max(total, 1.0)
        return min(0.999, max(0.0, float(empirical)))

    def _update_conf_calibration(self, symbol: str, action: str, conf: float, success: bool) -> None:
        cluster = self._symbol_cluster(symbol)
        b = self._confidence_bin(conf)
        act_map = self._conf_calibration.setdefault(cluster, {})
        bins = act_map.setdefault(str(action), {})
        row = bins.setdefault(b, {"wins": 0.0, "total": 0.0})
        row["total"] = _safe_float(row.get("total"), 0.0) + 1.0
        if bool(success):
            row["wins"] = _safe_float(row.get("wins"), 0.0) + 1.0

    def _blend_expert_weight(self, symbol: str, expert_name: str, global_w: float, settings: dict) -> float:
        if not self._flag_enabled(settings, "ai_tm_symbol_weight_blend_enabled", False):
            return float(global_w)
        alpha = min(1.0, max(0.0, _safe_float(settings.get("ai_tm_symbol_weight_alpha"), 0.7)))
        min_local = max(20, _safe_int(settings.get("ai_tm_symbol_weight_min_samples"), 200))
        perf = ((self._symbol_expert_perf.get(str(symbol).upper()) or {}).get(str(expert_name)) or {})
        local_samples = _safe_int(perf.get("total"), 0)
        local_rate = _safe_float(perf.get("success_rate"), 0.0)
        if local_samples < min_local:
            return float(global_w)
        local_w = max(1e-6, local_rate)
        return float(alpha * float(global_w) + (1.0 - alpha) * float(local_w))

    def _update_symbol_expert_perf(
        self,
        symbol: str,
        expert_votes: Dict[str, str],
        action_success: Dict[str, bool],
    ) -> None:
        s = str(symbol).upper()
        s_map = self._symbol_expert_perf.setdefault(s, {})
        for name, vote_act in (expert_votes or {}).items():
            ok = bool((action_success or {}).get(str(vote_act), False))
            row = s_map.setdefault(str(name), {"wins": 0.0, "total": 0.0, "success_rate": 0.0})
            row["total"] = _safe_float(row.get("total"), 0.0) + 1.0
            if ok:
                row["wins"] = _safe_float(row.get("wins"), 0.0) + 1.0
            row["success_rate"] = _safe_float(row.get("wins"), 0.0) / max(_safe_float(row.get("total"), 1.0), 1.0)

    def _trend_score(self, features: Dict[str, object], side: str, market_state: str) -> float:
        mom = _safe_float((features or {}).get("momentum"), 0.0)
        acc = _safe_float((features or {}).get("acceleration"), 0.0)
        rsi = _safe_float((features or {}).get("rsi"), 50.0)
        s = str(side).upper()
        score = 0.0
        if s in ("LONG", "BUY"):
            score += 0.45 if mom > 0 else -0.45
            score += 0.25 if acc > 0 else -0.25
            score += min(max((rsi - 50.0) / 30.0, -0.30), 0.30)
        else:
            score += 0.45 if mom < 0 else -0.45
            score += 0.25 if acc < 0 else -0.25
            score += min(max((50.0 - rsi) / 30.0, -0.30), 0.30)
        if str(market_state).upper() == "PASSIVE":
            score -= 0.10
        return min(1.0, max(-1.0, score))

    def _immediate_loss_risk(
        self,
        deep_dd_count: int,
        unreal: float,
        max_dd_cfg: float,
        has_missing_protection: bool,
    ) -> float:
        risk = 0.0
        if deep_dd_count >= 3:
            risk += 0.55
        if unreal <= -abs(max_dd_cfg):
            risk += 0.35
        if has_missing_protection:
            risk += 0.35
        return min(1.0, max(0.0, risk))

    def _giveback_risk(
        self,
        peak_unreal: float,
        dd_from_peak: float,
        market_state: str,
        momentum: float,
        side: str,
    ) -> float:
        score = 0.0
        if peak_unreal >= 1.0:
            score += min(max(dd_from_peak / 1.0, 0.0), 1.0) * 0.50
        if str(market_state).upper() == "PASSIVE":
            score += 0.20
        s = str(side).upper()
        adverse = (momentum < 0.0) if s in ("LONG", "BUY") else (momentum > 0.0)
        if adverse:
            score += 0.25
        return min(1.0, max(0.0, score))

    def _resolve_final_action(
        self,
        ml_action: str,
        ml_conf: float,
        override_meta: dict,
        risk_score: float,
        protection_score: float,
        margin: float = 0.15,
    ) -> Tuple[str, str]:
        final_action = str(ml_action or "HOLD")
        reason = "ml_primary"
        cand_action = str((override_meta or {}).get("candidate_action") or final_action)
        cand_reason = str((override_meta or {}).get("candidate_reason") or "")
        critical = bool((override_meta or {}).get("critical_risk"))
        override_score = min(1.0, max(0.0, max(float(risk_score), float(protection_score))))
        ml_score = min(1.0, max(0.0, float(ml_conf)))
        if critical and cand_action in _ACTIONS:
            return cand_action, cand_reason or "critical_override"
        if cand_action in _ACTIONS and (override_score - ml_score) >= float(margin):
            return cand_action, cand_reason or "override_margin_passed"
        return final_action, reason

    def _classify_mode(
        self,
        key: Tuple[int, str],
        settings: dict,
        market_state: str,
        exposure_ratio: float,
        dd_from_peak: float,
        trend_score: float,
        unreal: float,
    ) -> str:
        mode_enabled = self._flag_enabled(settings, "ai_tm_mode_policy_enabled", False)
        if not mode_enabled:
            return "conservative_mode"
        now = time.time()
        hold_sec = min(240.0, max(60.0, _safe_float(settings.get("ai_tm_mode_hold_sec"), 90.0)))
        with self._state_lock:
            ctx = self._ctx.get(key) or {}
            last_mode = str(ctx.get("policy_mode") or "conservative_mode")
            last_switch = _safe_float(ctx.get("policy_mode_switch_ts"), 0.0)
        to_conservative = (
            exposure_ratio >= _safe_float(settings.get("ai_tm_mode_exposure_conservative"), 0.72)
            or dd_from_peak >= _safe_float(settings.get("ai_tm_mode_dd_conservative"), 0.60)
            or (str(market_state).upper() == "PASSIVE" and trend_score < -0.10)
        )
        to_aggressive = (
            str(market_state).upper() == "ACTIVE"
            and trend_score >= _safe_float(settings.get("ai_tm_mode_trend_aggressive"), 0.20)
            and dd_from_peak < _safe_float(settings.get("ai_tm_mode_dd_aggressive"), 0.40)
            and unreal > -1.0
        )
        mode = last_mode
        if (now - last_switch) < hold_sec:
            return mode
        if to_conservative:
            mode = "conservative_mode"
        elif to_aggressive:
            mode = "aggressive_trend_mode"
        if mode != last_mode:
            with self._state_lock:
                mctx = self._ctx.get(key) or {}
                mctx["policy_mode"] = mode
                mctx["policy_mode_switch_ts"] = now
                self._ctx[key] = mctx
        return mode

    def _decide_position_priority(
        self,
        ctx: dict,
        market_state: str,
        trend_score: float,
        risk_score: float,
        ml_conf: float,
        volatility_pct: float,
        acc_idx: Optional[int] = None,
        symbol: Optional[str] = None,
        unreal_pnl_pct: float = 0.0,
    ) -> str:
        """Priority engine for SL-vs-recovery decisions."""
        dd_from_peak = _safe_float((ctx or {}).get("dd_from_peak"), 0.0)
        regime = self._detect_market_regime(volatility_pct=float(volatility_pct), trend_score=float(trend_score))
        th = self._adaptive_thresholds(regime)
        risk_th = float(th.get("risk_protect", 0.70))
        trend_th = float(th.get("trend_recover", 0.35))
        log(
            f"[AI-TM][ADAPTIVE] account={acc_idx} symbol={symbol} regime={regime} "
            f"risk_th={risk_th:.2f} trend_th={trend_th:.2f} vol_pct={float(volatility_pct):.3f}"
        )
        if float(unreal_pnl_pct) < -0.30:
            return "PROTECT"
        if float(risk_score) > risk_th:
            return "PROTECT"
        if float(dd_from_peak) > 0.70:
            return "PROTECT"
        if str(market_state or "").upper() == "PASSIVE" and float(risk_score) > min(risk_th, 0.60):
            return "PROTECT"
        if float(trend_score) > trend_th and float(ml_conf) > 0.60:
            return "RECOVER"
        return "BALANCED"

    def _is_recovery_mode(self, ctx: dict, unreal_pnl_pct: float, trend_score: float) -> bool:
        """Stronger recovery detection than pending flag/count-only checks."""
        if float(unreal_pnl_pct) >= -0.3:
            return False
        if float(trend_score) <= -0.35:
            return False
        return True

    def _policy_recovery_tp_pct(
        self,
        settings: dict,
        long_side: bool,
        model_conf: float,
        trend_score: float,
        mode: str,
    ) -> Tuple[float, str]:
        fallback = 0.50 if long_side else 0.20
        if not self._flag_enabled(settings, "ai_tm_ml_recovery_tp_enabled", False):
            return float(fallback), "deterministic_fallback"
        conf_min = _safe_float(settings.get("ai_tm_ml_recovery_conf_min"), 0.58)
        if model_conf < conf_min:
            return float(fallback), "deterministic_low_conf"
        base = _safe_float(settings.get("ai_tm_ml_recovery_base_pct"), fallback)
        trend_adj = min(max(float(trend_score) * 0.15, -0.20), 0.20)
        mode_adj = 0.10 if str(mode) == "aggressive_trend_mode" else -0.05
        use = float(base + trend_adj + mode_adj)
        min_pct = _safe_float(settings.get("ai_tm_ml_recovery_min_pct"), 0.05)
        max_pct = _safe_float(settings.get("ai_tm_ml_recovery_max_pct"), 0.90)
        use = min(max_pct, max(min_pct, use))
        return float(use), "ml_policy"

    def _compute_tranche_breakeven_tp(
        self,
        settings: dict,
        long_side: bool,
        tranche_entry: float,
        overall_entry: float,
        mode: str,
        model_conf: float,
        trend_score: float,
    ) -> dict:
        """Compute tranche TP once, with strict single-pass contract."""
        pipeline: List[dict] = []
        strict_mode = self._flag_enabled(settings, "ai_tm_ml_strict_mode", True)
        # Dynamic TP target for better risk/reward:
        # weak 0.05%, normal 0.08-0.12%, strong 0.15-0.25%.
        abs_trend = abs(float(trend_score))
        conf_v = float(max(0.0, min(1.0, model_conf)))
        if abs_trend >= 0.45 and conf_v >= 0.64:
            strength = "strong"
            base_offset_pct = 0.15 + min(0.10, max(0.0, abs_trend - 0.45) * 0.40 + max(0.0, conf_v - 0.64) * 0.10)
            base_offset_pct = min(0.25, max(0.15, base_offset_pct))
        elif abs_trend >= 0.20 and conf_v >= 0.58:
            strength = "normal"
            base_offset_pct = 0.08 + min(0.04, max(0.0, abs_trend - 0.20) * 0.10 + max(0.0, conf_v - 0.58) * 0.05)
            base_offset_pct = min(0.12, max(0.08, base_offset_pct))
        else:
            strength = "weak"
            base_offset_pct = 0.05
        pipeline.append(
            {
                "step_name": "base_offset",
                "input_price": None,
                "output_price": None,
                "delta_pct": float(base_offset_pct),
                "reason": f"dynamic_tp_strength={strength}",
            }
        )

        # Apply fee/funding buffer only once. Defaults are 0 to avoid hidden drift.
        taker_fee_rate = _safe_float(settings.get("ai_tm_scale_in_taker_fee_rate"), 0.0)
        taker_fee_rate = min(max(taker_fee_rate, 0.0), 0.003)
        # strict mode: include one-side fee once; non-strict keeps previous behavior.
        fee_buffer_pct = float(taker_fee_rate * (100.0 if strict_mode else 200.0))
        funding_buffer_pct = min(max(_safe_float(settings.get("ai_tm_scale_in_funding_buffer_pct"), 0.0), 0.0), 0.30)
        one_pass_buffer_pct = float(fee_buffer_pct + funding_buffer_pct)
        pipeline.append(
            {
                "step_name": "buffer_once",
                "input_price": None,
                "output_price": None,
                "delta_pct": float(one_pass_buffer_pct),
                "reason": "fee_plus_funding_once",
            }
        )

        total_offset_pct = float(base_offset_pct + one_pass_buffer_pct)
        total_offset_pct = max(0.0, total_offset_pct)

        # Strictly anchor around overall/main entry (or fallback to tranche entry)
        # and hard-cap deviation within ±0.5% by default.
        ref_entry = float(overall_entry) if float(overall_entry) > 0 else float(tranche_entry)
        if ref_entry <= 0:
            ref_entry = float(tranche_entry)
        # Side-specific strict cap:
        # LONG  <= +0.50% from overall entry
        # SHORT <= -0.20% from overall entry
        side_cap_pct = 0.50 if long_side else 0.20
        side_cap_cfg = _safe_float(
            settings.get("ai_tm_tranche_tp_long_max_dev_pct" if long_side else "ai_tm_tranche_tp_short_max_dev_pct"),
            side_cap_pct,
        )
        side_cap_pct = max(0.02, side_cap_cfg)
        global_cap = _safe_float(settings.get("ai_tm_tranche_tp_max_deviation_pct"), 0.50)
        max_dev_pct = min(side_cap_pct, min(0.50, max(0.02, global_cap)))
        # Guarantee requested defaults even when global cap is looser.
        if long_side:
            max_dev_pct = min(max_dev_pct, 0.50)
        else:
            max_dev_pct = min(max_dev_pct, 0.20)

        if long_side:
            target_px = float(ref_entry) * (1.0 + float(total_offset_pct) / 100.0)
        else:
            target_px = float(ref_entry) * (1.0 - float(total_offset_pct) / 100.0)
        pipeline.append(
            {
                "step_name": "raw_target",
                "input_price": float(ref_entry),
                "output_price": float(target_px),
                "delta_pct": float(abs(target_px - ref_entry) / max(abs(ref_entry), 1e-12) * 100.0),
                "reason": "base_plus_single_buffer",
            }
        )

        if long_side:
            min_px = float(ref_entry)
            max_px = float(ref_entry) * (1.0 + float(max_dev_pct) / 100.0)
        else:
            min_px = float(ref_entry) * (1.0 - float(max_dev_pct) / 100.0)
            max_px = float(ref_entry)
        target_px = min(max(target_px, min_px), max_px)
        pipeline.append(
            {
                "step_name": "clamp_bounds",
                "input_price": float(ref_entry),
                "output_price": float(target_px),
                "delta_pct": float(abs(target_px - ref_entry) / max(abs(ref_entry), 1e-12) * 100.0),
                "reason": (
                    "long_cap_entry_plus_0_5pct"
                    if long_side
                    else "short_cap_entry_minus_0_2pct"
                ),
            }
        )

        # Hard runtime assertion + force-correct boundary (must never be violated).
        allowed_limit_pct = float(max_dev_pct)
        actual_dev_pct = float(abs(target_px - ref_entry) / max(abs(ref_entry), 1e-12) * 100.0)
        if actual_dev_pct > (allowed_limit_pct + 1e-9):
            if long_side:
                target_px = float(ref_entry) * (1.0 + float(allowed_limit_pct) / 100.0)
            else:
                target_px = float(ref_entry) * (1.0 - float(allowed_limit_pct) / 100.0)
            log(
                f"[AI-TM][TP][CRITICAL] symbol_tp_deviation_violation corrected "
                f"entry={ref_entry:.8f} target={target_px:.8f} "
                f"actual_dev_pct={actual_dev_pct:.6f} allowed_pct={allowed_limit_pct:.6f}"
            )
            pipeline.append(
                {
                    "step_name": "hard_assert_corrected",
                    "input_price": float(ref_entry),
                    "output_price": float(target_px),
                    "delta_pct": float(abs(target_px - ref_entry) / max(abs(ref_entry), 1e-12) * 100.0),
                    "reason": "forced_to_allowed_boundary",
                }
            )

        return {
            "target_px": float(target_px),
            "tp_pct": float(total_offset_pct),
            "ref_entry": float(ref_entry),
            "max_dev_pct": float(max_dev_pct),
            "fee_buffer_pct": float(fee_buffer_pct),
            "funding_buffer_pct": float(funding_buffer_pct),
            "tiny_plus_pct": 0.0,
            "policy": "tranche_breakeven_tp_strict" if strict_mode else "tranche_breakeven_tp",
            "pipeline_trace": pipeline,
        }

    def _enforce_tranche_breakeven_lock(
        self,
        acc_idx: int,
        symbol: str,
        settings: dict,
        long_side: bool,
        entry: float,
        cur: float,
        position_side,
    ) -> Optional[dict]:
        """Promote SL to breakeven when tranche TP proximity is reached.

        Trigger:
          - current price is within configurable proximity window around
            tranche TP price (default ±2% of tranche entry).
        Action:
          - place/refresh protective STOP_MARKET near tranche entry with
            fee/funding-aware breakeven buffer.
          - mark tranche as breakeven_locked to avoid duplicate triggers.
        """
        if not self._flag_enabled(settings, "ai_tm_breakeven_lock_enabled", True):
            return None
        key = (int(acc_idx), str(symbol))
        ctx = self._ctx.get(key) or {}
        tranches = ctx.get("scale_in_tranches")
        if not isinstance(tranches, list) or not tranches:
            return None

        now = time.time()
        cd_sec = max(10.0, _safe_float(settings.get("ai_tm_breakeven_lock_cooldown_sec"), 30.0))
        last_lock_ts = _safe_float(ctx.get("last_breakeven_lock_ts"), 0.0)
        if (now - last_lock_ts) < cd_sec:
            return None

        proximity_pct = min(5.0, max(0.2, _safe_float(settings.get("ai_tm_breakeven_lock_proximity_pct"), 2.0)))
        taker_fee_rate = _safe_float(settings.get("ai_tm_scale_in_taker_fee_rate"), 0.0004)
        funding_buffer_pct = _safe_float(settings.get("ai_tm_scale_in_funding_buffer_pct"), 0.01)
        tiny_plus_pct = _safe_float(settings.get("ai_tm_breakeven_lock_tiny_plus_pct"), 0.01)
        be_lock_pct = min(
            0.50,
            max(
                0.01,
                float(taker_fee_rate * 200.0 + max(funding_buffer_pct, 0.0) + max(tiny_plus_pct, 0.0)),
            ),
        )

        close_side = "SELL" if long_side else "BUY"
        lock_candidates: List[float] = []
        locked_ids: List[str] = []
        updated = False
        for tr in tranches:
            if not isinstance(tr, dict):
                continue
            if bool(tr.get("breakeven_locked")):
                continue
            tp_px = _safe_float(tr.get("tp_price"), 0.0)
            tr_entry = _safe_float(tr.get("entry_price"), 0.0)
            if tp_px <= 0.0:
                continue
            if tr_entry <= 0.0:
                tr_entry = _safe_float(entry, 0.0)
            if tr_entry <= 0.0:
                continue
            proximity_abs = abs(float(tr_entry) * float(proximity_pct) / 100.0)
            if abs(float(cur) - float(tp_px)) > max(proximity_abs, 1e-12):
                continue

            if long_side:
                sl_px = float(tr_entry) * (1.0 + float(be_lock_pct) / 100.0)
                if cur > 0:
                    sl_px = min(sl_px, float(cur) * 0.9995)
            else:
                sl_px = float(tr_entry) * (1.0 - float(be_lock_pct) / 100.0)
                if cur > 0:
                    sl_px = max(sl_px, float(cur) * 1.0005)
            if sl_px <= 0.0:
                continue
            lock_candidates.append(float(sl_px))
            tid = str(tr.get("tranche_id") or "")
            if tid:
                locked_ids.append(tid)
            tr["breakeven_locked"] = True
            tr["breakeven_lock_ts"] = float(now)
            tr["protective_sl_price"] = float(sl_px)
            tr["tp_proximity_window_pct"] = float(proximity_pct)
            updated = True

        if not lock_candidates:
            return None
        protective_sl = max(lock_candidates) if long_side else min(lock_candidates)
        try:
            place_exchange_stop_order(
                int(acc_idx),
                str(symbol),
                close_side,
                float(protective_sl),
                order_type="STOP_MARKET",
                quantity=None,
                close_position=True,
                position_side=position_side,
                caller="AITM",
            )
            ctx["last_breakeven_lock_ts"] = float(now)
            ctx["last_breakeven_lock_price"] = float(protective_sl)
            if updated:
                ctx["scale_in_tranches"] = tranches
            self._ctx[key] = ctx
            return {
                "breakeven_locked": True,
                "protective_sl_price": float(protective_sl),
                "tranche_ids": locked_ids,
                "proximity_pct": float(proximity_pct),
                "be_lock_pct": float(be_lock_pct),
            }
        except Exception as e:
            return {
                "breakeven_locked": False,
                "protective_sl_price": float(protective_sl),
                "tranche_ids": locked_ids,
                "error": str(e),
            }

    def _register_decision_credit_event(
        self,
        key: Tuple[int, str],
        lifecycle_id: str,
        decision_id: str,
        decision_seq: int,
        symbol: str,
        account_index: int,
        ts: float,
        unreal_pnl_pct: float,
        ml_action: str,
        final_action: str,
        conf: float,
        features_vector: List[float],
        expert_votes: Dict[str, str],
    ) -> None:
        ev = {
            "position_lifecycle_id": str(lifecycle_id),
            "decision_id": str(decision_id),
            "decision_seq": int(decision_seq),
            "account_index": int(account_index),
            "symbol": str(symbol),
            "ts": float(ts),
            "unreal_pnl_pct_at_decision": float(unreal_pnl_pct),
            "ml_action": str(ml_action),
            "final_action": str(final_action),
            "confidence": float(conf),
            "features_vector": list(features_vector or []),
            "expert_votes": dict(expert_votes or {}),
            "horizons_sec": [30, 120, 300, 900],
            "logged_horizons": {},
        }
        with self._state_lock:
            arr = self._pending_credit.setdefault(key, [])
            arr.append(ev)

    def _flush_decision_credit(
        self,
        key: Tuple[int, str],
        symbol: str,
        cur_unreal: float,
        now_ts: float,
    ) -> None:
        with self._state_lock:
            arr = list(self._pending_credit.get(key) or [])
        if not arr:
            return
        still_pending: List[dict] = []
        for ev in arr:
            ts0 = _safe_float(ev.get("ts"), now_ts)
            u0 = _safe_float(ev.get("unreal_pnl_pct_at_decision"), 0.0)
            d_unreal = float(cur_unreal) - float(u0)
            logged = ev.get("logged_horizons") or {}
            all_done = True
            for h in [30, 120, 300, 900]:
                if bool(logged.get(str(h))):
                    continue
                if (now_ts - ts0) < float(h):
                    all_done = False
                    continue
                success = d_unreal >= 0.0
                _append_jsonl(
                    _CREDIT_LOG_PATH,
                    {
                        "event": "horizon_label",
                        "position_lifecycle_id": ev.get("position_lifecycle_id"),
                        "decision_id": ev.get("decision_id"),
                        "decision_seq": ev.get("decision_seq"),
                        "account_index": ev.get("account_index"),
                        "symbol": ev.get("symbol"),
                        "ts_decision": ts0,
                        "ts_label": float(now_ts),
                        "horizon_sec": int(h),
                        "x_t": ev.get("features_vector"),
                        "ml_action": ev.get("ml_action"),
                        "final_action": ev.get("final_action"),
                        "confidence": ev.get("confidence"),
                        "unreal_at_decision": float(u0),
                        "unreal_now": float(cur_unreal),
                        "delta_unreal_pct": float(d_unreal),
                        "label_success": bool(success),
                    },
                )
                self._update_conf_calibration(symbol, str(ev.get("final_action") or "HOLD"), _safe_float(ev.get("confidence"), 0.0), success)
                logged[str(h)] = True
            ev["logged_horizons"] = logged
            all_done = all_done and all(bool(logged.get(str(h))) for h in [30, 120, 300, 900])
            if not all_done:
                still_pending.append(ev)
            else:
                self._update_symbol_expert_perf(
                    symbol=symbol,
                    expert_votes=ev.get("expert_votes") or {},
                    action_success={str(ev.get("final_action") or "HOLD"): bool(d_unreal >= 0.0)},
                )
        with self._state_lock:
            if still_pending:
                self._pending_credit[key] = still_pending
            else:
                self._pending_credit.pop(key, None)

    def _get_symbol_lock(self, key: Tuple[int, str]) -> threading.Lock:
        with self._state_lock:
            lk = self._symbol_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._symbol_locks[key] = lk
            return lk

    def _build_eval_row_task(
        self,
        ai: int,
        p: dict,
        now_ts: float,
        prev_ctx: dict,
        eff_interval_sec: Optional[float],
        settings: Optional[dict] = None,
    ) -> Optional[dict]:
        symbol = str(p.get("symbol") or "")
        if not symbol:
            return None
        amt = _safe_float(p.get("position_amt"), 0.0)
        if amt == 0.0:
            return None
        side = "LONG" if amt > 0 else "SHORT"
        entry = _safe_float(p.get("entry_price"), 0.0)
        candles = fetch_candles(symbol, "1m", limit=120) or []
        if len(candles) < 20:
            return None
        features = build_features(candles) or {}
        try:
            last = candles[-1]
            cur = _safe_float(last.get("close"), 0.0)
        except Exception:
            cur = 0.0
        if entry <= 0 or cur <= 0:
            return None
        direction = 1.0 if amt > 0 else -1.0
        unreal = ((cur - entry) / entry) * 100.0 * direction

        ctx = dict(prev_ctx or {})
        if not isinstance(ctx, dict) or not ctx:
            ctx = {
                "entry_ts": now_ts,
                "entry_price": float(entry),
                "base_amt": float(abs(amt)),
                "scale_in_count": 0,
                "market_state": "ACTIVE",
                "peak_unreal": float("-inf"),
                "deep_dd_count": 0,
                "last_sl_move_ts": 0.0,
                "last_tp_move_ts": 0.0,
            }
        peak_missing_before = "peak_unreal" not in ctx
        if peak_missing_before:
            ctx["peak_unreal"] = float("-inf")
        if "deep_dd_count" not in ctx:
            ctx["deep_dd_count"] = 0
        if "last_sl_move_ts" not in ctx:
            ctx["last_sl_move_ts"] = 0.0
        if "last_tp_move_ts" not in ctx:
            ctx["last_tp_move_ts"] = 0.0
        ctx["last_ts"] = now_ts
        ctx["candles"] = candles
        ctx["features"] = features
        ctx["unreal_pnl_pct"] = float(unreal)
        ctx["peak_unreal"] = max(_safe_float(ctx.get("peak_unreal"), float("-inf")), float(unreal))
        dd_from_peak = _safe_float(ctx.get("peak_unreal"), float(unreal)) - float(unreal)
        if unreal <= -3.5:
            ctx["deep_dd_count"] = _safe_int(ctx.get("deep_dd_count"), 0) + 1
        else:
            ctx["deep_dd_count"] = 0
        atr = _safe_float(features.get("atr"), 0.0)
        atr_pct = abs(atr / entry) * 100.0 if entry > 0 and atr > 0 else 0.0
        market_state = "PASSIVE" if atr_pct < 0.45 else "ACTIVE"
        ctx["market_state"] = market_state
        data_ready_ts = time.time()

        x = _build_management_features(
            symbol=symbol,
            side=side,
            entry_price=entry,
            current_price=cur,
            unreal_pnl_pct=unreal,
            time_since_entry_sec=max(0.0, data_ready_ts - _safe_float(ctx.get("entry_ts"), data_ready_ts)),
            candles=candles,
            features=features,
            ctx=ctx,
        )
        x_raw = list(x)
        x, norm_meta = self._apply_symbol_normalization(
            symbol=symbol,
            x=x,
            entry=entry,
            settings=(settings or {}),
        )
        model_action, model_conf, model_meta = self._action_from_model(
            x,
            symbol=symbol,
            settings=(settings or {}),
        )
        return {
            "ai": int(ai),
            "symbol": symbol,
            "position": p,
            "amt": float(amt),
            "side": side,
            "entry": float(entry),
            "cur": float(cur),
            "unreal": float(unreal),
            "features": features,
            "momentum": _safe_float(features.get("momentum"), 0.0),
            "accel": _safe_float(features.get("acceleration"), 0.0),
            "market_state": market_state,
            "x": x,
            "x_raw": x_raw,
            "norm_meta": norm_meta,
            "model_action": model_action,
            "model_conf": float(model_conf),
            "model_raw_conf": float(_safe_float((model_meta or {}).get("raw_confidence"), 0.0)),
            "model_meta": model_meta or {},
            "dd_from_peak": float(dd_from_peak),
            "deep_dd_count": int(_safe_int(ctx.get("deep_dd_count"), 0)),
            "peak_unreal": float(_safe_float(ctx.get("peak_unreal"), float("-inf"))),
            "ctx_update": ctx,
            "peak_missing_before": bool(peak_missing_before),
            "effective_symbol_interval_sec": float(eff_interval_sec) if eff_interval_sec is not None else None,
            "data_ready_ts": float(data_ready_ts),
        }

    def _enabled_accounts(self) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        try:
            cfg = self._get_accounts_cfg_live()
        except Exception:
            cfg = {}
        if not isinstance(cfg, dict):
            return out
        accs = cfg.get("accounts") or []
        if not isinstance(accs, list):
            return out
        for i, a in enumerate(accs):
            if not isinstance(a, dict):
                continue
            # Respect account trading switch: disabled accounts are fully
            # ignored by AI-TM and never managed/closed by this module.
            if a.get("trade_enabled") is False:
                continue
            settings = a.get("settings") or {}
            if not isinstance(settings, dict):
                settings = {}
            if bool(settings.get("ai_dynamic_trade_management")):
                out[int(i)] = settings
        return out

    def _action_from_model(
        self,
        x: List[float],
        symbol: str = "",
        settings: Optional[dict] = None,
    ) -> Tuple[str, float, dict]:
        model = _load_model()
        if model is None:
            return "HOLD", 0.0, {"raw_confidence": 0.0, "expert_votes": {}}
        if np is None:
            return "HOLD", 0.0, {"raw_confidence": 0.0, "expert_votes": {}}
        try:
            arr = np.array([x], dtype=float)
            # New bundle format: {"experts": {name:model}, "weights": {...}, "actions":[...]}
            if isinstance(model, dict) and isinstance(model.get("experts"), dict):
                experts = model.get("experts") or {}
                weights = model.get("weights") or {}
                actions = list(model.get("actions") or list(_ACTIONS))
                act_to_idx = {a: i for i, a in enumerate(actions)}
                agg = np.zeros((len(actions),), dtype=float)
                used = 0
                expert_votes: Dict[str, str] = {}
                weight_used: Dict[str, float] = {}
                for name, expert in experts.items():
                    if expert is None:
                        continue
                    global_w = float(weights.get(name, 1.0))
                    w = self._blend_expert_weight(
                        symbol=str(symbol or ""),
                        expert_name=str(name),
                        global_w=global_w,
                        settings=(settings or {}),
                    )
                    if w <= 0:
                        continue
                    try:
                        probs = expert.predict_proba(arr)
                        row = probs[0]
                        cls = getattr(expert, "classes_", None)
                        if cls is None:
                            # Torch wrapper uses fixed action order
                            cls = actions
                        for i, c in enumerate(cls):
                            a = str(c)
                            j = act_to_idx.get(a)
                            if j is None:
                                continue
                            try:
                                agg[j] += w * float(row[i])
                            except Exception:
                                continue
                        try:
                            top_idx = int(np.argmax(row))
                            top_action = str(cls[top_idx])
                            if top_action in _ACTIONS:
                                expert_votes[str(name)] = top_action
                                weight_used[str(name)] = float(w)
                        except Exception:
                            pass
                        used += 1
                    except Exception:
                        continue
                if used > 0 and float(agg.sum()) > 0:
                    agg = agg / float(agg.sum())
                    idx = int(np.argmax(agg))
                    action = str(actions[idx]) if 0 <= idx < len(actions) else "HOLD"
                    raw_conf = float(agg[idx]) if 0 <= idx < len(actions) else 0.0
                    conf = self._calibrate_confidence(
                        symbol=str(symbol or ""),
                        action=action,
                        raw_conf=raw_conf,
                        min_samples=max(10, _safe_int((settings or {}).get("ai_tm_conf_calibration_min_samples"), 30)),
                    )
                    if action not in _ACTIONS:
                        action = "HOLD"
                    return action, conf, {
                        "raw_confidence": float(raw_conf),
                        "expert_votes": expert_votes,
                        "expert_weights_used": weight_used,
                    }

            # Legacy single-model fallback
            pred = model.predict(arr)
            action = str(pred[0]) if isinstance(pred, (list, tuple, np.ndarray)) else str(pred)
            raw_conf = 0.0
            try:
                probs = model.predict_proba(arr)
                row = probs[0]
                raw_conf = float(max(row)) if len(row) > 0 else 0.0
            except Exception:
                raw_conf = 0.0
            conf = self._calibrate_confidence(
                symbol=str(symbol or ""),
                action=action,
                raw_conf=raw_conf,
                min_samples=max(10, _safe_int((settings or {}).get("ai_tm_conf_calibration_min_samples"), 30)),
            )
            if action not in _ACTIONS:
                action = "HOLD"
            return action, conf, {"raw_confidence": float(raw_conf), "expert_votes": {}}
        except Exception:
            return "HOLD", 0.0, {"raw_confidence": 0.0, "expert_votes": {}}

    def _prefer_profit_protection_action(
        self,
        acc_idx: int,
        symbol: str,
        settings: dict,
        side: str,
        market_state: str,
        momentum: float,
        accel: float,
        model_action: str,
        model_conf: float,
        unreal_pnl_pct: float,
        entry_price: float,
        current_price: float,
    ) -> Tuple[str, float, Optional[str]]:
        """Soft-priority policy: in profit / near TP zones prefer SL tightening.

        This is intentionally advisory (not a hard force): we only override
        HOLD when conditions are favorable and cooldown permits.
        """
        action = str(model_action or "HOLD")
        conf = float(model_conf)
        reason: Optional[str] = None
        if action != "HOLD":
            return action, conf, reason

        try:
            tp_pcts_raw = settings.get("tp_pcts")
            tp_pcts = [float(x) for x in tp_pcts_raw] if isinstance(tp_pcts_raw, list) else []
        except Exception:
            tp_pcts = []
        tp1 = tp_pcts[0] if len(tp_pcts) >= 1 and tp_pcts[0] > 0 else 2.0
        tp2 = tp_pcts[1] if len(tp_pcts) >= 2 and tp_pcts[1] > 0 else max(tp1 * 1.6, tp1 + 1.0)

        # Profit zones where SL tightening is typically desirable.
        # Default is intentionally conservative (2%+), per user preference.
        min_profit_lock_pct = max(0.20, _safe_float(settings.get("ai_tm_min_profit_lock_pct"), 0.8))
        near_tp1 = unreal_pnl_pct >= max(0.25, 0.35 * tp1)
        near_tp2 = unreal_pnl_pct >= max(0.9, 0.80 * tp2)
        in_profit = unreal_pnl_pct > 0.0
        in_mild_profit = unreal_pnl_pct >= min_profit_lock_pct

        key = (int(acc_idx), str(symbol))
        ctx = self._ctx.get(key) or {}
        now = time.time()
        cd_sec = max(30.0, _safe_float(settings.get("ai_tm_profit_lock_cooldown_sec"), 60.0))
        last_adj = _safe_float(ctx.get("last_sl_adjust_ts"), 0.0)
        cooldown_ok = (now - last_adj) >= cd_sec
        # Do not micro-manage immediately after entry.
        min_age_sec = max(30.0, _safe_float(settings.get("ai_tm_profit_lock_min_age_sec"), 180.0))
        entry_ts = _safe_float(ctx.get("entry_ts"), now)
        age_ok = (now - entry_ts) >= min_age_sec
        peak_unreal = _safe_float(ctx.get("peak_unreal"), unreal_pnl_pct)
        giveback_pct = max(0.0, float(peak_unreal) - float(unreal_pnl_pct))

        s_up = str(side or "").upper()
        adverse_momentum = False
        if s_up in ("BUY", "LONG"):
            adverse_momentum = (momentum < 0.0 and accel < 0.0)
        else:
            adverse_momentum = (momentum > 0.0 and accel > 0.0)

        passive_state = str(market_state or "").upper() == "PASSIVE"
        risk_of_giveback = adverse_momentum or passive_state

        # Soft priority: only move from HOLD -> MOVE_SL when conditions suggest
        # locking gains/reducing give-back risk.
        should_lock = False
        if near_tp2 and in_profit:
            should_lock = True
        elif in_mild_profit and (near_tp1 or risk_of_giveback):
            should_lock = True
        elif in_profit and giveback_pct >= 0.40:
            should_lock = True

        # If model strongly prefers HOLD, respect it unless high-profit zone.
        hold_is_strong = conf >= 0.90
        if hold_is_strong and not near_tp2 and not near_tp1:
            should_lock = False

        if cooldown_ok and age_ok and in_profit and should_lock:
            action = "MOVE_SL"
            if near_tp2:
                conf = max(conf, 0.60)
            elif near_tp1:
                conf = max(conf, 0.55)
            else:
                conf = max(conf, 0.52)
            reason = "profit_lock_soft_priority"
        return action, conf, reason

    def _portfolio_close_advisory(
        self,
        acc_idx: int,
        settings: dict,
        rows: List[dict],
    ) -> Tuple[bool, float, str, dict]:
        """Advisory portfolio-level close idea (not a hard rule).

        The model can receive a strong hint to close all positions on an account
        when the *combined* portfolio context suggests banking gains is wiser
        than managing legs independently.
        """
        details = {
            "positions": 0,
            "portfolio_unreal_pct": 0.0,
            "breadth": 0.0,
            "passive_ratio": 0.0,
            "adverse_ratio": 0.0,
            "score": 0.0,
        }
        if not isinstance(rows, list) or len(rows) < 2:
            return False, 0.0, "not_enough_positions", details

        total_notional = 0.0
        total_pnl_usdt = 0.0
        winners = 0
        passive = 0
        adverse = 0
        for r in rows:
            try:
                amt = abs(float(r.get("amt") or 0.0))
                entry = float(r.get("entry") or 0.0)
                unreal = float(r.get("unreal") or 0.0)
                side = str(r.get("side") or "")
                momentum = float(r.get("momentum") or 0.0)
                accel = float(r.get("accel") or 0.0)
            except Exception:
                continue
            n = max(amt * entry, 0.0)
            total_notional += n
            total_pnl_usdt += n * (unreal / 100.0)
            if unreal > 0:
                winners += 1
            if str(r.get("market_state") or "").upper() == "PASSIVE":
                passive += 1
            if side == "LONG":
                if momentum < 0.0 and accel < 0.0:
                    adverse += 1
            elif side == "SHORT":
                if momentum > 0.0 and accel > 0.0:
                    adverse += 1

        n_pos = max(len(rows), 1)
        breadth = float(winners) / float(n_pos)
        passive_ratio = float(passive) / float(n_pos)
        adverse_ratio = float(adverse) / float(n_pos)
        portfolio_unreal_pct = (total_pnl_usdt / total_notional * 100.0) if total_notional > 0 else 0.0

        details.update(
            {
                "positions": int(n_pos),
                "portfolio_unreal_pct": float(portfolio_unreal_pct),
                "breadth": float(breadth),
                "passive_ratio": float(passive_ratio),
                "adverse_ratio": float(adverse_ratio),
            }
        )

        pctx = self._portfolio_ctx.get(int(acc_idx)) or {}
        now = time.time()
        cooldown_sec = max(120.0, _safe_float(settings.get("ai_tm_portfolio_hint_cooldown_sec"), 900.0))
        last_hint = _safe_float(pctx.get("last_close_hint_ts"), 0.0)
        if (now - last_hint) < cooldown_sec:
            return False, 0.0, "portfolio_hint_cooldown", details

        # Advisory score (continuous), not a rigid trigger.
        score = 0.0
        score += min(max(portfolio_unreal_pct, 0.0) / 6.0, 1.0) * 0.45
        score += min(max(breadth - 0.35, 0.0) / 0.65, 1.0) * 0.25
        score += max(passive_ratio, adverse_ratio) * 0.20
        score += (0.10 if n_pos >= 4 else 0.0)
        score = min(max(score, 0.0), 1.0)
        details["score"] = float(score)

        # Soft gating: only hint on positive combined context.
        suggest = bool(portfolio_unreal_pct > 0.0 and score >= 0.78)
        if suggest:
            pctx["last_close_hint_ts"] = now
            self._portfolio_ctx[int(acc_idx)] = pctx
            return True, float(score), "portfolio_green_close_hint", details
        return False, float(score), "portfolio_hold", details

    def _fetch_open_protection_orders(self, acc_idx: int, symbol: str) -> List[dict]:
        out: List[dict] = []
        try:
            clients = _get_clients()
            if not clients or acc_idx < 0 or acc_idx >= len(clients):
                return out
            client = clients[int(acc_idx)]
        except Exception:
            return out

        def _as_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            s = str(v or "").strip().lower()
            return s in ("1", "true", "yes", "y")

        def _norm_order(raw: dict, is_algo: bool) -> Optional[dict]:
            if not isinstance(raw, dict):
                return None
            try:
                order_type = str(raw.get("type") or raw.get("orderType") or "").upper()
                side = str(raw.get("side") or "").upper()
                close_pos = _as_bool(raw.get("closePosition"))
                qty = _safe_float(raw.get("origQty") if raw.get("origQty") is not None else raw.get("quantity"), 0.0)
                if qty <= 0.0:
                    qty = _safe_float(raw.get("qty"), 0.0)
                trig = _safe_float(
                    raw.get("stopPrice")
                    if raw.get("stopPrice") is not None
                    else (raw.get("triggerPrice") if raw.get("triggerPrice") is not None else raw.get("price")),
                    0.0,
                )
                oid = raw.get("algoId") if is_algo else raw.get("orderId")
            except Exception:
                return None
            return {
                "id": oid,
                "is_algo": bool(is_algo),
                "type": order_type,
                "side": side,
                "close_position": bool(close_pos),
                "qty": float(max(qty, 0.0)),
                "trigger_price": float(max(trig, 0.0)),
            }

        try:
            regular = client.open_orders(symbol=str(symbol).upper())
            if isinstance(regular, list):
                for r in regular:
                    nr = _norm_order(r, False)
                    if nr:
                        out.append(nr)
        except Exception:
            pass

        try:
            algo = client.get_algo_open_orders(symbol=str(symbol).upper())
            algo_list = algo if isinstance(algo, list) else (algo.get("orders") or [] if isinstance(algo, dict) else [])
            if isinstance(algo_list, list):
                for r in algo_list:
                    nr = _norm_order(r, True)
                    if nr:
                        out.append(nr)
        except Exception:
            pass
        return out

    def _fetch_open_orders_full(self, acc_idx: int, symbol: str) -> List[dict]:
        key = (int(acc_idx), str(symbol).upper())
        now = time.time()
        try:
            ttl = min(5.0, max(0.5, _safe_float(getattr(app_settings, "AI_TM_OPEN_ORDER_CACHE_TTL_SEC", 2.0), 2.0)))
        except Exception:
            ttl = 2.0
        cached = self._open_orders_cache.get(key)
        if isinstance(cached, dict):
            ts = _safe_float(cached.get("ts"), 0.0)
            if (now - ts) <= ttl:
                data = cached.get("data")
                if isinstance(data, list):
                    return list(data)
        out: List[dict] = []
        try:
            clients = _get_clients()
            if not clients or acc_idx < 0 or acc_idx >= len(clients):
                return out
            client = clients[int(acc_idx)]
        except Exception:
            return out

        def _as_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            s = str(v or "").strip().lower()
            return s in ("1", "true", "yes", "y")

        def _norm(raw: dict, is_algo: bool) -> Optional[dict]:
            if not isinstance(raw, dict):
                return None
            try:
                order_type = str(raw.get("type") or raw.get("orderType") or "").upper()
                side = str(raw.get("side") or "").upper()
                close_pos = _as_bool(raw.get("closePosition"))
                reduce_only = _as_bool(raw.get("reduceOnly"))
                qty = _safe_float(raw.get("origQty") if raw.get("origQty") is not None else raw.get("quantity"), 0.0)
                if qty <= 0.0:
                    qty = _safe_float(raw.get("qty"), 0.0)
                px = _safe_float(raw.get("price"), 0.0)
                trig = _safe_float(
                    raw.get("stopPrice")
                    if raw.get("stopPrice") is not None
                    else (raw.get("triggerPrice") if raw.get("triggerPrice") is not None else raw.get("price")),
                    0.0,
                )
                oid = raw.get("algoId") if is_algo else raw.get("orderId")
            except Exception:
                return None
            return {
                "id": oid,
                "is_algo": bool(is_algo),
                "type": order_type,
                "side": side,
                "close_position": bool(close_pos),
                "reduce_only": bool(reduce_only),
                "qty": float(max(qty, 0.0)),
                "price": float(max(px, 0.0)),
                "trigger_price": float(max(trig, 0.0)),
            }

        try:
            regular = client.open_orders(symbol=str(symbol).upper())
            if isinstance(regular, list):
                for r in regular:
                    nr = _norm(r, False)
                    if nr:
                        out.append(nr)
        except Exception:
            pass
        try:
            algo = client.get_algo_open_orders(symbol=str(symbol).upper())
            algo_list = algo if isinstance(algo, list) else (algo.get("orders") or [] if isinstance(algo, dict) else [])
            if isinstance(algo_list, list):
                for r in algo_list:
                    nr = _norm(r, True)
                    if nr:
                        out.append(nr)
        except Exception:
            pass
        self._open_orders_cache[key] = {"ts": float(now), "data": list(out)}
        return out

    def _enforce_sl_scale_conflict(
        self,
        acc_idx: int,
        symbol: str,
        settings: dict,
        long_side: bool,
        entry: float,
        position_side,
        market_state: str,
        trend_score: float,
        risk_score: float,
        ml_conf: float,
        unreal_pnl_pct: float,
        atr_pct: float,
    ) -> dict:
        out = {
            "sl_scale_conflict_detected": False,
            "conflict_resolution_action": "",
            "scale_orders_cancelled": 0,
            "sl_price": None,
            "scale_prices": [],
            "priority_mode": "",
            "sl_adjustment_reason": "",
            "scale_in_blocked": False,
            "volatility_buffer_pct": None,
        }
        if entry <= 0:
            return out
        orders = self._fetch_open_orders_full(int(acc_idx), str(symbol))
        if not orders:
            return out

        close_side = "SELL" if long_side else "BUY"
        open_side = "BUY" if long_side else "SELL"
        sl_types = {"STOP", "STOP_MARKET"}

        sl_orders: List[dict] = []
        scale_orders: List[dict] = []
        for o in orders:
            typ = str(o.get("type") or "").upper()
            side = str(o.get("side") or "").upper()
            if side == close_side and typ in sl_types:
                sl_orders.append(o)
                continue
            # SCALE_IN pending add-limits: same-side non-reduce-only LIMITs.
            if side == open_side and typ == "LIMIT" and (not bool(o.get("close_position"))) and (not bool(o.get("reduce_only"))):
                px = _safe_float(o.get("price"), 0.0)
                if px > 0:
                    if long_side and px < float(entry):
                        scale_orders.append(o)
                    if (not long_side) and px > float(entry):
                        scale_orders.append(o)

        if not sl_orders or not scale_orders:
            return out

        sl_prices = []
        for o in sl_orders:
            p = _safe_float(o.get("trigger_price"), 0.0)
            if p > 0:
                sl_prices.append(float(p))
        if not sl_prices:
            return out
        sl_price = max(sl_prices) if long_side else min(sl_prices)
        scale_prices = [float(_safe_float(o.get("price"), 0.0)) for o in scale_orders if _safe_float(o.get("price"), 0.0) > 0]
        if not scale_prices:
            return out

        # Conflict:
        # LONG  -> no scale-in may sit below SL
        # SHORT -> no scale-in may sit above SL
        conflict = any(px <= sl_price for px in scale_prices) if long_side else any(px >= sl_price for px in scale_prices)
        if not conflict:
            return out

        out["sl_scale_conflict_detected"] = True
        out["sl_price"] = float(sl_price)
        out["scale_prices"] = [float(x) for x in sorted(scale_prices)]

        key = (int(acc_idx), str(symbol))
        ctx = self._ctx.get(key) or {}
        in_recovery_mode = self._is_recovery_mode(ctx, unreal_pnl_pct=unreal_pnl_pct, trend_score=trend_score)
        priority = self._decide_position_priority(
            ctx=ctx,
            market_state=market_state,
            trend_score=trend_score,
            risk_score=risk_score,
            ml_conf=ml_conf,
            volatility_pct=atr_pct,
            acc_idx=int(acc_idx),
            symbol=str(symbol),
            unreal_pnl_pct=float(unreal_pnl_pct),
        )
        out["priority_mode"] = str(priority)

        vol_buffer_pct = max(float(atr_pct) * 0.5, 0.2)
        vol_buffer_pct = min(vol_buffer_pct, 5.0)
        out["volatility_buffer_pct"] = float(vol_buffer_pct)

        def _cancel_scale_orders(orders: List[dict]) -> int:
            c = 0
            for so in orders:
                oid = so.get("id")
                if oid is None:
                    continue
                try:
                    cancel_order_by_id(int(acc_idx), str(symbol), int(oid), is_algo=bool(so.get("is_algo")), caller="AITM")
                    c += 1
                except Exception:
                    continue
            return c

        if priority == "PROTECT":
            cancelled = _cancel_scale_orders(scale_orders)
            out["conflict_resolution_action"] = "cancel_scaleins"
            out["sl_adjustment_reason"] = "priority_protect"
            out["scale_orders_cancelled"] = int(cancelled)
            out["scale_in_blocked"] = True
        elif priority == "RECOVER" and in_recovery_mode:
            # Move SL below/above last scale level.
            floor = min(scale_prices) if long_side else max(scale_prices)
            if long_side:
                new_sl = float(floor) * (1.0 - float(vol_buffer_pct) / 100.0)
            else:
                new_sl = float(floor) * (1.0 + float(vol_buffer_pct) / 100.0)
            # cancel old sl orders first
            for so in sl_orders:
                oid = so.get("id")
                if oid is None:
                    continue
                try:
                    cancel_order_by_id(int(acc_idx), str(symbol), int(oid), is_algo=bool(so.get("is_algo")), caller="AITM")
                except Exception:
                    pass
            try:
                place_exchange_stop_order(
                    int(acc_idx),
                    str(symbol),
                    close_side,
                    float(new_sl),
                    order_type="STOP_MARKET",
                    quantity=None,
                    close_position=True,
                    position_side=position_side,
                    caller="AITM",
                )
                out["conflict_resolution_action"] = "move_sl"
                out["sl_adjustment_reason"] = "priority_recover_volatility_buffer"
            except Exception:
                out["conflict_resolution_action"] = "move_sl_failed"
                out["sl_adjustment_reason"] = "priority_recover_failed"
        else:
            # BALANCED: keep nearest scale-in only, cancel deeper ones, and slight SL shift.
            nearest = None
            if long_side:
                nearest = max(scale_orders, key=lambda o: _safe_float(o.get("price"), 0.0))
            else:
                nearest = min(scale_orders, key=lambda o: _safe_float(o.get("price"), 0.0))
            to_cancel = [o for o in scale_orders if o is not nearest]
            cancelled = _cancel_scale_orders(to_cancel)
            out["scale_orders_cancelled"] = int(cancelled)
            out["conflict_resolution_action"] = "balanced_keep_nearest_scalein"
            out["sl_adjustment_reason"] = "priority_balanced_partial_sl_move"

            floor = min(scale_prices) if long_side else max(scale_prices)
            slight_buffer_pct = max(0.1, float(vol_buffer_pct) * 0.5)
            if long_side:
                new_sl = float(floor) * (1.0 - float(slight_buffer_pct) / 100.0)
            else:
                new_sl = float(floor) * (1.0 + float(slight_buffer_pct) / 100.0)
            try:
                for so in sl_orders:
                    oid = so.get("id")
                    if oid is None:
                        continue
                    try:
                        cancel_order_by_id(int(acc_idx), str(symbol), int(oid), is_algo=bool(so.get("is_algo")), caller="AITM")
                    except Exception:
                        pass
                place_exchange_stop_order(
                    int(acc_idx),
                    str(symbol),
                    close_side,
                    float(new_sl),
                    order_type="STOP_MARKET",
                    quantity=None,
                    close_position=True,
                    position_side=position_side,
                    caller="AITM",
                )
                out["conflict_resolution_action"] = "balanced_keep_nearest_scalein_move_sl"
            except Exception:
                pass

        log(
            f"[AI-TM][PRIORITY] symbol={symbol} account={acc_idx} "
            f"priority={priority} risk={risk_score:.3f} trend={trend_score:.3f} "
            f"ml_conf={ml_conf:.3f} action={out.get('conflict_resolution_action')}"
        )
        log(
            f"[AI-TM][CONFLICT][SL_vs_SCALE] symbol={symbol} account={acc_idx} "
            f"sl_price={out.get('sl_price')} scale_prices={out.get('scale_prices')} "
            f"action={out.get('conflict_resolution_action')} cancelled={out.get('scale_orders_cancelled')}"
        )
        return out

    def _maintain_scale_in_recovery(
        self,
        acc_idx: int,
        symbol: str,
        settings: dict,
        long_side: bool,
        entry: float,
        cur: float,
        amt_abs: float,
        position_side,
    ) -> Optional[dict]:
        """Keep trying to enforce scale-in recovery leg exit.

        If recovery TP placement failed during SCALE_IN, we persist a pending
        recovery plan in ctx and retry placement. When price reaches the
        recovery target and TP is still missing, close only the added quantity
        at market (fallback) and keep base position alive.
        """
        key = (int(acc_idx), str(symbol))
        ctx = self._ctx.get(key) or {}
        rec = ctx.get("pending_scale_recovery")
        if not isinstance(rec, dict):
            return None

        target_px = _safe_float(rec.get("target_px"), 0.0)
        pending_qty = max(_safe_float(rec.get("qty"), 0.0), 0.0)
        strict_mode = self._flag_enabled(settings, "ai_tm_ml_strict_mode", True)
        if (not strict_mode) and target_px > 0.0 and entry > 0.0:
            be = self._compute_tranche_breakeven_tp(
                settings=settings,
                long_side=bool(long_side),
                tranche_entry=_safe_float(rec.get("tranche_entry_price"), entry),
                overall_entry=float(entry),
                mode=str((self._ctx.get(key) or {}).get("mode") or "conservative_mode"),
                model_conf=0.0,
                trend_score=0.0,
            )
            clamped_target = _safe_float(be.get("target_px"), target_px)
            if clamped_target > 0.0 and abs(clamped_target - target_px) > 1e-12:
                target_px = float(clamped_target)
                rec["target_px"] = float(clamped_target)
                ctx["pending_scale_recovery"] = rec
                self._ctx[key] = ctx
        base_amt = max(_safe_float(ctx.get("base_amt"), amt_abs), 0.0)
        remaining_extra = max(float(amt_abs) - float(base_amt), 0.0)
        if pending_qty <= 0.0 or target_px <= 0.0:
            ctx.pop("pending_scale_recovery", None)
            self._ctx[key] = ctx
            return {
                "status": "cleared_invalid_pending",
                "reason": "bad_pending_payload",
                "target_px": float(target_px),
                "qty": float(pending_qty),
                "skip_decision": False,
            }

        # If added leg is already flattened by exchange TP fills, clear pending.
        if remaining_extra <= max(1e-12, pending_qty * 0.02):
            ctx.pop("pending_scale_recovery", None)
            ctx["scale_in_count"] = max(0, _safe_int(ctx.get("scale_in_count"), 0) - 1)
            self._ctx[key] = ctx
            return {
                "status": "recovery_already_flattened",
                "reason": "extra_qty_gone",
                "target_px": float(target_px),
                "qty": float(pending_qty),
                "remaining_extra": float(remaining_extra),
                "skip_decision": False,
            }

        now = time.time()
        retry_sec = max(5.0, _safe_float(settings.get("ai_tm_scale_in_recovery_retry_sec"), 15.0))
        last_try_ts = _safe_float(rec.get("last_try_ts"), 0.0)
        close_side = "SELL" if long_side else "BUY"
        qty_to_place = max(min(float(pending_qty), float(remaining_extra)), 0.0)

        if qty_to_place > 0.0 and (now - last_try_ts) >= retry_sec:
            try:
                tp_place_started = time.time()
                place_exchange_stop_order(
                    int(acc_idx),
                    str(symbol),
                    close_side,
                    float(target_px),
                    order_type="TAKE_PROFIT_MARKET",
                    quantity=float(qty_to_place),
                    close_position=False,
                    position_side=position_side,
                    caller="AITM",
                )
                tp_place_elapsed = time.time() - tp_place_started
                self._emit_latency_warnings(
                    action="recovery_tp_retry",
                    acc_idx=int(acc_idx),
                    symbol=str(symbol),
                    elapsed_sec=float(tp_place_elapsed),
                    settings=settings,
                )
                ctx.pop("pending_scale_recovery", None)
                self._ctx[key] = ctx
                return {
                    "status": "recovery_tp_placed_retry",
                    "reason": "retry_success",
                    "target_px": float(target_px),
                    "qty": float(qty_to_place),
                    "skip_decision": False,
                }
            except Exception as e:
                rec["last_try_ts"] = float(now)
                rec["attempts"] = _safe_int(rec.get("attempts"), 0) + 1
                rec["last_error"] = str(e)
                ctx["pending_scale_recovery"] = rec
                self._ctx[key] = ctx

        # If we still don't have TP placed, wait for price to reach target and
        # then force-close ONLY the added chunk.
        reached = (cur >= target_px) if long_side else (cur <= target_px)
        if not reached:
            return None

        qty_to_close = max(min(float(pending_qty), float(remaining_extra)), 0.0)
        if qty_to_close <= 0.0:
            return None

        try:
            signed_amt = float(qty_to_close) if long_side else -float(qty_to_close)
            ok = close_position_market(
                int(acc_idx),
                str(symbol),
                signed_amt,
                force_full_close=False,
                position_side=position_side,
                caller="AITM",
            )
            if ok:
                ctx.pop("pending_scale_recovery", None)
                ctx["scale_in_count"] = max(0, _safe_int(ctx.get("scale_in_count"), 0) - 1)
                self._ctx[key] = ctx
                return {
                    "status": "recovery_closed_market_fallback",
                    "reason": "tp_not_placed_target_reached",
                    "target_px": float(target_px),
                    "qty": float(qty_to_close),
                    "skip_decision": True,
                }
            return {
                "status": "recovery_market_close_failed",
                "reason": "target_reached_but_close_failed",
                "target_px": float(target_px),
                "qty": float(qty_to_close),
                "skip_decision": False,
            }
        except Exception as e:
            return {
                "status": "recovery_market_close_error",
                "reason": "target_reached_close_exception",
                "target_px": float(target_px),
                "qty": float(qty_to_close),
                "error": str(e),
                "skip_decision": False,
            }

    def _cleanup_orphan_orders_for_account(
        self,
        acc_idx: int,
        active_symbols: set,
        settings: dict,
    ) -> Optional[dict]:
        if not self._flag_enabled(settings, "ai_tm_orphan_cleanup_enabled", True):
            return None
        now = time.time()
        pc = self._portfolio_ctx.get(int(acc_idx)) or {}
        cd_sec = max(10.0, _safe_float(settings.get("ai_tm_orphan_cleanup_cooldown_sec"), 20.0))
        if (now - _safe_float(pc.get("last_orphan_cleanup_ts"), 0.0)) < cd_sec:
            return None
        pc["last_orphan_cleanup_ts"] = float(now)
        self._portfolio_ctx[int(acc_idx)] = pc

        clients = _get_clients() or []
        if int(acc_idx) < 0 or int(acc_idx) >= len(clients):
            return None
        client = clients[int(acc_idx)]
        active_set = {str(s).upper() for s in (active_symbols or set()) if str(s)}

        raw_orders = []
        try:
            regular = client.open_orders()
            if isinstance(regular, list):
                for o in regular:
                    if isinstance(o, dict):
                        x = dict(o)
                        x["_is_algo"] = False
                        raw_orders.append(x)
        except Exception:
            pass
        try:
            algo = client.get_algo_open_orders()
            algo_list = algo if isinstance(algo, list) else (algo.get("orders") or [] if isinstance(algo, dict) else [])
            if isinstance(algo_list, list):
                for o in algo_list:
                    if isinstance(o, dict):
                        x = dict(o)
                        x["_is_algo"] = True
                        raw_orders.append(x)
        except Exception:
            pass

        if not raw_orders:
            return {"checked": 0, "cancelled_count": 0, "cancelled": []}

        cancelled = []
        for o in raw_orders:
            sym = str(o.get("symbol") or "").upper()
            if not sym or sym in active_set:
                continue
            oid = o.get("algoId") if bool(o.get("_is_algo")) else o.get("orderId")
            if oid is None:
                continue
            lock = self._get_symbol_lock((int(acc_idx), str(sym)))
            with lock:
                t0 = time.time()
                ok = False
                err = None
                # Run cancel in daemon thread with 8s timeout so a slow/no-proxy
                # account never blocks the AITM loop (was causing 85s hangs).
                import threading as _thr_oc
                _oc_result: list = [None]
                _oc_done = _thr_oc.Event()
                _oc_acc = int(acc_idx)
                _oc_sym = str(sym)
                _oc_oid = int(oid)
                _oc_algo = bool(o.get("_is_algo"))
                def _oc_worker(_a=_oc_acc, _s=_oc_sym, _o=_oc_oid, _al=_oc_algo):
                    try:
                        cancel_order_by_id(_a, _s, _o, is_algo=_al, caller="AITM")
                        _oc_result[0] = True
                    except Exception as _e:
                        _oc_result[0] = str(_e)
                    finally:
                        _oc_done.set()
                _thr_oc.Thread(target=_oc_worker, daemon=True).start()
                _oc_done.wait(timeout=8.0)
                if _oc_done.is_set():
                    if _oc_result[0] is True:
                        ok = True
                    else:
                        err = str(_oc_result[0])
                else:
                    err = "timeout_8s"
                elapsed = time.time() - t0
                rec = {
                    "account_index": int(acc_idx),
                    "symbol": str(sym),
                    "order_id": int(oid),
                    "is_algo": bool(o.get("_is_algo")),
                    "status": "cancelled" if ok else "error",
                    "error": err,
                    "latency_sec": float(elapsed),
                    "reason": "no_open_position",
                }
                cancelled.append(rec)
                self._emit_latency_warnings(
                    action="orphan_cancel",
                    acc_idx=int(acc_idx),
                    symbol=str(sym),
                    elapsed_sec=float(elapsed),
                    settings=settings,
                )
                _append_jsonl(
                    _LIFECYCLE_LOG_PATH,
                    {
                        "event": "orphan_order_cleanup",
                        "timestamp": int(time.time()),
                        **rec,
                    },
                )
        if cancelled:
            self._metrics["orphan_cancel_count"] = _safe_float(self._metrics.get("orphan_cancel_count"), 0.0) + float(
                len([x for x in cancelled if x.get("status") == "cancelled"])
            )
        return {"checked": len(raw_orders), "cancelled_count": len(cancelled), "cancelled": cancelled}

    def _order_consistency_advisory(
        self,
        acc_idx: int,
        symbol: str,
        settings: dict,
        side: str,
        entry: float,
        current_price: float,
        amt_abs: float,
    ) -> dict:
        details = {
            "suggest": False,
            "score": 0.0,
            "reason": "ok",
            "issues": [],
            "expected_tp_legs": 0,
            "found_tp_legs": 0,
            "found_sl_legs": 0,
            "expected_tp_total_qty": 0.0,
            "found_tp_total_qty": 0.0,
            "position_qty": float(max(amt_abs, 0.0)),
        }
        if amt_abs <= 0.0 or entry <= 0.0:
            return details

        key = (int(acc_idx), str(symbol))
        ctx = self._ctx.get(key) or {}
        now = time.time()
        cd_sec = max(60.0, _safe_float(settings.get("ai_tm_order_reconcile_cooldown_sec"), 180.0))
        last_fix = _safe_float(ctx.get("last_order_reconcile_ts"), 0.0)
        if (now - last_fix) < cd_sec:
            details["reason"] = "cooldown"
            return details

        close_side = "SELL" if str(side).upper() in ("LONG", "BUY") else "BUY"
        tp_types = {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}
        sl_types = {"STOP", "STOP_MARKET"}

        orders = self._fetch_open_protection_orders(int(acc_idx), str(symbol))
        tp_orders = [o for o in orders if str(o.get("type") or "") in tp_types and str(o.get("side") or "") == close_side]
        sl_orders = [o for o in orders if str(o.get("type") or "") in sl_types and str(o.get("side") or "") == close_side]

        try:
            tp_mode = _safe_int(settings.get("tp_mode"), 2)
            if tp_mode < 1:
                tp_mode = 1
            if tp_mode > 3:
                tp_mode = 3
            tp_raw = settings.get("tp_pcts")
            tp_pcts = [float(v) for v in tp_raw] if isinstance(tp_raw, list) else []
            tp_pcts = [v for v in tp_pcts if v > 0]
        except Exception:
            tp_mode = 2
            tp_pcts = []
        if not tp_pcts:
            tp_pcts = [1.0, 2.0, 3.0]
        expected_tp_legs = max(1, min(tp_mode, len(tp_pcts)))
        base_amt = max(_safe_float(ctx.get("base_amt"), amt_abs), 0.0)
        extra_qty = max(float(amt_abs) - float(base_amt), 0.0)
        allow_recovery_leg = bool(
            _safe_int(ctx.get("scale_in_count"), 0) > 0
            or extra_qty > 0.0
        )

        expected_tp_total_qty = float(amt_abs)
        found_tp_total_qty = float(sum(max(_safe_float(o.get("qty"), 0.0), 0.0) for o in tp_orders))
        details["expected_tp_legs"] = int(expected_tp_legs)
        details["found_tp_legs"] = int(len(tp_orders))
        details["found_sl_legs"] = int(len(sl_orders))
        details["expected_tp_total_qty"] = float(expected_tp_total_qty)
        details["found_tp_total_qty"] = float(found_tp_total_qty)

        issues: List[str] = []
        score = 0.0

        # If scale-in recovery TP is present, allow one extra TP leg.
        max_tp_legs = int(expected_tp_legs + (1 if allow_recovery_leg else 0))
        if len(tp_orders) < expected_tp_legs or len(tp_orders) > max_tp_legs:
            issues.append("tp_legs_mismatch")
            score += 0.35

        sl_ok = False
        for so in sl_orders:
            if bool(so.get("close_position")):
                sl_ok = True
                break
            q = _safe_float(so.get("qty"), 0.0)
            if q >= max(0.0, amt_abs * 0.85):
                sl_ok = True
                break
        if not sl_ok:
            issues.append("sl_not_covering_position")
            score += 0.35

        if expected_tp_total_qty > 0.0:
            ratio = found_tp_total_qty / expected_tp_total_qty if expected_tp_total_qty > 0 else 0.0
            upper_ratio = 1.55 if allow_recovery_leg else 1.25
            if ratio > upper_ratio or ratio < 0.60:
                issues.append("tp_qty_mismatch")
                score += 0.35

        # Duplicate TP trigger prices are a critical symptom of stale rebuild logic.
        tp_prices = []
        for tpo in tp_orders:
            px = _safe_float(tpo.get("trigger_price"), 0.0)
            if px > 0:
                tp_prices.append(round(px, 8))
        if len(tp_prices) >= 2 and len(set(tp_prices)) < len(tp_prices):
            issues.append("tp_duplicate_price")
            score += 0.45

        # If all TP prices are already on the wrong side of current market, they are stale.
        if tp_orders and current_price > 0:
            wrong_side = 0
            for tpo in tp_orders:
                px = _safe_float(tpo.get("trigger_price"), 0.0)
                if px <= 0:
                    continue
                if close_side == "SELL" and px <= current_price:
                    wrong_side += 1
                elif close_side == "BUY" and px >= current_price:
                    wrong_side += 1
            if wrong_side == len(tp_orders):
                issues.append("tp_all_stale_vs_price")
                score += 0.25

        score = min(max(score, 0.0), 1.0)
        details["issues"] = issues
        details["score"] = float(score)
        if issues and score >= 0.45:
            details["suggest"] = True
            details["reason"] = "order_consistency_mismatch"
        return details

    def _order_consistency_advisory_cached(
        self,
        acc_idx: int,
        symbol: str,
        settings: dict,
        side: str,
        entry: float,
        current_price: float,
        amt_abs: float,
    ) -> dict:
        """Short TTL cache to reduce repeated advisory I/O in fast loop."""
        k = (int(acc_idx), str(symbol).upper())
        ttl = min(5.0, max(0.5, _safe_float(settings.get("ai_tm_order_advisory_cache_ttl_sec"), 2.0)))
        now = time.time()
        cached = self._order_advisory_cache.get(k)
        if isinstance(cached, dict):
            ts = _safe_float(cached.get("ts"), 0.0)
            if (now - ts) <= ttl:
                data = cached.get("data")
                if isinstance(data, dict):
                    return dict(data)
        data = self._order_consistency_advisory(
            acc_idx=acc_idx,
            symbol=symbol,
            settings=settings,
            side=side,
            entry=entry,
            current_price=current_price,
            amt_abs=amt_abs,
        )
        self._order_advisory_cache[k] = {"ts": float(now), "data": dict(data or {})}
        return dict(data or {})

    def _enforce_minute_protection_guard(
        self,
        acc_idx: int,
        symbol: str,
        settings: dict,
        side: str,
        entry: float,
        cur: float,
        amt_abs: float,
        position_side,
        atr: float,
    ) -> Optional[dict]:
        """Every-minute guard:
        1) ensure protective SL exists
        2) if qty > base_qty, ensure add-only TP near base entry exists.
        """
        key = (int(acc_idx), str(symbol))
        ctx = self._ctx.get(key) or {}
        now = time.time()
        long_side = str(side).upper() in ("LONG", "BUY")
        close_side = "SELL" if long_side else "BUY"
        base_amt = max(_safe_float(ctx.get("base_amt"), amt_abs), 0.0)
        base_entry = _safe_float(ctx.get("entry_price"), 0.0)
        if base_entry <= 0.0:
            base_entry = _safe_float(entry, 0.0)
        extra_qty = max(float(amt_abs) - float(base_amt), 0.0)

        tp_types = {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}
        sl_types = {"STOP", "STOP_MARKET"}
        orders = self._fetch_open_protection_orders(int(acc_idx), str(symbol))
        sl_orders = [o for o in orders if str(o.get("type") or "") in sl_types and str(o.get("side") or "") == close_side]
        tp_orders = [o for o in orders if str(o.get("type") or "") in tp_types and str(o.get("side") or "") == close_side]
        guard_cd = max(5.0, _safe_float(settings.get("ai_tm_minute_guard_cooldown_sec"), 10.0))
        has_sl_now = bool(sl_orders)
        has_tp_now = bool(tp_orders)
        # Cooldown applies only when both protections already exist.
        # If either SL or TP is missing, enforce immediately.
        if has_sl_now and has_tp_now and (now - _safe_float(ctx.get("last_minute_guard_ts"), 0.0)) < guard_cd:
            return None

        out = {"sl_placed": False, "tp_placed": False, "recovery_tp_placed": False, "extra_qty": float(extra_qty)}

        # --- early profit protection ---
        # Once small profit zone is reached, move SL to entry (breakeven lock).
        try:
            unreal_now = ((float(cur) - float(entry)) / max(float(entry), 1e-12) * 100.0) * (1.0 if long_side else -1.0)
        except Exception:
            unreal_now = 0.0
        be_lock_min_profit = max(0.05, _safe_float(settings.get("ai_tm_early_be_lock_profit_pct"), 0.20))
        if entry > 0 and cur > 0 and unreal_now >= be_lock_min_profit:
            sl_be_ok = False
            for so in sl_orders:
                sp = _safe_float(so.get("trigger_price"), 0.0)
                if sp <= 0:
                    continue
                if long_side and sp >= (entry * 0.9995):
                    sl_be_ok = True
                    break
                if (not long_side) and sp <= (entry * 1.0005):
                    sl_be_ok = True
                    break
            if not sl_be_ok:
                try:
                    be_sl_px = float(entry)
                    place_exchange_stop_order(
                        int(acc_idx),
                        str(symbol),
                        close_side,
                        be_sl_px,
                        order_type="STOP_MARKET",
                        quantity=None,
                        close_position=True,
                        position_side=position_side,
                        caller="AITM",
                    )
                    out["sl_placed"] = True
                    out["breakeven_lock"] = True
                except Exception:
                    pass

        # Partial TP rule: once TP1 (+0.05%) zone is touched, move SL to entry.
        tranches = ctx.get("scale_in_tranches")
        if isinstance(tranches, list) and entry > 0 and cur > 0:
            changed = False
            for tr in tranches:
                if not isinstance(tr, dict):
                    continue
                if not bool(tr.get("partial_tp_enabled")):
                    continue
                if bool(tr.get("be_after_tp1")):
                    continue
                tp1_px = _safe_float(tr.get("tp1_price"), 0.0)
                if tp1_px <= 0:
                    continue
                tp1_reached = (cur >= tp1_px) if long_side else (cur <= tp1_px)
                if not tp1_reached:
                    continue
                try:
                    place_exchange_stop_order(
                        int(acc_idx),
                        str(symbol),
                        close_side,
                        float(entry),
                        order_type="STOP_MARKET",
                        quantity=None,
                        close_position=True,
                        position_side=position_side,
                        caller="AITM",
                    )
                    tr["be_after_tp1"] = True
                    tr["protective_sl_price"] = float(entry)
                    changed = True
                    out["sl_placed"] = True
                    out["breakeven_lock"] = True
                except Exception:
                    continue
            if changed:
                ctx["scale_in_tranches"] = tranches
                self._ctx[key] = ctx

        # --- SL guard ---
        sl_ok = False
        for so in sl_orders:
            if bool(so.get("close_position")):
                sl_ok = True
                break
            q = _safe_float(so.get("qty"), 0.0)
            if q >= max(amt_abs * 0.85, 0.0):
                sl_ok = True
                break
        if not sl_ok and amt_abs > 0 and entry > 0:
            try:
                sl_pct = _safe_float(settings.get("sl_pct"), 0.0)
                if sl_pct <= 0.0:
                    sl_pct = max(1.2, min(4.5, abs((atr / max(entry, 1e-12)) * 100.0) * 1.4))
                if long_side:
                    sl_px = entry * (1.0 - sl_pct / 100.0)
                else:
                    sl_px = entry * (1.0 + sl_pct / 100.0)
                place_exchange_stop_order(
                    int(acc_idx),
                    str(symbol),
                    close_side,
                    float(sl_px),
                    order_type="STOP_MARKET",
                    quantity=None,
                    close_position=True,
                    position_side=position_side,
                    caller="AITM",
                )
                out["sl_placed"] = True
            except Exception:
                pass

        # --- base TP guard ---
        # Always keep at least one TP when position is open, even without add-qty.
        if amt_abs > 0.0 and entry > 0.0 and (not tp_orders):
            try:
                tp_raw = settings.get("tp_pcts")
                tp_pcts = [float(v) for v in tp_raw] if isinstance(tp_raw, list) else []
                tp_pcts = [v for v in tp_pcts if v > 0]
            except Exception:
                tp_pcts = []
            tp_pct = float(tp_pcts[0]) if tp_pcts else 1.0
            if long_side:
                base_tp_px = entry * (1.0 + tp_pct / 100.0)
            else:
                base_tp_px = entry * (1.0 - tp_pct / 100.0)
            try:
                info = place_exchange_stop_order(
                    int(acc_idx),
                    str(symbol),
                    close_side,
                    float(base_tp_px),
                    order_type="TAKE_PROFIT_MARKET",
                    quantity=float(amt_abs),
                    close_position=False,
                    position_side=position_side,
                    caller="AITM",
                )
                if info:
                    out["tp_placed"] = True
                    out["tp_price"] = float(base_tp_px)
                    out["tp_qty"] = float(amt_abs)
                    out["tp_pct"] = float(tp_pct)
                else:
                    # Exchange can reject TP_MARKET as "immediate trigger" when
                    # target is already reached. In that case, close now.
                    reached = (float(cur) >= float(base_tp_px)) if long_side else (float(cur) <= float(base_tp_px))
                    if reached:
                        signed_amt = float(amt_abs) if long_side else -float(amt_abs)
                        ok_close = close_position_market(
                            int(acc_idx),
                            str(symbol),
                            signed_amt,
                            force_full_close=False,
                            position_side=position_side,
                            caller="AITM",
                        )
                        if ok_close:
                            out["tp_market_fallback"] = True
                            out["tp_price"] = float(base_tp_px)
                            out["tp_qty"] = float(amt_abs)
                            out["tp_pct"] = float(tp_pct)
            except Exception:
                pass

        # --- add-only recovery TP guard ---
        # If current size is larger than base position, keep an add-only TP near base entry.
        if extra_qty > 0.0 and base_entry > 0.0 and self._flag_enabled(settings, "ai_tm_tranche_tp_enabled", True):
            try:
                be = self._compute_tranche_breakeven_tp(
                    settings=settings,
                    long_side=bool(long_side),
                    tranche_entry=float(base_entry),
                    overall_entry=float(base_entry),
                    mode=str(ctx.get("mode") or "conservative_mode"),
                    model_conf=0.0,
                    trend_score=0.0,
                )
                target_px = _safe_float(be.get("target_px"), 0.0)
                max_dev_pct = _safe_float(be.get("max_dev_pct"), 0.50)
                if target_px <= 0.0:
                    return None
                min_qty = extra_qty * 0.55
                max_qty = max(extra_qty * 1.45, extra_qty + 1e-12)
                recovery_orders = []
                for o in tp_orders:
                    if bool(o.get("close_position")):
                        continue
                    q = abs(_safe_float(o.get("qty"), 0.0))
                    if q <= 0.0:
                        continue
                    if q < min_qty or q > max_qty:
                        continue
                    recovery_orders.append(o)

                def _is_recovery_tp(o: dict) -> bool:
                    try:
                        q = abs(_safe_float(o.get("qty"), 0.0))
                        px = _safe_float(o.get("trigger_price"), 0.0)
                        if q <= 0 or px <= 0:
                            return False
                        q_ok = (q >= min_qty) and (q <= max_qty)
                        dev_pct = abs(px - base_entry) / max(abs(base_entry), 1e-12) * 100.0
                        side_ok = (px > base_entry) if long_side else (px < base_entry)
                        near_ok = abs(px - target_px) / max(abs(target_px), 1e-12) <= 0.002
                        return bool(q_ok and side_ok and near_ok and dev_pct <= (max_dev_pct + 1e-9))
                    except Exception:
                        return False

                keep = None
                for o in recovery_orders:
                    if _is_recovery_tp(o):
                        if keep is None:
                            keep = o
                        else:
                            px0 = abs(_safe_float(keep.get("trigger_price"), 0.0) - target_px)
                            px1 = abs(_safe_float(o.get("trigger_price"), 0.0) - target_px)
                            if px1 < px0:
                                keep = o

                for o in recovery_orders:
                    if keep is not None and o is keep:
                        continue
                    oid = o.get("id")
                    if oid is None:
                        continue
                    t0 = time.time()
                    try:
                        cancel_order_by_id(
                            int(acc_idx),
                            str(symbol),
                            int(oid),
                            is_algo=bool(o.get("is_algo")),
                            caller="AITM",
                        )
                        out.setdefault("orphan_recovery_tp_cancelled", 0)
                        out["orphan_recovery_tp_cancelled"] = int(out.get("orphan_recovery_tp_cancelled", 0)) + 1
                    except Exception:
                        pass
                    elapsed = time.time() - t0
                    self._emit_latency_warnings(
                        action="recovery_tp_cancel",
                        acc_idx=int(acc_idx),
                        symbol=str(symbol),
                        elapsed_sec=float(elapsed),
                        settings=settings,
                    )

                if keep is None:
                    tp_place_started = time.time()
                    place_exchange_stop_order(
                        int(acc_idx),
                        str(symbol),
                        close_side,
                        float(target_px),
                        order_type="TAKE_PROFIT_MARKET",
                        quantity=float(extra_qty),
                        close_position=False,
                        position_side=position_side,
                        caller="AITM",
                    )
                    tp_place_elapsed = time.time() - tp_place_started
                    self._emit_latency_warnings(
                        action="recovery_tp_place",
                        acc_idx=int(acc_idx),
                        symbol=str(symbol),
                        elapsed_sec=float(tp_place_elapsed),
                        settings=settings,
                    )
                    out["recovery_tp_placed"] = True
                    out["recovery_tp_price"] = float(target_px)
                    out["recovery_tp_qty"] = float(extra_qty)
                    out["recovery_tp_rule_pct"] = float(abs(target_px - base_entry) / max(abs(base_entry), 1e-12) * 100.0)
                else:
                    out["recovery_tp_price"] = float(_safe_float(keep.get("trigger_price"), target_px))
                    out["recovery_tp_qty"] = float(abs(_safe_float(keep.get("qty"), extra_qty)))
            except Exception:
                pass

        ctx["last_minute_guard_ts"] = float(now)
        self._ctx[key] = ctx
        if out.get("sl_placed") or out.get("tp_placed") or out.get("recovery_tp_placed"):
            return out
        return None

    def _reconcile_protection_orders(
        self,
        acc_idx: int,
        symbol: str,
        settings: dict,
        long_side: bool,
        entry: float,
        cur: float,
        amt: float,
        position_side,
        atr: float,
    ) -> dict:
        out = {"status": "skipped", "reason": "no_action", "details": {}}
        if amt <= 0 or entry <= 0:
            out["reason"] = "bad_position"
            return out
        close_side = "SELL" if long_side else "BUY"
        try:
            tp_mode = _safe_int(settings.get("tp_mode"), 2)
            if tp_mode < 1:
                tp_mode = 1
            if tp_mode > 3:
                tp_mode = 3
            tp_raw = settings.get("tp_pcts")
            tp_pcts = [float(v) for v in tp_raw] if isinstance(tp_raw, list) else []
            tp_pcts = [v for v in tp_pcts if v > 0]
        except Exception:
            tp_mode = 2
            tp_pcts = []
        if not tp_pcts:
            tp_pcts = [1.0, 2.0, 3.0]
        tp_legs = max(1, min(tp_mode, len(tp_pcts)))
        tp_use = tp_pcts[:tp_legs]

        sl_pct = _safe_float(settings.get("sl_pct"), 0.0)
        if sl_pct <= 0.0:
            # conservative fallback if account SL config missing
            sl_pct = max(1.2, min(4.5, abs((atr / max(entry, 1e-12)) * 100.0) * 1.4))
        # Profit-protect floor for reconciliation path as well, so order
        # rebuilding never degrades SL back below/above entry in good profit.
        profit_lock_floor_pct = max(0.0, _safe_float(settings.get("ai_tm_profit_lock_min_pct"), 0.10))
        hi_profit1_lock = max(profit_lock_floor_pct, _safe_float(settings.get("ai_tm_profit_lock_tier1_pct"), 0.12))
        hi_profit2_lock = max(hi_profit1_lock, _safe_float(settings.get("ai_tm_profit_lock_tier2_pct"), 0.25))
        hi_profit3_lock = max(hi_profit2_lock, _safe_float(settings.get("ai_tm_profit_lock_tier3_pct"), 0.45))
        if long_side:
            sl_px = entry * (1.0 - sl_pct / 100.0)
            if cur > entry:
                unreal = ((cur - entry) / max(entry, 1e-12)) * 100.0
                lock_pct = profit_lock_floor_pct
                if unreal >= 3.0:
                    lock_pct = hi_profit3_lock
                elif unreal >= 2.0:
                    lock_pct = hi_profit2_lock
                elif unreal >= 1.0:
                    lock_pct = hi_profit1_lock
                sl_px = max(sl_px, entry * (1.0 + lock_pct / 100.0))
            if cur > 0:
                sl_px = min(sl_px, cur * 0.9995)
        else:
            sl_px = entry * (1.0 + sl_pct / 100.0)
            if cur < entry:
                unreal = ((entry - cur) / max(entry, 1e-12)) * 100.0
                lock_pct = profit_lock_floor_pct
                if unreal >= 3.0:
                    lock_pct = hi_profit3_lock
                elif unreal >= 2.0:
                    lock_pct = hi_profit2_lock
                elif unreal >= 1.0:
                    lock_pct = hi_profit1_lock
                sl_px = min(sl_px, entry * (1.0 - lock_pct / 100.0))
            if cur > 0:
                sl_px = max(sl_px, cur * 1.0005)

        # Cancel existing orders FIRST (synchronous) so Binance doesn't reject
        # new orders due to duplicate SL/TP constraints.  Wait for completion
        # before placing any new orders.
        cancel_symbol_open_orders(acc_idx, symbol, "AITM")
        import time as _time_reconcile
        _time_reconcile.sleep(0.3)  # brief settle after bulk cancel

        # Place SL first, then TP legs SEQUENTIALLY with 500 ms between each.
        # Parallel submission was the root cause of Binance order rejections.
        placed_tp = 0
        leg_qty = amt / float(tp_legs)
        try:
            ok_sl = place_exchange_stop_order(
                acc_idx, symbol, close_side, sl_px,
                "STOP_MARKET", None, True, position_side, "MARK_PRICE", "AITM",
            )
            if ok_sl:
                placed_tp += 1
        except Exception:
            pass

        for i, pct in enumerate(tp_use):
            _time_reconcile.sleep(0.5)  # 500 ms between every order
            q = (amt - leg_qty * i) if i == (tp_legs - 1) else leg_qty
            if q <= 0:
                continue
            if long_side:
                tp_px = entry * (1.0 + float(pct) / 100.0)
            else:
                tp_px = entry * (1.0 - float(pct) / 100.0)
            try:
                ok_tp = place_exchange_stop_order(
                    acc_idx, symbol, close_side, tp_px,
                    "TAKE_PROFIT_MARKET", q, False, position_side, "MARK_PRICE", "AITM",
                )
                if ok_tp:
                    placed_tp += 1
            except Exception:
                pass

        out["status"] = "applied"
        out["reason"] = "orders_reconciled"
        out["details"] = {
            "tp_mode": int(tp_mode),
            "tp_legs_target": int(tp_legs),
            "tp_legs_placed": int(max(0, placed_tp - 1)),
            "position_qty": float(amt),
            "sl_price": float(sl_px),
            "sl_pct": float(sl_pct),
        }
        return out

    def _apply_action(
        self,
        acc_idx: int,
        symbol: str,
        pos: dict,
        settings: dict,
        action: str,
        features: Dict[str, object],
        decision_meta: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        side = "BUY" if _safe_float(pos.get("position_amt")) > 0 else "SELL"
        long_side = side == "BUY"
        entry = _safe_float(pos.get("entry_price"))
        cur = _safe_float(features.get("last_price"), 0.0)
        amt = abs(_safe_float(pos.get("position_amt")))
        position_side = pos.get("position_side")
        atr = _safe_float(features.get("atr"))
        out = {
            "new_sl": None,
            "new_tp": None,
            "scale_qty": None,
            "status": "noop",
            "reason": "",
            "details": {},
            "api_latency_ms": 0.0,
            "model_action": (decision_meta or {}).get("model_action"),
            "model_confidence": (decision_meta or {}).get("model_confidence"),
            "override_reason": (decision_meta or {}).get("override_reason"),
        }
        mode = str((decision_meta or {}).get("mode") or "conservative_mode")
        risk_score = _safe_float((decision_meta or {}).get("risk_score"), 0.0)
        trend_score = _safe_float((decision_meta or {}).get("trend_score"), 0.0)

        def _similar_order_exists(order_type: str, close_side: str, trigger_px: float, qty: float, close_pos: bool) -> bool:
            try:
                existing = self._fetch_open_protection_orders(int(acc_idx), str(symbol))
            except Exception:
                return False
            for o in existing:
                try:
                    if str(o.get("type") or "").upper() != str(order_type).upper():
                        continue
                    if str(o.get("side") or "").upper() != str(close_side).upper():
                        continue
                    o_close = bool(o.get("close_position"))
                    if bool(close_pos) != o_close:
                        continue
                    opx = _safe_float(o.get("trigger_price"), 0.0)
                    if opx <= 0.0:
                        continue
                    if abs(opx - float(trigger_px)) > max(1e-10, abs(float(trigger_px)) * 0.0006):
                        continue
                    if not close_pos:
                        oq = _safe_float(o.get("qty"), 0.0)
                        if abs(oq - float(qty)) > max(1e-8, abs(float(qty)) * 0.10):
                            continue
                    return True
                except Exception:
                    continue
            return False

        def _emit_exec_latency_critical(local_action: str) -> None:
            lat_ms = _safe_float(out.get("api_latency_ms"), 0.0)
            if lat_ms > 3000.0:
                log(
                    f"[AI-TM][LATENCY][CRITICAL] account={acc_idx} symbol={symbol} "
                    f"action={local_action} api_latency_ms={lat_ms:.1f}"
                )

        order_hint = (decision_meta or {}).get("order_hint") if isinstance(decision_meta, dict) else None
        if isinstance(order_hint, dict) and bool(order_hint.get("suggest")):
            log(
                f"[AI-TM][ORDER-CHECK][TRY] account={acc_idx} symbol={symbol} "
                f"issues={order_hint.get('issues')} score={_safe_float(order_hint.get('score'), 0.0):.3f} "
                f"position_qty={amt:.8f}"
            )
            try:
                rec = self._reconcile_protection_orders(
                    acc_idx=acc_idx,
                    symbol=symbol,
                    settings=settings,
                    long_side=long_side,
                    entry=entry,
                    cur=cur,
                    amt=amt,
                    position_side=position_side,
                    atr=atr,
                )
                out["status"] = str(rec.get("status") or "applied")
                out["reason"] = str(rec.get("reason") or "orders_reconciled")
                out["details"] = {
                    **(rec.get("details") or {}),
                    "order_hint": order_hint,
                }
                key = (acc_idx, symbol)
                ctx = self._ctx.get(key) or {}
                ctx["last_order_reconcile_ts"] = time.time()
                self._ctx[key] = ctx
            except Exception as e:
                out["status"] = "error"
                out["reason"] = "order_reconcile_failed"
                out["details"] = {"order_hint": order_hint, "error": str(e)}
            log(
                f"[AI-TM][ORDER-CHECK][RESULT] account={acc_idx} symbol={symbol} "
                f"status={out.get('status')} reason={out.get('reason')} details={out.get('details')}"
            )
            return "HOLD", out

        # safety limits
        max_scale_in = min(2, max(0, _safe_int(settings.get("ai_tm_max_scale_in_count"), 2)))
        max_mult = max(_safe_float(settings.get("ai_tm_max_notional_multiplier"), 1.8), 1.0)
        max_dd = abs(_safe_float(settings.get("ai_tm_max_drawdown_pct"), 6.0))

        key = (acc_idx, symbol)
        ctx = self._ctx.get(key, {})
        scale_count = _safe_int(ctx.get("scale_in_count"), 0)
        base_amt = max(_safe_float(ctx.get("base_amt"), amt), 1e-12)
        unreal = _safe_float(ctx.get("unreal_pnl_pct"), 0.0)
        model_conf_live = _safe_float((decision_meta or {}).get("model_confidence"), 0.0)

        # Smart early exit: if trend weakens and ML confidence degrades, close early.
        smart_exit_trend_abs_max = max(0.01, _safe_float(settings.get("ai_tm_smart_exit_trend_abs_max"), 0.08))
        smart_exit_conf_max = max(0.10, _safe_float(settings.get("ai_tm_smart_exit_conf_max"), 0.52))
        if action in ("HOLD", "MOVE_TP", "MOVE_SL") and unreal > 0.02:
            if abs(float(trend_score)) <= smart_exit_trend_abs_max and model_conf_live <= smart_exit_conf_max:
                action = "CLOSE_TRADE"
                out["details"] = {
                    **(out.get("details") or {}),
                    "smart_exit_triggered": True,
                    "trend_score": float(trend_score),
                    "trend_abs_max": float(smart_exit_trend_abs_max),
                    "model_confidence": float(model_conf_live),
                    "conf_max": float(smart_exit_conf_max),
                }

        if action == "CLOSE_TRADE":
            log(
                f"[AI-TM][ACTION][TRY] account={acc_idx} symbol={symbol} action=CLOSE_TRADE "
                f"side={'LONG' if long_side else 'SHORT'} qty={amt:.8f}"
            )
            try:
                signed_amt = amt if long_side else -amt
                # ── STEP 1: Cancel ALL open orders FIRST (especially DCA limit
                # orders).  If we close position first and DCA orders are still
                # open, they may fill immediately and re-open a new position.
                import time as _time_close
                cancel_symbol_open_orders(acc_idx, symbol, "AITM")
                _time_close.sleep(0.3)  # wait for cancels to reach exchange
                # ── STEP 2: Close the position.
                close_position_market(
                    acc_idx,
                    symbol,
                    signed_amt,
                    force_full_close=True,
                    position_side=position_side,
                    caller="AITM",
                )
                out["status"] = "applied"
                out["reason"] = "closed_by_ai"
                out["details"] = {
                    "position_amt": float(amt),
                    "side": "LONG" if long_side else "SHORT",
                }
            except Exception:
                out["status"] = "error"
                out["reason"] = "close_failed"
            log(
                f"[AI-TM][ACTION][RESULT] account={acc_idx} symbol={symbol} action=CLOSE_TRADE "
                f"status={out.get('status')} reason={out.get('reason')} details={out.get('details')}"
            )
            return action, out

        if action == "MOVE_SL" and entry > 0 and cur > 0:
            # Advisory guard (not hard rule):
            # In low-profit area, keep MOVE_SL only when there is enough edge.
            min_profit_hint = max(0.20, _safe_float(settings.get("ai_tm_min_profit_lock_pct"), 0.8))
            model_conf = _safe_float((decision_meta or {}).get("model_confidence"), 0.0)
            override_reason = str((decision_meta or {}).get("override_reason") or "")
            forced_override = override_reason in (
                "hard_profit_protection",
                "hold_demoted_to_sl",
                "profit_lock_soft_priority",
            )
            mkt_state = str(features.get("market_state") or "ACTIVE").upper()
            mom = _safe_float(features.get("momentum"), 0.0)
            acc = _safe_float(features.get("acceleration"), 0.0)
            if long_side:
                adverse = mom < 0.0 and acc < 0.0
            else:
                adverse = mom > 0.0 and acc > 0.0

            if unreal < min_profit_hint and (not forced_override):
                # Soft-priority hint should not force action too early.
                if override_reason == "profit_lock_soft_priority":
                    out["status"] = "skipped"
                    out["reason"] = "profit_lock_not_ripe"
                    out["details"] = {
                        "unreal_pnl_pct": float(unreal),
                        "min_profit_hint_pct": float(min_profit_hint),
                        "model_confidence": float(model_conf),
                    }
                    return "HOLD", out
                # For model-originated MOVE_SL, require stronger evidence in low-profit.
                if model_conf < 0.72 and (not adverse) and mkt_state != "PASSIVE":
                    out["status"] = "skipped"
                    out["reason"] = "model_sl_low_edge"
                    out["details"] = {
                        "unreal_pnl_pct": float(unreal),
                        "min_profit_hint_pct": float(min_profit_hint),
                        "model_confidence": float(model_conf),
                        "market_state": mkt_state,
                        "adverse_momentum": bool(adverse),
                    }
                    return "HOLD", out
            # AI-driven trailing SL proposal from ATR & pnl context.
            step_mult = 0.45 if mode == "aggressive_trend_mode" else 0.55
            step = max((abs(atr / entry) * 100.0) * step_mult, 0.25)
            lock_floor_pct = max(0.0, _safe_float(settings.get("ai_tm_profit_lock_min_pct"), 0.10))
            hi_profit1_lock = max(lock_floor_pct, _safe_float(settings.get("ai_tm_profit_lock_tier1_pct"), 0.12))
            hi_profit2_lock = max(hi_profit1_lock, _safe_float(settings.get("ai_tm_profit_lock_tier2_pct"), 0.25))
            hi_profit3_lock = max(hi_profit2_lock, _safe_float(settings.get("ai_tm_profit_lock_tier3_pct"), 0.45))
            lock_pct_use = lock_floor_pct
            if unreal >= 3.0:
                lock_pct_use = hi_profit3_lock
            elif unreal >= 2.0:
                lock_pct_use = hi_profit2_lock
            elif unreal >= 1.0:
                lock_pct_use = hi_profit1_lock
            if long_side:
                sl_px = cur * (1.0 - step / 100.0)
                # In profit, do not leave SL below entry (plus tiny lock floor).
                if unreal > 0.0:
                    sl_px = max(sl_px, entry * (1.0 + lock_pct_use / 100.0))
                    sl_px = min(sl_px, cur * 0.9995)
                close_side = "SELL"
            else:
                sl_px = cur * (1.0 + step / 100.0)
                # In profit, do not leave SL above entry (minus tiny lock floor).
                if unreal > 0.0:
                    sl_px = min(sl_px, entry * (1.0 - lock_pct_use / 100.0))
                    sl_px = max(sl_px, cur * 1.0005)
                close_side = "BUY"
            log(
                f"[AI-TM][ACTION][TRY] account={acc_idx} symbol={symbol} action=MOVE_SL "
                f"side={'LONG' if long_side else 'SHORT'} cur={cur:.8f} entry={entry:.8f} unreal={unreal:.4f}% "
                f"sl_candidate={sl_px:.8f} atr_step_pct={step:.4f}"
            )
            attach_tp = bool(settings.get("ai_tm_attach_tp_on_sl_move", True))
            if mode == "aggressive_trend_mode":
                tp_cd_sec = max(120.0, _safe_float(settings.get("ai_tm_tp_refresh_cooldown_sec"), 600.0))
            else:
                tp_cd_sec = max(60.0, _safe_float(settings.get("ai_tm_tp_refresh_cooldown_sec"), 600.0))
            last_tp = _safe_float(ctx.get("last_tp_refresh_ts"), 0.0)
            tp_px = None
            tp_qty = 0.0
            tp_step = max((abs(atr / entry) * 100.0) * 1.2, 0.6)
            should_place_tp = bool(attach_tp and (time.time() - last_tp) >= tp_cd_sec and amt > 0.0)
            if should_place_tp:
                if long_side:
                    tp_px = cur * (1.0 + tp_step / 100.0)
                else:
                    tp_px = cur * (1.0 - tp_step / 100.0)
                tp_qty = max(amt * 0.25, 0.0)
            try:
                api_t0 = time.time()
                futures = []
                # Idempotency: skip if near-identical protection order already exists.
                if not _similar_order_exists("STOP_MARKET", close_side, float(sl_px), 0.0, True):
                    futures.append(
                        self._io_pool.submit(
                            place_exchange_stop_order,
                            acc_idx,
                            symbol,
                            close_side,
                            sl_px,
                            "STOP_MARKET",
                            None,
                            True,
                            position_side,
                            "MARK_PRICE",
                            "AITM",
                        )
                    )
                if should_place_tp and tp_px is not None and tp_qty > 0.0:
                    if not _similar_order_exists("TAKE_PROFIT_MARKET", close_side, float(tp_px), float(tp_qty), False):
                        futures.append(
                            self._io_pool.submit(
                                place_exchange_stop_order,
                                acc_idx,
                                symbol,
                                close_side,
                                tp_px,
                                "TAKE_PROFIT_MARKET",
                                tp_qty,
                                False,
                                position_side,
                                "MARK_PRICE",
                                "AITM",
                            )
                        )
                for f in as_completed(futures, timeout=8.0):
                    try:
                        f.result()
                    except Exception:
                        continue
                out["api_latency_ms"] = max(0.0, (time.time() - api_t0) * 1000.0)
                out["new_sl"] = float(sl_px)
                out["status"] = "applied"
                out["reason"] = "sl_moved"
                out["details"] = {
                    "atr_step_pct": float(step),
                    "lock_floor_pct": float(lock_floor_pct),
                    "lock_pct_used": float(lock_pct_use),
                }
                ctx["last_sl_adjust_ts"] = time.time()
                ctx["last_sl_move_ts"] = time.time()
                self._ctx[(acc_idx, symbol)] = ctx
                if should_place_tp and tp_px is not None and tp_qty > 0.0:
                    out["new_tp"] = float(tp_px)
                    out["details"]["tp_refresh"] = {
                        "status": "applied",
                        "tp_qty": float(tp_qty),
                        "tp_step_pct": float(tp_step),
                    }
                    ctx["last_tp_refresh_ts"] = time.time()
                    ctx["last_tp_move_ts"] = time.time()
                    self._ctx[(acc_idx, symbol)] = ctx
            except Exception:
                out["status"] = "error"
                out["reason"] = "move_sl_failed"
            log(
                f"[AI-TM][ACTION][RESULT] account={acc_idx} symbol={symbol} action=MOVE_SL "
                f"status={out.get('status')} reason={out.get('reason')} new_sl={out.get('new_sl')} "
                f"new_tp={out.get('new_tp')} details={out.get('details')}"
            )
            _emit_exec_latency_critical("MOVE_SL")
            return action, out

        if action == "MOVE_TP" and entry > 0 and cur > 0:
            tp_step = max((abs(atr / entry) * 100.0) * 1.2, 0.6)
            if long_side:
                tp_px = cur * (1.0 + tp_step / 100.0)
                close_side = "SELL"
            else:
                tp_px = cur * (1.0 - tp_step / 100.0)
                close_side = "BUY"
            log(
                f"[AI-TM][ACTION][TRY] account={acc_idx} symbol={symbol} action=MOVE_TP "
                f"side={'LONG' if long_side else 'SHORT'} cur={cur:.8f} entry={entry:.8f} "
                f"tp_candidate={tp_px:.8f} atr_step_pct={tp_step:.4f}"
            )
            try:
                qty = max(amt * 0.33, 0.0)
                api_t0 = time.time()
                if not _similar_order_exists("TAKE_PROFIT_MARKET", close_side, float(tp_px), float(qty), False):
                    place_exchange_stop_order(
                        acc_idx, symbol, close_side, tp_px,
                        order_type="TAKE_PROFIT_MARKET",
                        quantity=qty,
                        close_position=False,
                        position_side=position_side,
                        caller="AITM",
                    )
                out["api_latency_ms"] = max(0.0, (time.time() - api_t0) * 1000.0)
                out["new_tp"] = float(tp_px)
                out["status"] = "applied"
                out["reason"] = "tp_moved"
                out["details"] = {"atr_step_pct": float(tp_step), "tp_qty": float(qty)}
                key = (acc_idx, symbol)
                ctx = self._ctx.get(key) or {}
                ctx["last_tp_move_ts"] = time.time()
                self._ctx[key] = ctx
            except Exception:
                out["status"] = "error"
                out["reason"] = "move_tp_failed"
            log(
                f"[AI-TM][ACTION][RESULT] account={acc_idx} symbol={symbol} action=MOVE_TP "
                f"status={out.get('status')} reason={out.get('reason')} new_tp={out.get('new_tp')} details={out.get('details')}"
            )
            _emit_exec_latency_critical("MOVE_TP")
            return action, out

        if action == "REMOVE_TP":
            log(
                f"[AI-TM][ACTION][TRY] account={acc_idx} symbol={symbol} action=REMOVE_TP "
                f"side={'LONG' if long_side else 'SHORT'}"
            )
            try:
                # keep only protection SL by canceling open orders in background
                import threading as _thr_rmtp
                _thr_rmtp.Thread(
                    target=cancel_symbol_open_orders,
                    args=(acc_idx, symbol, "AITM"),
                    daemon=True,
                ).start()
                sl_step = max((abs(atr / max(entry, 1e-12)) * 100.0), 0.6)
                if long_side:
                    sl_px = cur * (1.0 - sl_step / 100.0)
                    close_side = "SELL"
                else:
                    sl_px = cur * (1.0 + sl_step / 100.0)
                    close_side = "BUY"
                _rmtp_fut = self._io_pool.submit(
                    place_exchange_stop_order,
                    acc_idx, symbol, close_side, sl_px,
                    "STOP_MARKET", None, True, position_side, "MARK_PRICE", "AITM",
                )
                try:
                    for _f in as_completed([_rmtp_fut], timeout=8.0):
                        _f.result()
                except (FuturesTimeoutError, Exception):
                    pass
                out["new_sl"] = float(sl_px)
                out["status"] = "applied"
                out["reason"] = "tp_removed_keep_sl_only"
                out["details"] = {"sl_step_pct": float(sl_step)}
            except Exception:
                out["status"] = "error"
                out["reason"] = "remove_tp_failed"
            log(
                f"[AI-TM][ACTION][RESULT] account={acc_idx} symbol={symbol} action=REMOVE_TP "
                f"status={out.get('status')} reason={out.get('reason')} new_sl={out.get('new_sl')} details={out.get('details')}"
            )
            return action, out

        if action == "SCALE_IN":
            log(
                f"[AI-TM][ACTION][TRY] account={acc_idx} symbol={symbol} action=SCALE_IN "
                f"side={'LONG' if long_side else 'SHORT'} amt={amt:.8f} base_amt={base_amt:.8f} "
                f"scale_count={scale_count}/{max_scale_in} unreal={unreal:.4f}%"
            )
            # Strict risk gates for scale-in.
            if scale_count >= max_scale_in:
                out["status"] = "skipped"
                out["reason"] = "scale_in_limit_reached"
                out["details"] = {"scale_in_count": int(scale_count), "max_scale_in": int(max_scale_in)}
                return "HOLD", out
            # Entry-quality gate (profitability-first):
            # require minimum confidence and direction-aligned trend.
            model_conf_gate = _safe_float((decision_meta or {}).get("model_confidence"), 0.0)
            if model_conf_gate < 0.64:
                out["status"] = "skipped"
                out["reason"] = "scale_in_blocked_low_ml_conf"
                out["details"] = {"ml_conf": float(model_conf_gate), "ml_conf_min": 0.64}
                return "HOLD", out
            trend_min_long = _safe_float(settings.get("ai_tm_scale_in_trend_long_min"), 0.20)
            trend_max_short = _safe_float(settings.get("ai_tm_scale_in_trend_short_max"), -0.20)
            if long_side and float(trend_score) < float(trend_min_long):
                out["status"] = "skipped"
                out["reason"] = "scale_in_blocked_trend_long"
                out["details"] = {"trend_score": float(trend_score), "trend_min": float(trend_min_long)}
                return "HOLD", out
            if (not long_side) and float(trend_score) > float(trend_max_short):
                out["status"] = "skipped"
                out["reason"] = "scale_in_blocked_trend_short"
                out["details"] = {"trend_score": float(trend_score), "trend_max": float(trend_max_short)}
                return "HOLD", out
            if unreal < -max_dd:
                out["status"] = "skipped"
                out["reason"] = "scale_in_blocked_max_drawdown"
                out["details"] = {"unreal_pnl_pct": float(unreal), "max_drawdown_pct": float(max_dd)}
                return "HOLD", out
            if unreal < -0.30:
                out["status"] = "skipped"
                out["reason"] = "scale_in_blocked_loss_control"
                out["details"] = {"unreal_pnl_pct": float(unreal), "loss_block_pct": -0.30}
                return "HOLD", out
            if amt >= (base_amt * max_mult):
                out["status"] = "skipped"
                out["reason"] = "scale_in_blocked_max_multiplier"
                out["details"] = {
                    "position_amt": float(amt),
                    "base_amt": float(base_amt),
                    "max_multiplier": float(max_mult),
                }
                return "HOLD", out
            if self._flag_enabled(settings, "ai_tm_scale_in_ml_gate_enabled", False):
                model_conf = _safe_float((decision_meta or {}).get("model_confidence"), 0.0)
                exposure_ratio = min(1.0, max(0.0, amt / max(base_amt * max_mult, 1e-12)))
                trend_min = _safe_float(settings.get("ai_tm_scale_in_trend_min"), 0.05)
                exposure_max = _safe_float(settings.get("ai_tm_scale_in_exposure_max"), 0.80)
                conf_min = _safe_float(settings.get("ai_tm_scale_in_conf_min"), 0.55)
                if trend_score < trend_min or exposure_ratio > exposure_max or model_conf < conf_min or risk_score > 0.80:
                    out["status"] = "skipped"
                    out["reason"] = "scale_in_blocked_ml_gate"
                    out["details"] = {
                        "trend_score": float(trend_score),
                        "trend_min": float(trend_min),
                        "exposure_ratio": float(exposure_ratio),
                        "exposure_max": float(exposure_max),
                        "model_confidence": float(model_conf),
                        "model_conf_min": float(conf_min),
                        "risk_score": float(risk_score),
                    }
                    return "HOLD", out
            # Safety filter: block scale-in under high immediate risk
            # or strongly adverse trend.
            if risk_score > 0.70 or (long_side and trend_score <= 0.0) or ((not long_side) and trend_score >= 0.0):
                out["status"] = "skipped"
                out["reason"] = "scale_in_blocked_safety_filter"
                out["details"] = {
                    "scale_in_blocked": True,
                    "risk_score": float(risk_score),
                    "trend_score": float(trend_score),
                }
                return "HOLD", out
            # Trade frequency control: short cooldown after close before re-entry/scale.
            cd_low = max(1.0, _safe_float(settings.get("ai_tm_reentry_cooldown_min_sec"), 5.0))
            cd_high = max(cd_low, _safe_float(settings.get("ai_tm_reentry_cooldown_max_sec"), 10.0))
            cd_use = (cd_low + cd_high) / 2.0
            last_close_ts = _safe_float(self._last_close_ts.get((acc_idx, symbol)), 0.0)
            if last_close_ts > 0 and (time.time() - last_close_ts) < cd_use:
                out["status"] = "skipped"
                out["reason"] = "scale_in_blocked_cooldown"
                out["details"] = {"cooldown_sec": float(cd_use), "elapsed_sec": float(time.time() - last_close_ts)}
                return "HOLD", out
            try:
                add_frac = _safe_float(settings.get("ai_tm_scale_in_add_fraction"), 0.18)
                if add_frac < 0.08:
                    add_frac = 0.08
                if add_frac > 0.30:
                    add_frac = 0.30
                max_add_qty = max(0.0, (base_amt * max_mult) - amt)
                add_qty = max(base_amt * add_frac, 0.0)
                if max_add_qty > 0:
                    add_qty = min(add_qty, max_add_qty)
                else:
                    add_qty = 0.0
                if add_qty > 0:
                    place_order(
                        symbol=symbol,
                        side="BUY" if long_side else "SELL",
                        quantity=add_qty,
                        target_account_index=acc_idx,
                        caller="AITM",
                    )
                    # After scale-in, place a tranche-specific TP (partial close).
                    # This closes ONLY the newly added size near breakeven.
                    fixed_recovery_pct, recovery_policy = self._policy_recovery_tp_pct(
                        settings=settings,
                        long_side=long_side,
                        model_conf=_safe_float((decision_meta or {}).get("model_confidence"), 0.0),
                        trend_score=float(trend_score),
                        mode=mode,
                    )
                    tiny_plus_pct = 0.0
                    fee_buffer_pct = 0.0
                    funding_buffer_pct = 0.0
                    be_buffer_pct = float(fixed_recovery_pct)

                    recovery_tp_px = None
                    recovery_tp_qty = float(add_qty)
                    recovery_tp_status = "skipped"
                    recovery_tp_reason = "entry_missing"
                    blended_entry = float(entry) if entry > 0 else 0.0
                    added_entry = float(entry) if entry > 0 else 0.0
                    blended_source = "pre_scale_entry"

                    # Try to read refreshed position (post-fill) to use actual blended entry.
                    try:
                        for _ in range(3):
                            snap = get_open_positions_snapshot() or []
                            hit = None
                            for row in snap:
                                if not isinstance(row, dict):
                                    continue
                                if _safe_int(row.get("account_index"), -1) != int(acc_idx):
                                    continue
                                if str(row.get("symbol") or "").upper() != str(symbol).upper():
                                    continue
                                if abs(_safe_float(row.get("position_amt"), 0.0)) <= 0:
                                    continue
                                hit = row
                                break
                            if hit is not None:
                                n_entry = _safe_float(hit.get("entry_price"), 0.0)
                                n_amt = abs(_safe_float(hit.get("position_amt"), 0.0))
                                if n_entry > 0:
                                    blended_entry = n_entry
                                    blended_source = "post_scale_blended_entry"
                                if n_amt > amt:
                                    recovery_tp_qty = max(n_amt - amt, 0.0)
                                    if n_entry > 0 and amt > 0:
                                        try:
                                            # Derive fill price of added chunk from weighted-average update.
                                            calc_added = ((n_entry * n_amt) - (entry * amt)) / max((n_amt - amt), 1e-12)
                                            if calc_added > 0:
                                                added_entry = float(calc_added)
                                        except Exception:
                                            added_entry = blended_entry
                                break
                            time.sleep(0.25)
                    except Exception:
                        pass

                    tranche_id = f"{int(time.time()*1000)}:{acc_idx}:{symbol}:tranche:{scale_count + 1}"
                    tranche_entry = float(added_entry if added_entry > 0 else (blended_entry if blended_entry > 0 else entry))
                    tranche_meta = {
                        "tranche_id": str(tranche_id),
                        "qty": float(recovery_tp_qty),
                        "entry_price": float(tranche_entry) if tranche_entry > 0 else None,
                        "tp_price": None,
                        "tp_policy": None,
                        "tp_order_id": None,
                        "breakeven_locked": False,
                        "protective_sl_price": None,
                        "tp_proximity_window_pct": float(
                            min(5.0, max(0.2, _safe_float(settings.get("ai_tm_breakeven_lock_proximity_pct"), 2.0)))
                        ),
                        "status": "created",
                        "created_ts": float(time.time()),
                    }
                    if blended_entry > 0 and recovery_tp_qty > 0:
                        # STRICT CONTRACT: TP is computed exactly once, then executed (or retried) without mutation.
                        ref_entry = tranche_entry if tranche_entry > 0 else (added_entry if added_entry > 0 else blended_entry)
                        overall_ref_entry = float(entry) if float(entry) > 0 else float(blended_entry)
                        be = self._compute_tranche_breakeven_tp(
                            settings=settings,
                            long_side=long_side,
                            tranche_entry=float(ref_entry),
                            overall_entry=float(overall_ref_entry),
                            mode=mode,
                            model_conf=_safe_float((decision_meta or {}).get("model_confidence"), 0.0),
                            trend_score=float(trend_score),
                        )
                        recovery_policy = str(be.get("policy") or "tranche_breakeven_tp")
                        recovery_tp_px = float(be.get("target_px") or 0.0)
                        be_buffer_pct = float(be.get("tp_pct") or be_buffer_pct)
                        fee_buffer_pct = float(be.get("fee_buffer_pct") or 0.0)
                        funding_buffer_pct = float(be.get("funding_buffer_pct") or 0.0)
                        tiny_plus_pct = float(be.get("tiny_plus_pct") or 0.0)
                        try:
                            trace_rows = be.get("pipeline_trace") if isinstance(be, dict) else None
                            if isinstance(trace_rows, list):
                                for row in trace_rows:
                                    if not isinstance(row, dict):
                                        continue
                                    log(
                                        f"[AI-TM][TP_PIPELINE] account={acc_idx} symbol={symbol} "
                                        f"step={row.get('step_name')} input={row.get('input_price')} "
                                        f"output={row.get('output_price')} delta_pct={row.get('delta_pct')} "
                                        f"reason={row.get('reason')}"
                                    )
                        except Exception:
                            pass

                        attempts = 1
                        # Partial TP: 50% at +0.05%, 50% runner at dynamic TP.
                        tp1_offset_pct = max(0.01, _safe_float(settings.get("ai_tm_partial_tp1_offset_pct"), 0.05))
                        tp1_px = (float(overall_ref_entry) * (1.0 + tp1_offset_pct / 100.0)) if long_side else (
                            float(overall_ref_entry) * (1.0 - tp1_offset_pct / 100.0)
                        )
                        tp1_qty = max(0.0, float(recovery_tp_qty) * 0.5)
                        tp2_qty = max(0.0, float(recovery_tp_qty) - tp1_qty)
                        api_t0 = time.time()
                        if recovery_tp_px > 0 and tp2_qty > 0 and _similar_order_exists(
                            "TAKE_PROFIT_MARKET", close_side, float(recovery_tp_px), float(tp2_qty), False
                        ):
                            recovery_tp_status = "placed"
                            recovery_tp_reason = "scale_in_recovery_tp_exists"
                        elif recovery_tp_px > 0 and tp2_qty > 0:
                            futures = []
                            if tp1_qty > 0 and tp1_px > 0 and not _similar_order_exists(
                                "TAKE_PROFIT_MARKET", close_side, float(tp1_px), float(tp1_qty), False
                            ):
                                futures.append(
                                    self._io_pool.submit(
                                        place_exchange_stop_order,
                                        acc_idx,
                                        symbol,
                                        close_side,
                                        float(tp1_px),
                                        "TAKE_PROFIT_MARKET",
                                        float(tp1_qty),
                                        False,
                                        position_side,
                                        "MARK_PRICE",
                                        "AITM",
                                    )
                                )
                            futures.append(
                                self._io_pool.submit(
                                    place_exchange_stop_order,
                                    acc_idx,
                                    symbol,
                                    close_side,
                                    float(recovery_tp_px),
                                    "TAKE_PROFIT_MARKET",
                                    float(tp2_qty),
                                    False,
                                    position_side,
                                    "MARK_PRICE",
                                    "AITM",
                                )
                            )
                            try:
                                info = None
                                for fut in as_completed(futures, timeout=2.0):
                                    try:
                                        res = fut.result()
                                        if isinstance(res, dict):
                                            info = res
                                    except Exception:
                                        continue
                                recovery_tp_status = "placed"
                                recovery_tp_reason = "scale_in_recovery_tp_placed_async"
                                tranche_meta["tp_order_id"] = (info or {}).get("id") if isinstance(info, dict) else None
                                log(
                                    f"[AI-TM][SCALE-IN][TRANCHE][TP] account={acc_idx} symbol={symbol} "
                                    f"tranche_id={tranche_id} qty={recovery_tp_qty:.8f} "
                                    f"entry={tranche_entry:.8f} tp1={tp1_px:.8f} tp2={recovery_tp_px:.8f} "
                                    f"policy={recovery_policy}"
                                )
                            except FuturesTimeoutError:
                                recovery_tp_status = "pending"
                                recovery_tp_reason = "scale_in_recovery_tp_async_timeout"
                            except Exception:
                                recovery_tp_status = "error"
                                recovery_tp_reason = "scale_in_recovery_tp_failed"
                        else:
                            recovery_tp_status = "error"
                            recovery_tp_reason = "scale_in_recovery_tp_invalid_target"

                        out["api_latency_ms"] = max(0.0, (time.time() - api_t0) * 1000.0)
                        # Fees awareness: TP edge must exceed fee floor.
                        fee_rate = min(max(_safe_float(settings.get("ai_tm_scale_in_taker_fee_rate"), 0.0004), 0.0), 0.003)
                        fee_floor_pct = float(fee_rate * 100.0 * 2.0)
                        if be_buffer_pct < fee_floor_pct:
                            recovery_tp_status = "error"
                            recovery_tp_reason = "scale_in_tp_below_fee_floor"
                        if recovery_tp_status != "placed":
                            # Keep exact target for retry; no secondary TP transformation allowed.
                            ctx["pending_scale_recovery"] = {
                                "target_px": float(recovery_tp_px),
                                "qty": float(tp2_qty) if 'tp2_qty' in locals() else float(recovery_tp_qty),
                                "entry_ref": float(ref_entry),
                                "tranche_id": str(tranche_id),
                                "tranche_entry_price": float(tranche_entry) if tranche_entry > 0 else None,
                                "tranche_tp_price": float(recovery_tp_px) if recovery_tp_px > 0 else None,
                                "created_ts": float(time.time()),
                                "last_try_ts": float(time.time()),
                                "attempts": int(max(attempts, 1)),
                            }
                            self._ctx[(acc_idx, symbol)] = ctx
                            log(
                                f"[AI-TM][SCALE-IN][WARN] account={acc_idx} symbol={symbol} "
                                f"recovery_tp_not_placed qty={recovery_tp_qty:.8f} attempts={attempts} "
                                f"pending_target={ctx.get('pending_scale_recovery', {}).get('target_px')}"
                            )
                    tranche_meta["tp_price"] = float(recovery_tp_px) if recovery_tp_px else None
                    tranche_meta["tp1_price"] = float(tp1_px) if 'tp1_px' in locals() else None
                    tranche_meta["tp1_qty"] = float(tp1_qty) if 'tp1_qty' in locals() else None
                    tranche_meta["tp2_price"] = float(recovery_tp_px) if recovery_tp_px else None
                    tranche_meta["tp2_qty"] = float(tp2_qty) if 'tp2_qty' in locals() else None
                    tranche_meta["partial_tp_enabled"] = True
                    tranche_meta["be_after_tp1"] = False
                    tranche_meta["tp_policy"] = str(recovery_policy)
                    tranche_meta["breakeven_locked"] = bool(recovery_tp_status == "placed")
                    tranche_meta["status"] = "tp_placed" if recovery_tp_status == "placed" else "tp_pending"
                    tranches = ctx.get("scale_in_tranches")
                    if not isinstance(tranches, list):
                        tranches = []
                    tranches.append(tranche_meta)
                    if len(tranches) > 30:
                        tranches = tranches[-30:]
                    ctx["scale_in_tranches"] = tranches

                    out["scale_qty"] = float(add_qty)
                    ctx["scale_in_count"] = scale_count + 1
                    self._ctx[(acc_idx, symbol)] = ctx
                    out["status"] = "applied"
                    out["reason"] = "scale_in_executed"
                    out["details"] = {
                        "new_scale_in_count": int(ctx["scale_in_count"]),
                        "max_scale_in": int(max_scale_in),
                        "scale_in_add_fraction": float(add_frac),
                        "recovery_tp_tiny_plus_pct": float(tiny_plus_pct),
                        "recovery_tp_fee_buffer_pct": float(fee_buffer_pct),
                        "recovery_tp_funding_buffer_pct": float(funding_buffer_pct),
                        "recovery_tp_final_buffer_pct": float(be_buffer_pct),
                        "recovery_tp_rule_pct": float(fixed_recovery_pct),
                        "recovery_tp_policy": str(recovery_policy),
                        "recovery_tp_entry_source": blended_source,
                        "recovery_tp_entry_used": float(blended_entry) if blended_entry > 0 else None,
                        "recovery_tp_added_entry_used": float(added_entry) if added_entry > 0 else None,
                        "recovery_tp_price": float(recovery_tp_px) if recovery_tp_px else None,
                        "recovery_tp_qty": float(recovery_tp_qty),
                        "recovery_tp_status": recovery_tp_status,
                        "recovery_tp_reason": recovery_tp_reason,
                        "recovery_tp_pending": bool(recovery_tp_status != "placed"),
                        "tranche": tranche_meta,
                    }
                else:
                    out["status"] = "skipped"
                    out["reason"] = "scale_in_qty_zero"
            except Exception:
                out["status"] = "error"
                out["reason"] = "scale_in_failed"
            log(
                f"[AI-TM][ACTION][RESULT] account={acc_idx} symbol={symbol} action=SCALE_IN "
                f"status={out.get('status')} reason={out.get('reason')} scale={out.get('scale_qty')} details={out.get('details')}"
            )
            _emit_exec_latency_critical("SCALE_IN")
            return action, out

        out["status"] = "skipped"
        out["reason"] = "model_hold_or_noop"
        return "HOLD", out

    def _train_model_if_due(self) -> None:
        now = time.time()
        if (now - self._last_train_ts) < 3600.0:
            return
        if self._train_future is not None and not self._train_future.done():
            return
        self._last_train_ts = now

        def _task():
            if np is None:
                return
            if not _DECISION_LOG_PATH.exists() or not _OUTCOME_LOG_PATH.exists():
                return
            decisions = {}
            try:
                with _DECISION_LOG_PATH.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        decisions[row.get("decision_id")] = row
            except Exception:
                return
            X, y = [], []
            try:
                with _OUTCOME_LOG_PATH.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            out = json.loads(line)
                        except Exception:
                            continue
                        did = out.get("decision_id")
                        d = decisions.get(did)
                        if not isinstance(d, dict):
                            continue
                        feat = d.get("features_vector") or []
                        if not isinstance(feat, list) or len(feat) < 8:
                            continue
                        act = str(d.get("decision") or "HOLD")
                        unreal_at_decision = _safe_float(d.get("unreal_pnl_pct"), 0.0)
                        final_pnl = _safe_float(out.get("final_unreal_pnl_pct"), 0.0)
                        # Label shaping to avoid HOLD-only collapse:
                        # - when trade gave back from positive area, teach SL tightening
                        # - when deep negative persists, allow close label
                        # - otherwise keep model action
                        label = act if act in _ACTIONS else "HOLD"
                        if final_pnl < -0.1 and unreal_at_decision > 1.5:
                            label = "MOVE_SL"
                        elif final_pnl < -2.5 and unreal_at_decision < -2.0:
                            label = "CLOSE_TRADE"
                        elif final_pnl >= -0.1 and label == "HOLD" and unreal_at_decision > 2.0:
                            label = "MOVE_SL"

                        row = [float(v) for v in feat]
                        X.append(row)
                        y.append(label if label in _ACTIONS else "HOLD")
                        # light oversampling for rare non-HOLD labels
                        if label in ("MOVE_SL", "CLOSE_TRADE"):
                            X.append(row)
                            y.append(label)
            except Exception:
                return
            # Action-conditioned horizon labels (X_t -> Y_horizon) from credit stream.
            try:
                if _CREDIT_LOG_PATH.exists():
                    with _CREDIT_LOG_PATH.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                ev = json.loads(line)
                            except Exception:
                                continue
                            if str(ev.get("event") or "") != "horizon_label":
                                continue
                            h = _safe_int(ev.get("horizon_sec"), 0)
                            if h not in (30, 120, 300, 900):
                                continue
                            feat = ev.get("x_t") or []
                            if not isinstance(feat, list) or len(feat) < 8:
                                continue
                            success = bool(ev.get("label_success"))
                            final_action = str(ev.get("final_action") or "HOLD")
                            delta = _safe_float(ev.get("delta_unreal_pct"), 0.0)
                            if success and final_action in _ACTIONS:
                                label = final_action
                            else:
                                if delta <= -1.0:
                                    label = "CLOSE_TRADE"
                                elif delta < 0.0:
                                    label = "MOVE_SL"
                                else:
                                    label = "HOLD"
                            row = [float(v) for v in feat]
                            X.append(row)
                            y.append(label if label in _ACTIONS else "HOLD")
                            if label in ("MOVE_SL", "CLOSE_TRADE", "MOVE_TP"):
                                X.append(row)
                                y.append(label)
            except Exception:
                pass
            if len(X) < 120:
                return
            try:
                arr_x = np.array(X, dtype=float)
                arr_y = np.array(y, dtype=object)
                n = arr_x.shape[0]
                split = int(n * 0.8)
                if split <= 20 or split >= n:
                    split = max(20, n // 2)
                x_tr, x_va = arr_x[:split], arr_x[split:]
                y_tr, y_va = arr_y[:split], arr_y[split:]

                experts = {}
                scores = {}

                def _fit_and_score(name, model_obj):
                    try:
                        model_obj.fit(x_tr, y_tr)
                        pred = model_obj.predict(x_va)
                        acc = float((pred == y_va).mean()) if len(y_va) > 0 else 0.5
                        experts[name] = model_obj
                        scores[name] = max(acc, 1e-3)
                    except Exception:
                        return

                # 1) Random Forest
                if RandomForestClassifier is not None:
                    _fit_and_score(
                        "rf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=10,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                            random_state=42,
                        ),
                    )
                # 2) Gradient Boosting
                if GradientBoostingClassifier is not None:
                    _fit_and_score("gbr", GradientBoostingClassifier(random_state=42))
                # 3) MLP
                if MLPClassifier is not None:
                    _fit_and_score(
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=(96, 48),
                            activation="relu",
                            solver="adam",
                            max_iter=300,
                            random_state=42,
                        ),
                    )
                # 4) Temporal model (sequence proxy)
                if ExtraTreesClassifier is not None:
                    _fit_and_score(
                        "temporal_tree",
                        ExtraTreesClassifier(
                            n_estimators=260,
                            max_depth=12,
                            n_jobs=-1,
                            random_state=42,
                        ),
                    )
                # 5) Probabilistic / Bayesian
                if GaussianNB is not None:
                    _fit_and_score("bayes", GaussianNB())
                # Optional experts
                if XGBClassifier is not None:
                    _fit_and_score(
                        "xgb",
                        XGBClassifier(
                            n_estimators=260,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            eval_metric="mlogloss",
                            n_jobs=4,
                        ),
                    )
                if LGBMClassifier is not None:
                    _fit_and_score(
                        "lgbm",
                        LGBMClassifier(
                            n_estimators=260,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            n_jobs=4,
                            verbosity=-1,
                        ),
                    )
                # Optional LSTM temporal expert.
                try:
                    lstm_expert = _TorchLSTMActionModel(input_dim=int(arr_x.shape[1]), actions=list(_ACTIONS))
                    lstm_expert.fit(x_tr, y_tr)
                    pred_l = lstm_expert.predict(x_va)
                    acc_l = float((pred_l == y_va).mean()) if len(y_va) > 0 else 0.5
                    experts["lstm_temporal"] = lstm_expert
                    scores["lstm_temporal"] = max(acc_l, 1e-3)
                except Exception:
                    pass

                if not experts:
                    return
                total = float(sum(scores.values()))
                if total <= 0:
                    total = float(len(scores))
                weights = {k: float(v) / total for k, v in scores.items()}

                bundle = {
                    "experts": experts,
                    "weights": weights,
                    "actions": list(_ACTIONS),
                    "meta": {
                        "samples": int(len(X)),
                        "scores": scores,
                        "trained_at": int(time.time()),
                    },
                }
                import pickle
                _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                with _MODEL_PATH.open("wb") as f:
                    pickle.dump(bundle, f)
                log(f"[AI-TM][TRAIN] ensemble updated, experts={len(experts)} samples={len(X)}")
            except Exception as e:
                log(f"[AI-TM][TRAIN] failed: {e}")

        self._train_future = self._train_pool.submit(_task)

    def run_forever(self) -> None:
        log("[AI-TM] started (async per-symbol scheduler, account-flag controlled)")
        while not self._stop.is_set():
            enabled: Dict[int, dict] = {}
            try:
                enabled = self._enabled_accounts()
                positions = get_open_positions_snapshot() or []
                tick_ts = time.time()
                active_keys = set()
                due_jobs: List[Tuple[int, dict, dict, Optional[float], dict]] = []

                # Build due symbol jobs (global tick is 1s; per-symbol schedule is 5-10s jittered).
                for p in positions:
                    try:
                        ai = int(p.get("account_index"))
                    except Exception:
                        continue
                    if ai not in enabled:
                        continue
                    symbol = str(p.get("symbol") or "")
                    if not symbol:
                        continue
                    amt = _safe_float(p.get("position_amt"), 0.0)
                    if amt == 0.0:
                        continue
                    key = (ai, symbol)
                    active_keys.add(key)
                    settings = enabled.get(ai, {})
                    with self._state_lock:
                        sched = dict(self._symbol_schedule.get(key) or {})
                        prev_ctx = dict(self._ctx.get(key) or {})
                    due_ts = _safe_float(sched.get("due_ts"), 0.0)
                    last_eval_ts = _safe_float(sched.get("last_eval_ts"), 0.0)
                    if due_ts > tick_ts:
                        continue
                    eff_interval = (tick_ts - last_eval_ts) if last_eval_ts > 0 else None
                    due_jobs.append((ai, p, prev_ctx, eff_interval, settings))

                eval_rows: List[dict] = []
                by_account: Dict[int, List[dict]] = {}
                if due_jobs:
                    future_map = {}
                    for ai, p, prev_ctx, eff_interval, settings in due_jobs:
                        fut = self._io_pool.submit(
                            self._build_eval_row_task,
                            int(ai),
                            p,
                            tick_ts,
                            prev_ctx,
                            eff_interval,
                            settings,
                        )
                        future_map[fut] = (int(ai), str(p.get("symbol") or ""))
                    for fut in as_completed(future_map):
                        ai, symbol = future_map[fut]
                        try:
                            row = fut.result()
                        except Exception as e:
                            log(f"[AI-TM][ASYNC][ERROR] account={ai} symbol={symbol} stage=fetch_features err={e}")
                            continue
                        if not isinstance(row, dict):
                            continue
                        key = (int(row["ai"]), str(row["symbol"]))
                        ctx_update = row.get("ctx_update")
                        if isinstance(ctx_update, dict):
                            with self._state_lock:
                                self._ctx[key] = ctx_update
                        if bool(row.get("peak_missing_before")):
                            log(f"[AI-TM][WARN] peak_unreal missing after update account={row['ai']} symbol={row['symbol']}")
                        eval_rows.append(row)
                        by_account.setdefault(int(row["ai"]), []).append(row)

                portfolio_hints: Dict[int, dict] = {}
                for ai, rows in by_account.items():
                    suggested, hint_score, hint_reason, hint_details = self._portfolio_close_advisory(
                        ai,
                        enabled.get(ai, {}),
                        rows,
                    )
                    if suggested:
                        portfolio_hints[ai] = {
                            "score": float(hint_score),
                            "reason": str(hint_reason),
                            "details": hint_details,
                        }
                        log(
                            f"[AI-TM][PORTFOLIO][HINT] account={ai} reason={hint_reason} score={hint_score:.3f} "
                            f"details={hint_details}"
                        )

                # Parallel order-consistency audits.
                order_hints: Dict[Tuple[int, str], dict] = {}
                if eval_rows:
                    audit_futures = {}
                    for row in eval_rows:
                        ai = int(row["ai"])
                        symbol = str(row["symbol"])
                        fut = self._io_pool.submit(
                            self._order_consistency_advisory_cached,
                            acc_idx=ai,
                            symbol=symbol,
                            settings=enabled.get(ai, {}),
                            side=str(row["side"]),
                            entry=float(row["entry"]),
                            current_price=float(row["cur"]),
                            amt_abs=abs(float(row["amt"])),
                        )
                        audit_futures[fut] = (ai, symbol)
                    for fut in as_completed(audit_futures):
                        key = audit_futures[fut]
                        try:
                            hint = fut.result()
                        except Exception:
                            hint = {}
                        if not isinstance(hint, dict):
                            hint = {}
                        order_hints[key] = hint

                for row in eval_rows:
                    ai = int(row["ai"])
                    symbol = str(row["symbol"])
                    key = (ai, symbol)
                    p = row["position"]
                    amt = float(row["amt"])
                    side = str(row["side"])
                    entry = float(row["entry"])
                    cur = float(row["cur"])
                    unreal = float(row["unreal"])
                    features = row["features"]
                    market_state = str(row["market_state"])
                    x = row["x"]
                    x_raw = row.get("x_raw") or x
                    norm_meta = row.get("norm_meta") or {"enabled": False}
                    model_action_raw = row.get("model_action")
                    model_action = str(model_action_raw) if model_action_raw is not None else ""
                    model_conf = float(row["model_conf"])
                    model_raw_conf = _safe_float(row.get("model_raw_conf"), model_conf)
                    model_meta = row.get("model_meta") or {}
                    momentum = _safe_float(row.get("momentum"), 0.0)
                    dd_from_peak = _safe_float(row.get("dd_from_peak"), 0.0)
                    deep_dd_count = _safe_int(row.get("deep_dd_count"), 0)
                    peak_unreal = _safe_float(row.get("peak_unreal"), float("-inf"))
                    data_ready_ts = _safe_float(row.get("data_ready_ts"), time.time())
                    eff_symbol_interval = row.get("effective_symbol_interval_sec")
                    portfolio_hint = portfolio_hints.get(ai)
                    order_hint = order_hints.get(key) or {
                        "suggest": False,
                        "score": 0.0,
                        "reason": "ok",
                        "issues": [],
                    }

                    settings = enabled.get(ai, {})
                    max_mult_cfg = max(_safe_float(settings.get("ai_tm_max_notional_multiplier"), 1.8), 1.0)
                    ctx_now = self._ctx.get(key) or {}
                    base_amt = max(_safe_float(ctx_now.get("base_amt"), abs(amt)), 1e-12)
                    exposure_ratio = min(1.0, max(0.0, abs(amt) / max(base_amt * max_mult_cfg, 1e-12)))
                    atr_pct = abs(_safe_float(features.get("atr"), 0.0) / max(entry, 1e-12)) * 100.0 if entry > 0 else 0.0
                    market_state_code = 1.0 if str(market_state).upper() == "ACTIVE" else 0.0
                    trend_score = self._trend_score(features, side, market_state)
                    mode = self._classify_mode(
                        key=key,
                        settings=settings,
                        market_state=market_state,
                        exposure_ratio=exposure_ratio,
                        dd_from_peak=dd_from_peak,
                        trend_score=trend_score,
                        unreal=unreal,
                    )
                    risk_score = min(
                        1.0,
                        max(
                            0.0,
                            0.45 * exposure_ratio
                            + 0.25 * min(max(dd_from_peak / 1.2, 0.0), 1.0)
                            + 0.20 * (1.0 - market_state_code)
                            + 0.10 * min(max(max(-trend_score, 0.0), 0.0), 1.0),
                        ),
                    )
                    if self._flag_enabled(settings, "ai_tm_credit_linkage_enabled", False):
                        self._flush_decision_credit(
                            key=key,
                            symbol=symbol,
                            cur_unreal=unreal,
                            now_ts=time.time(),
                        )
                    if bool(norm_meta.get("enabled")):
                        nctx = self._ctx.get(key) or {}
                        last_norm_log_ts = _safe_float(nctx.get("last_norm_log_ts"), 0.0)
                        if (time.time() - last_norm_log_ts) >= 300.0:
                            log(
                                f"[AI-TM][NORM] account={ai} symbol={symbol} "
                                f"window={norm_meta.get('window')} warmup={norm_meta.get('warmup')} "
                                f"fields={norm_meta.get('fields')}"
                            )
                            nctx["last_norm_log_ts"] = time.time()
                            self._ctx[key] = nctx
                    exec_lock = self._get_symbol_lock(key)
                    conflict_evt = {
                        "sl_scale_conflict_detected": False,
                        "conflict_resolution_action": "",
                        "scale_orders_cancelled": 0,
                    }
                    lock_wait_start = time.time()
                    with exec_lock:
                        queue_delay_ms = max(0.0, (time.time() - lock_wait_start) * 1000.0)
                        conflict_evt = self._enforce_sl_scale_conflict(
                            acc_idx=ai,
                            symbol=symbol,
                            settings=settings,
                            long_side=(side == "LONG"),
                            entry=entry,
                            position_side=p.get("position_side"),
                            market_state=market_state,
                            trend_score=float(trend_score),
                            risk_score=float(risk_score),
                            ml_conf=float(model_conf),
                            unreal_pnl_pct=float(unreal),
                            atr_pct=float(atr_pct),
                        )
                        guard_evt = self._enforce_minute_protection_guard(
                            acc_idx=ai,
                            symbol=symbol,
                            settings=settings,
                            side=side,
                            entry=entry,
                            cur=cur,
                            amt_abs=abs(amt),
                            position_side=p.get("position_side"),
                            atr=_safe_float(features.get("atr"), 0.0),
                        )
                        if isinstance(guard_evt, dict):
                            log(
                                f"[AI-TM][GUARD] account={ai} symbol={symbol} "
                                f"sl_placed={guard_evt.get('sl_placed')} "
                                f"tp_placed={guard_evt.get('tp_placed')} "
                                f"recovery_tp_placed={guard_evt.get('recovery_tp_placed')} "
                                f"extra_qty={guard_evt.get('extra_qty')} "
                                f"recovery_tp_price={guard_evt.get('recovery_tp_price')} "
                                f"recovery_tp_qty={guard_evt.get('recovery_tp_qty')}"
                            )
                        recovery_evt = self._maintain_scale_in_recovery(
                            acc_idx=ai,
                            symbol=symbol,
                            settings=settings,
                            long_side=(side == "LONG"),
                            entry=entry,
                            cur=cur,
                            amt_abs=abs(amt),
                            position_side=p.get("position_side"),
                        )
                        if isinstance(recovery_evt, dict):
                            log(
                                f"[AI-TM][SCALE-IN][RECOVERY] account={ai} symbol={symbol} "
                                f"status={recovery_evt.get('status')} reason={recovery_evt.get('reason')} "
                                f"target={recovery_evt.get('target_px')} qty={recovery_evt.get('qty')}"
                            )
                            if bool(recovery_evt.get("skip_decision")):
                                continue
                        be_lock_evt = self._enforce_tranche_breakeven_lock(
                            acc_idx=ai,
                            symbol=symbol,
                            settings=settings,
                            long_side=(side == "LONG"),
                            entry=entry,
                            cur=cur,
                            position_side=p.get("position_side"),
                        )
                        if isinstance(be_lock_evt, dict):
                            log(
                                f"[AI-TM][BREAKEVEN][LOCK] account={ai} symbol={symbol} "
                                f"locked={be_lock_evt.get('breakeven_locked')} "
                                f"sl={be_lock_evt.get('protective_sl_price')} "
                                f"tranches={be_lock_evt.get('tranche_ids')}"
                            )

                        override_reason = None
                        min_profit_lock_pct = max(0.30, _safe_float(settings.get("ai_tm_min_profit_lock_pct"), 2.0))
                        if mode == "aggressive_trend_mode":
                            min_profit_lock_pct = float(min_profit_lock_pct * 1.20)
                        else:
                            min_profit_lock_pct = float(min_profit_lock_pct * 0.90)
                        override_events: List[str] = []
                        ml_first_enabled = self._flag_enabled(settings, "ai_tm_ml_first_enabled", False)
                        if self._flag_enabled(settings, "ai_tm_ml_strict_mode", True):
                            ml_first_enabled = True
                        if model_conf <= 0.0 or not model_action:
                            action = "HOLD"
                            conf = 0.0
                            override_reason = "model_unavailable_fallback"
                            override_events.append("model_unavailable_fallback")
                            log(
                                f"[AI-TM][FALLBACK] account={ai} symbol={symbol} "
                                f"model_action={model_action_raw} model_conf={model_conf}"
                            )
                        elif portfolio_hint and model_action != "CLOSE_TRADE":
                            action = "CLOSE_TRADE"
                            conf = max(float(model_conf), float(portfolio_hint.get("score") or 0.0))
                            override_reason = "portfolio_close_hint"
                        else:
                            action, conf, override_reason = self._prefer_profit_protection_action(
                                ai,
                                symbol,
                                settings,
                                side,
                                market_state,
                                _safe_float(features.get("momentum"), 0.0),
                                _safe_float(features.get("acceleration"), 0.0),
                                model_action,
                                model_conf,
                                unreal,
                                entry,
                                cur,
                            )
                            if bool(order_hint.get("suggest")) and action == "HOLD":
                                action = "HOLD"
                                conf = max(float(conf), float(order_hint.get("score") or 0.0))
                                override_reason = "order_consistency_reconcile"

                        # Candidate override map (used by arbiter when ML-first is enabled).
                        candidate_action = action
                        candidate_reason = override_reason or "candidate_from_policy"
                        critical_override = False
                        max_dd_cfg = abs(_safe_float(settings.get("ai_tm_max_drawdown_pct"), 6.0))
                        missing_protection_severe = bool(order_hint.get("suggest")) and (
                            "sl_not_covering_position" in (order_hint.get("issues") or [])
                        )
                        risk_immediate = self._immediate_loss_risk(
                            deep_dd_count=deep_dd_count,
                            unreal=unreal,
                            max_dd_cfg=max_dd_cfg,
                            has_missing_protection=missing_protection_severe,
                        )
                        protection_score = self._giveback_risk(
                            peak_unreal=peak_unreal,
                            dd_from_peak=dd_from_peak,
                            market_state=market_state,
                            momentum=momentum,
                            side=side,
                        )
                        if deep_dd_count >= 3:
                            candidate_action = "CLOSE_TRADE"
                            candidate_reason = "forced_drawdown_close"
                            critical_override = True
                            override_events.append("forced_drawdown_close")
                        elif unreal <= -max_dd_cfg:
                            candidate_action = "CLOSE_TRADE"
                            candidate_reason = "max_drawdown_breach"
                            critical_override = True
                            override_events.append("max_drawdown_breach")
                        elif missing_protection_severe:
                            candidate_action = "MOVE_SL"
                            candidate_reason = "missing_protection_severe"
                            critical_override = True
                            override_events.append("missing_protection_severe")
                        elif peak_unreal >= 1.0 and dd_from_peak >= 0.7:
                            candidate_action = "MOVE_SL"
                            candidate_reason = "hard_profit_protection"
                            override_events.append("hard_profit_protection")
                        elif candidate_action == "SCALE_IN" and (market_state == "PASSIVE" or momentum < 0.0):
                            candidate_action = "HOLD"
                            candidate_reason = "scale_in_blocked_bad_conditions"
                            override_events.append("scale_in_blocked_bad_conditions")

                        if candidate_action == "HOLD" and unreal >= min_profit_lock_pct:
                            ctx_for_hold = self._ctx.get(key) or {}
                            if (time.time() - _safe_float(ctx_for_hold.get("last_sl_move_ts"), 0.0)) > 600.0:
                                candidate_action = "MOVE_SL"
                                candidate_reason = "hold_demoted_to_sl"
                                override_events.append("hold_demoted_to_sl")
                        if candidate_action == "HOLD" and unreal >= (3.0 if mode == "aggressive_trend_mode" else 2.2):
                            candidate_action = "MOVE_TP"
                            candidate_reason = "auto_tp_activation"
                            override_events.append("auto_tp_activation")

                        if ml_first_enabled:
                            margin = _safe_float(settings.get("ai_tm_override_margin"), 0.15)
                            action, resolve_reason = self._resolve_final_action(
                                ml_action=model_action,
                                ml_conf=float(conf),
                                override_meta={
                                    "candidate_action": candidate_action,
                                    "candidate_reason": candidate_reason,
                                    "critical_risk": critical_override,
                                },
                                risk_score=risk_immediate,
                                protection_score=protection_score,
                                margin=margin,
                            )
                            if action != model_action:
                                conf = max(float(conf), 0.70 if critical_override else 0.55)
                            override_reason = resolve_reason if action != model_action else None
                        else:
                            action = candidate_action
                            override_reason = candidate_reason if action != model_action else override_reason

                        override_flag = bool(action != model_action)
                        override_impact_est = float(unreal - peak_unreal) if peak_unreal > float("-inf") else 0.0
                        if override_flag and self._flag_enabled(settings, "ai_tm_override_transparency_enabled", False):
                            _append_jsonl(
                                _DECISION_LOG_PATH,
                                {
                                    "event": "override_audit",
                                    "timestamp": int(time.time()),
                                    "account_index": int(ai),
                                    "symbol": str(symbol),
                                    "ml_action": str(model_action),
                                    "ml_conf": float(model_conf),
                                    "final_action": str(action),
                                    "override_reason": str(override_reason),
                                    "risk_score": float(risk_score),
                                    "immediate_loss_risk_score": float(risk_immediate),
                                    "protection_score": float(protection_score),
                                    "estimated_pnl_impact": float(override_impact_est),
                                },
                            )
                            log(
                                f"[AI-TM][OVERRIDE][SUPPRESS] account={ai} symbol={symbol} "
                                f"ml={model_action}/{model_conf:.3f} final={action}/{conf:.3f} "
                                f"reason={override_reason} risk_score={risk_score:.3f} impact_est={override_impact_est:.4f}"
                            )
                        for evt in override_events:
                            log(
                                f"[AI-TM][OVERRIDE] account={ai} symbol={symbol} event={evt} "
                                f"model={model_action}/{model_conf:.3f} action={action} unreal={unreal:.3f} "
                                f"peak={peak_unreal:.3f} dd_from_peak={dd_from_peak:.3f} deep_dd_count={deep_dd_count}"
                            )

                        decision_ts = time.time()
                        decision_latency_ms = max(0.0, (decision_ts - data_ready_ts) * 1000.0)

                        self._metrics["hold_count"] += int(action == "HOLD")
                        self._metrics["move_sl_count"] += int(action == "MOVE_SL")
                        self._metrics["close_count"] += int(action == "CLOSE_TRADE")
                        self._metrics["total_decisions"] += 1
                        if peak_unreal > 0.0 and unreal < 0.0:
                            self._metrics["giveback_events"] += 1
                            if _safe_float((self._ctx.get(key) or {}).get("last_sl_move_ts"), 0.0) <= 0.0:
                                self._metrics["giveback_no_sl_events"] += 1

                        exec_start_ts = time.time()
                        action, out = self._apply_action(
                            ai,
                            symbol,
                            p,
                            settings,
                            action,
                            {**features, "last_price": cur, "market_state": market_state},
                            decision_meta={
                                "model_action": model_action,
                                "model_confidence": model_conf,
                                "model_raw_confidence": model_raw_conf,
                                "override_reason": override_reason,
                                "portfolio_hint": portfolio_hint,
                                "order_hint": order_hint,
                                "mode": mode,
                                "risk_score": float(risk_score),
                                "trend_score": float(trend_score),
                            },
                        )
                        execution_latency_ms = max(0.0, (time.time() - exec_start_ts) * 1000.0)
                        api_latency_ms = _safe_float(out.get("api_latency_ms"), 0.0) if isinstance(out, dict) else 0.0
                        if execution_latency_ms > 3000.0:
                            log(
                                f"[AI-TM][LATENCY][CRITICAL] account={ai} symbol={symbol} "
                                f"execution_latency_ms={execution_latency_ms:.1f} api_latency_ms={api_latency_ms:.1f} "
                                f"queue_delay_ms={queue_delay_ms:.1f}"
                            )
                        tp_before = None
                        tp_after = None
                        try:
                            dd = out.get("details") if isinstance(out, dict) else {}
                            if isinstance(dd, dict):
                                tp_before = _safe_float(dd.get("recovery_tp_entry_used"), 0.0)
                                tp_after = _safe_float(dd.get("recovery_tp_price"), 0.0)
                            if not tp_before or tp_before <= 0.0:
                                tp_before = _safe_float(entry, 0.0)
                            if (not tp_after or tp_after <= 0.0) and _safe_float(out.get("new_tp"), 0.0) > 0:
                                tp_after = _safe_float(out.get("new_tp"), 0.0)
                        except Exception:
                            tp_before = _safe_float(entry, 0.0)
                            tp_after = _safe_float(out.get("new_tp"), 0.0) if isinstance(out, dict) else 0.0
                        dev_pct = (
                            float(abs(tp_after - tp_before) / max(abs(tp_before), 1e-12) * 100.0)
                            if tp_before and tp_before > 0 and tp_after and tp_after > 0
                            else 0.0
                        )
                        log(
                            f"[AI-TM][ML_FLOW] account={ai} symbol={symbol} "
                            f"ml_action={model_action} ml_conf={model_conf:.3f} "
                            f"final_action={action} tp_before={tp_before} tp_after={tp_after} "
                            f"deviation_pct={dev_pct:.6f}"
                        )

                    now_decision_ts = time.time()
                    with self._state_lock:
                        lc = self._lifecycle.get(key) or {}
                        if not lc:
                            lc = {
                                "id": f"{int(now_decision_ts*1000)}:{ai}:{symbol}",
                                "start_ts": float(now_decision_ts),
                                "entry_unreal": float(unreal),
                                "decision_seq": 0,
                                "max_favorable_unreal": float(unreal),
                                "max_adverse_unreal": float(unreal),
                                "peak_unreal": float(unreal),
                                "dd_path": [],
                            }
                        lc["decision_seq"] = _safe_int(lc.get("decision_seq"), 0) + 1
                        lc["max_favorable_unreal"] = max(_safe_float(lc.get("max_favorable_unreal"), unreal), float(unreal))
                        lc["max_adverse_unreal"] = min(_safe_float(lc.get("max_adverse_unreal"), unreal), float(unreal))
                        lc["peak_unreal"] = max(_safe_float(lc.get("peak_unreal"), unreal), float(unreal))
                        dd_now = _safe_float(lc.get("peak_unreal"), unreal) - float(unreal)
                        dd_path = lc.get("dd_path") if isinstance(lc.get("dd_path"), list) else []
                        dd_path.append({"ts": float(now_decision_ts), "dd_from_peak": float(dd_now)})
                        if len(dd_path) > 240:
                            dd_path = dd_path[-240:]
                        lc["dd_path"] = dd_path
                        self._lifecycle[key] = lc
                    decision_id = f"{int(now_decision_ts*1000)}:{ai}:{symbol}"
                    lifecycle_id = str(lc.get("id"))
                    decision_seq = _safe_int(lc.get("decision_seq"), 0)
                    tranche_info = {}
                    try:
                        details_obj = out.get("details") if isinstance(out, dict) else {}
                        if isinstance(details_obj, dict):
                            tr = details_obj.get("tranche")
                            if isinstance(tr, dict):
                                tranche_info = tr
                    except Exception:
                        tranche_info = {}
                    if not tranche_info:
                        try:
                            cctx = self._ctx.get(key) or {}
                            arr_tr = cctx.get("scale_in_tranches")
                            if isinstance(arr_tr, list) and arr_tr:
                                # Use the newest known tranche for audit snapshot.
                                tranche_info = arr_tr[-1] if isinstance(arr_tr[-1], dict) else {}
                        except Exception:
                            tranche_info = {}
                    _append_jsonl(
                        _DECISION_LOG_PATH,
                        {
                            "decision_id": decision_id,
                            "timestamp": int(now_decision_ts),
                            "account_index": ai,
                            "symbol": symbol,
                            "model_decision": model_action,
                            "model_confidence": float(model_conf),
                            "ml_action_raw": model_action,
                            "ml_conf_raw": float(model_raw_conf),
                            "decision": action,
                            "final_action": action,
                            "ai_confidence": float(conf),
                            "override_flag": bool(action != model_action),
                            "override_reason": override_reason,
                            "portfolio_hint": portfolio_hint,
                            "order_hint": order_hint,
                            "features_vector": x,
                            "features_vector_raw": x_raw,
                            "normalization_meta": norm_meta,
                            "market_state": market_state,
                            "market_state_code": market_state_code,
                            "new_sl": out.get("new_sl"),
                            "new_tp": out.get("new_tp"),
                            "scale_qty": out.get("scale_qty"),
                            "status": out.get("status"),
                            "reason": out.get("reason"),
                            "details": out.get("details"),
                            "position_amt": float(amt),
                            "unreal_pnl_pct": float(unreal),
                            "peak_unreal": float(peak_unreal),
                            "dd_from_peak": float(dd_from_peak),
                            "deep_dd_count": int(deep_dd_count),
                            "position_lifecycle_id": lifecycle_id,
                            "decision_seq": int(decision_seq),
                            "override_meta": {
                                "override_reason": override_reason,
                                "override_events": override_events,
                                "override_impact_est": override_impact_est,
                            },
                            "position_notional": float(abs(amt) * max(entry, 0.0)),
                            "exposure_ratio": float(exposure_ratio),
                            "risk_score": float(risk_score),
                            "mode": mode,
                            "trend_score": float(trend_score),
                            "tranche_id": tranche_info.get("tranche_id"),
                            "tranche_qty": tranche_info.get("qty"),
                            "tranche_entry_price": tranche_info.get("entry_price"),
                            "tranche_tp_price": tranche_info.get("tp_price"),
                            "tranche_tp_policy": tranche_info.get("tp_policy"),
                            "tranche_tp_order_id": tranche_info.get("tp_order_id"),
                            "breakeven_locked": bool(tranche_info.get("breakeven_locked")) if isinstance(tranche_info, dict) else False,
                            "protective_sl_price": tranche_info.get("protective_sl_price") if isinstance(tranche_info, dict) else None,
                            "sl_scale_conflict_detected": bool((conflict_evt or {}).get("sl_scale_conflict_detected")),
                            "conflict_resolution_action": str((conflict_evt or {}).get("conflict_resolution_action") or ""),
                            "scale_orders_cancelled": int(_safe_int((conflict_evt or {}).get("scale_orders_cancelled"), 0)),
                            "priority_mode": str((conflict_evt or {}).get("priority_mode") or ""),
                            "sl_adjustment_reason": str((conflict_evt or {}).get("sl_adjustment_reason") or ""),
                            "scale_in_blocked": bool(
                                (conflict_evt or {}).get("scale_in_blocked")
                                or (
                                    isinstance(out.get("details"), dict)
                                    and bool((out.get("details") or {}).get("scale_in_blocked"))
                                )
                            ),
                            "volatility_buffer_pct": (
                                float((conflict_evt or {}).get("volatility_buffer_pct"))
                                if (conflict_evt or {}).get("volatility_buffer_pct") is not None
                                else None
                            ),
                            "decision_latency_ms": float(decision_latency_ms),
                            "execution_latency_ms": float(execution_latency_ms),
                            "api_latency_ms": float(api_latency_ms),
                            "queue_delay_ms": float(queue_delay_ms),
                            "effective_symbol_interval_sec": (
                                float(eff_symbol_interval)
                                if isinstance(eff_symbol_interval, (int, float))
                                else None
                            ),
                        },
                    )
                    self._register_decision_credit_event(
                        key=key,
                        lifecycle_id=lifecycle_id,
                        decision_id=decision_id,
                        decision_seq=decision_seq,
                        symbol=symbol,
                        account_index=ai,
                        ts=now_decision_ts,
                        unreal_pnl_pct=unreal,
                        ml_action=model_action,
                        final_action=action,
                        conf=float(conf),
                        features_vector=x,
                        expert_votes=(model_meta or {}).get("expert_votes") or {},
                    )
                    ctx = self._ctx.get(key) or {}
                    ctx["last_decision_id"] = decision_id

                    should_emit = True
                    is_hold_noop = bool(action == "HOLD" and str(out.get("reason") or "") == "model_hold_or_noop")
                    if is_hold_noop:
                        hold_sig = (
                            str(model_action),
                            str(market_state),
                            str(out.get("status") or ""),
                            str(out.get("reason") or ""),
                        )
                        last_sig = ctx.get("last_hold_log_sig")
                        last_unreal = _safe_float(ctx.get("last_hold_log_unreal"), float(unreal))
                        last_conf = _safe_float(ctx.get("last_hold_log_conf"), float(model_conf))
                        last_ts = _safe_float(ctx.get("last_hold_log_ts"), 0.0)
                        keepalive_sec = max(
                            300.0,
                            _safe_float(enabled.get(ai, {}).get("ai_tm_hold_log_keepalive_sec"), 900.0),
                        )
                        now_ts = time.time()
                        unreal_delta = abs(float(unreal) - float(last_unreal))
                        conf_delta = abs(float(model_conf) - float(last_conf))
                        state_changed = hold_sig != last_sig
                        meaningful_change = state_changed or unreal_delta >= 0.35 or conf_delta >= 0.05
                        if (not meaningful_change) and (now_ts - last_ts) < keepalive_sec:
                            should_emit = False
                        else:
                            ctx["last_hold_log_sig"] = hold_sig
                            ctx["last_hold_log_unreal"] = float(unreal)
                            ctx["last_hold_log_conf"] = float(model_conf)
                            ctx["last_hold_log_ts"] = now_ts
                    else:
                        ctx.pop("last_hold_log_sig", None)
                        ctx.pop("last_hold_log_unreal", None)
                        ctx.pop("last_hold_log_conf", None)
                        ctx.pop("last_hold_log_ts", None)

                    self._ctx[key] = ctx
                    if should_emit:
                        log(
                            f"[AI-TM] account={ai} symbol={symbol} model={model_action}/{model_conf:.3f} "
                            f"decision={action} conf={conf:.3f} override={override_reason} "
                            f"state={market_state} unreal={unreal:.2f}% sl={out.get('new_sl')} tp={out.get('new_tp')} "
                            f"scale={out.get('scale_qty')} status={out.get('status')} reason={out.get('reason')} "
                            f"mode={mode} risk_score={risk_score:.3f} exposure_ratio={exposure_ratio:.3f} "
                            f"decision_latency_ms={decision_latency_ms:.1f} execution_latency_ms={execution_latency_ms:.1f} "
                            f"api_latency_ms={api_latency_ms:.1f} queue_delay_ms={queue_delay_ms:.1f} "
                            f"effective_symbol_interval_sec={eff_symbol_interval} details={out.get('details')}"
                        )

                active_symbols_by_account: Dict[int, set] = {}
                for ai, sym in active_keys:
                    active_symbols_by_account.setdefault(int(ai), set()).add(str(sym).upper())
                for ai in enabled.keys():
                    orphan_evt = self._cleanup_orphan_orders_for_account(
                        acc_idx=int(ai),
                        active_symbols=active_symbols_by_account.get(int(ai), set()),
                        settings=enabled.get(int(ai), {}),
                    )
                    if isinstance(orphan_evt, dict) and _safe_int(orphan_evt.get("cancelled_count"), 0) > 0:
                        log(
                            f"[AI-TM][ORPHAN][CLEANUP] account={ai} checked={orphan_evt.get('checked')} "
                            f"cancelled={orphan_evt.get('cancelled_count')}"
                        )

                now_sched = time.time()
                with self._state_lock:
                    for ai, p, _prev_ctx, _eff, _settings in due_jobs:
                        symbol = str(p.get("symbol") or "")
                        key = (int(ai), symbol)
                        settings = enabled.get(int(ai), {})
                        self._symbol_schedule[key] = {
                            "last_eval_ts": float(now_sched),
                            "target_interval_sec": float(self._symbol_interval_target_sec(settings)),
                            "due_ts": float(now_sched + self._symbol_interval_jittered_sec(settings)),
                        }

                # finalize stale/closed positions
                stale = [k for k in list(self._ctx.keys()) if k not in active_keys]
                for k in stale:
                    ctx = self._ctx.get(k) or {}
                    did = ctx.get("last_decision_id")
                    lc = self._lifecycle.get(k) or {}
                    lifecycle_id = str(lc.get("id") or "")
                    max_fav = _safe_float(lc.get("max_favorable_unreal"), _safe_float(ctx.get("unreal_pnl_pct"), 0.0))
                    max_adv = _safe_float(lc.get("max_adverse_unreal"), _safe_float(ctx.get("unreal_pnl_pct"), 0.0))
                    peak = _safe_float(lc.get("peak_unreal"), _safe_float(ctx.get("unreal_pnl_pct"), 0.0))
                    final_unreal = _safe_float(ctx.get("unreal_pnl_pct"), 0.0)
                    realized_delta = final_unreal - _safe_float(lc.get("entry_unreal"), 0.0)
                    _append_jsonl(
                        _OUTCOME_LOG_PATH,
                        {
                            "decision_id": did,
                            "timestamp": int(time.time()),
                            "account_index": int(k[0]),
                            "symbol": str(k[1]),
                            "position_lifecycle_id": lifecycle_id,
                            "final_unreal_pnl_pct": final_unreal,
                            "realized_outcome_delta_unreal_pct": float(realized_delta),
                            "max_favorable_unreal_pnl_pct": float(max_fav),
                            "max_adverse_unreal_pnl_pct": float(max_adv),
                            "drawdown_from_peak_final_pct": float(max(0.0, peak - final_unreal)),
                            "drawdown_path": lc.get("dd_path") if isinstance(lc.get("dd_path"), list) else [],
                            "lifetime_sec": max(0.0, time.time() - _safe_float(ctx.get("entry_ts"), time.time())),
                        },
                    )
                    try:
                        self._last_close_ts[(int(k[0]), str(k[1]))] = float(time.time())
                        self._update_perf_stats(float(realized_delta))
                    except Exception:
                        pass
                    _append_jsonl(
                        _LIFECYCLE_LOG_PATH,
                        {
                            "event": "lifecycle_close",
                            "position_lifecycle_id": lifecycle_id,
                            "account_index": int(k[0]),
                            "symbol": str(k[1]),
                            "closed_ts": int(time.time()),
                            "entry_unreal_pnl_pct": _safe_float(lc.get("entry_unreal"), 0.0),
                            "final_unreal_pnl_pct": float(final_unreal),
                            "realized_outcome_delta_unreal_pct": float(realized_delta),
                            "max_favorable_unreal_pnl_pct": float(max_fav),
                            "max_adverse_unreal_pnl_pct": float(max_adv),
                            "drawdown_from_peak_final_pct": float(max(0.0, peak - final_unreal)),
                            "decision_count": _safe_int(lc.get("decision_seq"), 0),
                        },
                    )
                    self._ctx.pop(k, None)
                    self._lifecycle.pop(k, None)
                    self._pending_credit.pop(k, None)
                    with self._state_lock:
                        self._symbol_schedule.pop(k, None)
                        self._symbol_locks.pop(k, None)

                # TTL cleanup
                now = time.time()
                ttl_stale = []
                for k, v in self._ctx.items():
                    last_ts = _safe_float(v.get("last_ts"), now)
                    if (now - last_ts) > _CACHE_TTL_SEC:
                        ttl_stale.append(k)
                for k in ttl_stale:
                    self._ctx.pop(k, None)
                    self._lifecycle.pop(k, None)
                    self._pending_credit.pop(k, None)
                    with self._state_lock:
                        self._symbol_schedule.pop(k, None)
                        self._symbol_locks.pop(k, None)

                now_kpi = time.time()
                last_kpi = _safe_float(self._metrics.get("last_kpi_log_ts"), 0.0)
                total_decisions = max(_safe_float(self._metrics.get("total_decisions"), 0.0), 1.0)
                if (now_kpi - last_kpi) >= 3600.0 and _safe_float(self._metrics.get("total_decisions"), 0.0) > 0:
                    hold_ratio = _safe_float(self._metrics.get("hold_count"), 0.0) / total_decisions
                    sl_ratio = _safe_float(self._metrics.get("move_sl_count"), 0.0) / total_decisions
                    close_ratio = _safe_float(self._metrics.get("close_count"), 0.0) / total_decisions
                    log(f"[AITM KPI] HOLD={hold_ratio:.2f} SL={sl_ratio:.2f} CLOSE={close_ratio:.2f}")
                    giveback_events = _safe_float(self._metrics.get("giveback_events"), 0.0)
                    giveback_no_sl = _safe_float(self._metrics.get("giveback_no_sl_events"), 0.0)
                    giveback_no_sl_ratio = (giveback_no_sl / giveback_events) if giveback_events > 0 else 0.0
                    if hold_ratio > 0.90:
                        log(f"[AITM KPI][ALERT] HOLD ratio too high: {hold_ratio:.2%}")
                    if giveback_events > 0 and giveback_no_sl_ratio > 0.30:
                        log(
                            f"[AITM KPI][ALERT] giveback_without_sl ratio too high: "
                            f"{giveback_no_sl_ratio:.2%} ({int(giveback_no_sl)}/{int(giveback_events)})"
                        )
                    _append_jsonl(
                        _CALIBRATION_LOG_PATH,
                        {
                            "timestamp": int(now_kpi),
                            "conf_calibration": self._conf_calibration,
                            "symbol_expert_perf": self._symbol_expert_perf,
                        },
                    )
                    self._metrics["last_kpi_log_ts"] = now_kpi

                self._train_model_if_due()
            except Exception as e:
                log(f"[AI-TM] loop error: {e}")

            if any(self._flag_enabled(v, "ai_tm_fast_loop_enabled", False) for v in (enabled or {}).values()):
                self._stop.wait(timeout=1.0)
            else:
                self._stop.wait(timeout=max(1.0, float(_RECHECK_SEC)))


def run_ai_trade_manager(get_accounts_cfg_live) -> None:
    mgr = AITradeManager(get_accounts_cfg_live=get_accounts_cfg_live)
    mgr.run_forever()

