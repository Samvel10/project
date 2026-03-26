import json
import math
import os
import time
import atexit
import fcntl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import yaml as _pyyaml
except Exception:
    _pyyaml = None

from monitoring.telegram import send_telegram
from monitoring.subscribers import get_subscribers, remove_subscriber


BINANCE_FAPI_BASE = "https://fapi.binance.com"
DEFAULT_CONFIG_PATH = Path("config/market_regime.yaml")
DEFAULT_TRADING_CONFIG_PATH = Path("config/trading.yaml")
DEFAULT_PID_PATH = Path("data/market_regime.pid")
DEFAULT_DISPATCH_LOCK_PATH = Path("data/market_regime.dispatch.lock")


def _now_ts() -> float:
    return time.time()


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _load_yaml(path: Path) -> Dict[str, Any]:
    if _pyyaml is None:
        return {}
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = _pyyaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _acquire_singleton_pidfile(pid_path: Path) -> bool:
    """Ensure only one market_regime process is active."""
    try:
        if not pid_path.parent.exists():
            pid_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return True

    current_pid = os.getpid()
    existing_pid = None
    try:
        if pid_path.exists():
            raw = pid_path.read_text(encoding="utf-8").strip()
            if raw:
                existing_pid = int(raw)
    except Exception:
        existing_pid = None

    if existing_pid and existing_pid != current_pid and _pid_is_alive(existing_pid):
        print(f"[MARKET_REGIME] Another instance is already running (pid={existing_pid}); exiting.")
        return False

    try:
        pid_path.write_text(str(current_pid), encoding="utf-8")
    except Exception:
        return True

    def _cleanup_pidfile() -> None:
        try:
            raw = pid_path.read_text(encoding="utf-8").strip()
            if raw and int(raw) == current_pid:
                pid_path.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_cleanup_pidfile)
    return True


def _default_cfg() -> Dict[str, Any]:
    return {
        "enabled": True,
        "interval_minutes": 30,
        "telegram": {
            "token": "",
            "chat_id": 0,
            "only_on_regime_change": False,
            "send_startup_message": False,
            "send_first_analysis_on_startup": True,
            "broadcast_to_subscribers": True,
            "include_emoji": True,
            "detailed_message": True,
        },
        "market": {
            "base_symbol": "BTCUSDT",
            "ai_symbols": ["WLDUSDT", "FETUSDT", "AGIXUSDT", "AIUSDT", "ARKMUSDT", "NFPUSDT"],
            "analysis_symbols_per_cycle": 35,
            "microstructure_symbols_per_cycle": 25,
            "correlation_symbols_per_cycle": 20,
        },
        "thresholds": {
            "btc_volume_ratio_active": 1.5,
            "ai_volume_ratio_active": 1.3,
            "ai_oi_change_pct_active": 0.7,
            "spread_pct_max": 0.04,
            "top5_depth_usdt_min": 250000.0,
            "slippage_bps_max": 6.0,
            "correlation_min": 0.35,
            "score_active": 0.60,
            "score_quiet": 0.40,
        },
        "weights": {
            "volume": 0.4,
            "oi": 0.3,
            "microstructure": 0.2,
            "correlation": 0.1,
        },
        "windows": {
            "volume_5m_lookback_candles": 288,
            "correlation_window_minutes": 30,
        },
        "limits": {
            "slippage_estimate_notional_usdt": 1000.0,
        },
        "runtime": {
            "request_timeout_sec": 10,
            "data_path": "data/market_regime_state.json",
            "pid_path": str(DEFAULT_PID_PATH),
            "dispatch_lock_path": str(DEFAULT_DISPATCH_LOCK_PATH),
        },
    }


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    den = math.sqrt(den_x * den_y)
    if den <= 0.0:
        return None
    return num / den


def _returns_from_closes(closes: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0.0:
            continue
        out.append((cur - prev) / prev)
    return out


class BinancePublicApi:
    def __init__(self, timeout_sec: float = 10.0):
        self.timeout_sec = float(timeout_sec)

    def _get(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{BINANCE_FAPI_BASE}{path}"
        resp = requests.get(url, params=params, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.json()

    def klines(self, symbol: str, interval: str, limit: int) -> List[List[Any]]:
        # Supports large historical limits via pagination (Binance max 1500/request).
        symbol = str(symbol).upper()
        remaining = max(1, int(limit))
        chunks_rev: List[List[List[Any]]] = []
        end_time: Optional[int] = None
        while remaining > 0:
            batch = min(1500, remaining)
            params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": int(batch)}
            if end_time is not None:
                params["endTime"] = int(end_time)
            data = self._get("/fapi/v1/klines", params)
            if not isinstance(data, list) or not data:
                break
            chunks_rev.append(data)
            remaining -= len(data)
            first_open = _safe_float(data[0][0], 0.0)
            if first_open <= 0.0:
                break
            end_time = int(first_open) - 1
            if len(data) < batch:
                break

        if not chunks_rev:
            return []

        merged: List[List[Any]] = []
        for chunk in reversed(chunks_rev):
            merged.extend(chunk)
        return merged[-int(limit):]

    def open_interest_hist_5m(self, symbol: str, limit: int = 2) -> List[Dict[str, Any]]:
        symbol = str(symbol).upper()
        data = self._get("/futures/data/openInterestHist", {"symbol": symbol, "period": "5m", "limit": int(limit)})
        return data if isinstance(data, list) else []

    def book_ticker(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol).upper()
        data = self._get("/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        return data if isinstance(data, dict) else {}

    def depth(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        symbol = str(symbol).upper()
        data = self._get("/fapi/v1/depth", {"symbol": symbol, "limit": int(limit)})
        return data if isinstance(data, dict) else {}

    def exchange_info(self) -> Dict[str, Any]:
        data = self._get("/fapi/v1/exchangeInfo", {})
        return data if isinstance(data, dict) else {}


class ConfigManager:
    def __init__(self, cfg_path: Path, trading_cfg_path: Path):
        self.cfg_path = cfg_path
        self.trading_cfg_path = trading_cfg_path
        self._mtime: Optional[float] = None
        self._cfg: Dict[str, Any] = _default_cfg()

    def load_live(self) -> Dict[str, Any]:
        mtime = None
        try:
            if self.cfg_path.exists():
                mtime = self.cfg_path.stat().st_mtime
        except Exception:
            mtime = None

        if self._mtime is not None and mtime is not None and float(self._mtime) == float(mtime):
            return self._cfg

        base = _default_cfg()
        user = _load_yaml(self.cfg_path)
        merged = _deep_merge(base, user)

        # Telegram fallback to trading.yaml / env if local config is empty.
        tele = merged.get("telegram") if isinstance(merged.get("telegram"), dict) else {}
        token = str(tele.get("token") or "").strip()
        chat_id = _safe_float(tele.get("chat_id"), 0.0)
        if not token or int(chat_id) <= 0:
            tcfg = _load_yaml(self.trading_cfg_path)
            token2 = str(tcfg.get("telegram_token") or os.environ.get("TELEGRAM_TOKEN") or "").strip()
            chat_raw = tcfg.get("telegram_chat_id") or os.environ.get("TELEGRAM_CHAT_ID")
            chat2 = int(_safe_float(chat_raw, 0.0))
            if token2 and chat2 > 0:
                tele = dict(tele)
                if not token:
                    tele["token"] = token2
                if int(chat_id) <= 0:
                    tele["chat_id"] = chat2
                merged["telegram"] = tele

        self._cfg = merged
        self._mtime = mtime
        return self._cfg


@dataclass
class RegimeSnapshot:
    regime: str
    score: float
    confidence: float
    btc_volume_ratio: float
    ai_volume_ratio: float
    ai_oi_change_pct: float
    avg_spread_pct: float
    avg_depth_usdt: float
    slippage_bps: float
    correlation: float
    ts: float


class MarketRegimeDetector:
    def __init__(self, cfg_mgr: ConfigManager):
        self.cfg_mgr = cfg_mgr
        cfg = self.cfg_mgr.load_live()
        timeout_sec = _safe_float(((cfg.get("runtime") or {}).get("request_timeout_sec")), 10.0)
        self.api = BinancePublicApi(timeout_sec=timeout_sec)
        self._state = self._load_state(cfg)
        self._universe_cache: List[str] = []
        self._universe_cache_ts: float = 0.0

    def _state_path(self, cfg: Dict[str, Any]) -> Path:
        runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
        p = runtime.get("data_path") or "data/market_regime_state.json"
        return Path(str(p))

    def _dispatch_lock_path(self, cfg: Dict[str, Any]) -> Path:
        runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
        p = runtime.get("dispatch_lock_path") or str(DEFAULT_DISPATCH_LOCK_PATH)
        return Path(str(p))

    def _load_state(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        path = self._state_path(cfg)
        try:
            if path.exists():
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {"last_regime": None, "last_score": None, "last_sent_ts": 0.0}

    def _save_state(self, cfg: Dict[str, Any]) -> None:
        path = self._state_path(cfg)
        try:
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _send_telegram(self, cfg: Dict[str, Any], message: str) -> None:
        tele = cfg.get("telegram") if isinstance(cfg.get("telegram"), dict) else {}
        token = str(tele.get("token") or "").strip()
        chat_id = int(_safe_float(tele.get("chat_id"), 0.0))
        if not token or chat_id <= 0:
            print("[MARKET_REGIME] Telegram is not configured; skipping notification.")
            return

        broadcast = bool(tele.get("broadcast_to_subscribers", True))
        targets: List[int] = []
        if broadcast:
            try:
                targets = list(get_subscribers(chat_id, token=token))
            except Exception:
                targets = [chat_id]
        else:
            targets = [chat_id]

        if not targets:
            targets = [chat_id]

        sent_any = False
        for cid in targets:
            try:
                send_telegram(message, token, int(cid))
                sent_any = True
            except Exception as e:
                # If user blocked bot, prune subscriber to keep broadcasts clean.
                try:
                    err_txt = str(e)
                except Exception:
                    err_txt = ""
                if "403" in err_txt:
                    try:
                        remove_subscriber(int(cid), token=token)
                    except Exception:
                        pass
                print(f"[MARKET_REGIME] Telegram send failed for {cid}: {e}")

        if sent_any:
            self._state["last_sent_ts"] = _now_ts()

    def _acquire_dispatch_slot(self, cfg: Dict[str, Any]) -> bool:
        """Cross-process rate-limit: at most one send per interval.

        Even if multiple market_regime processes exist, only one can claim
        the dispatch slot for the current interval.
        """
        interval_sec = max(30.0, _safe_float(cfg.get("interval_minutes"), 30.0) * 60.0)
        lock_path = self._dispatch_lock_path(cfg)
        state_path = self._state_path(cfg)

        try:
            if not lock_path.parent.exists():
                lock_path.parent.mkdir(parents=True, exist_ok=True)
            if not state_path.parent.exists():
                state_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        now_ts = _now_ts()
        try:
            with lock_path.open("a+", encoding="utf-8") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

                disk_state: Dict[str, Any] = {}
                try:
                    if state_path.exists():
                        raw = state_path.read_text(encoding="utf-8")
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            disk_state = parsed
                except Exception:
                    disk_state = {}

                last_broadcast_ts = _safe_float(disk_state.get("last_broadcast_ts"), 0.0)
                if last_broadcast_ts > 0 and (now_ts - last_broadcast_ts) < interval_sec:
                    return False

                # Reserve this interval slot immediately to block duplicates.
                disk_state["last_broadcast_ts"] = now_ts
                try:
                    state_path.write_text(json.dumps(disk_state, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            # If lock mechanism fails, do not block sending.
            return True

        return True

    def _get_all_usdt_perpetual_symbols(self, force_refresh: bool = False) -> List[str]:
        now = _now_ts()
        if not force_refresh and self._universe_cache and (now - self._universe_cache_ts) < 1800.0:
            return list(self._universe_cache)
        try:
            info = self.api.exchange_info()
            raw = info.get("symbols") if isinstance(info.get("symbols"), list) else []
            out: List[str] = []
            for s in raw:
                if not isinstance(s, dict):
                    continue
                try:
                    if str(s.get("quoteAsset") or "").upper() != "USDT":
                        continue
                    if str(s.get("status") or "").upper() != "TRADING":
                        continue
                    if str(s.get("contractType") or "").upper() != "PERPETUAL":
                        continue
                    sym = str(s.get("symbol") or "").upper().strip()
                    if sym:
                        out.append(sym)
                except Exception:
                    continue
            if out:
                self._universe_cache = sorted(set(out))
                self._universe_cache_ts = now
        except Exception:
            pass
        return list(self._universe_cache)

    def _resolve_market_symbols(self, cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
        market = cfg.get("market") if isinstance(cfg.get("market"), dict) else {}
        base_symbol = str(market.get("base_symbol") or "BTCUSDT").upper()
        ai_symbols_raw = market.get("ai_symbols") if isinstance(market.get("ai_symbols"), list) else []
        ai_symbols = [str(s).upper().strip() for s in ai_symbols_raw if str(s).strip()]

        use_all = any(s == "ALL_BINANCE_FUTURES" for s in ai_symbols)
        if use_all:
            ai_symbols = self._get_all_usdt_perpetual_symbols()

        exclude_raw = market.get("exclude_symbols") if isinstance(market.get("exclude_symbols"), list) else []
        exclude_set = {str(s).upper().strip() for s in exclude_raw if str(s).strip()}
        if exclude_set:
            ai_symbols = [s for s in ai_symbols if s not in exclude_set]

        try:
            max_symbols = int(_safe_float(market.get("universe_max_symbols"), 0.0))
        except Exception:
            max_symbols = 0
        if max_symbols > 0 and len(ai_symbols) > max_symbols:
            ai_symbols = ai_symbols[:max_symbols]

        if not ai_symbols:
            ai_symbols = [base_symbol]
        return base_symbol, ai_symbols

    def _select_symbols_for_cycle(self, cfg: Dict[str, Any], symbols: List[str], key: str, default_count: int) -> List[str]:
        if not symbols:
            return []
        market = cfg.get("market") if isinstance(cfg.get("market"), dict) else {}
        try:
            per_cycle = int(_safe_float(market.get(key), float(default_count)))
        except Exception:
            per_cycle = int(default_count)
        if per_cycle <= 0 or per_cycle >= len(symbols):
            return list(symbols)

        rot_key = f"rotation_index_{key}"
        try:
            start = int(self._state.get(rot_key) or 0) % len(symbols)
        except Exception:
            start = 0
        out: List[str] = []
        for i in range(per_cycle):
            out.append(symbols[(start + i) % len(symbols)])
        self._state[rot_key] = int((start + per_cycle) % len(symbols))
        return out

    def _volume_ratio_5m(self, symbol: str, lookback: int) -> Optional[float]:
        kl = self.api.klines(symbol=symbol, interval="5m", limit=max(lookback + 2, 30))
        if len(kl) < 10:
            return None
        # Use last fully closed candle.
        last_closed = kl[-2]
        prev = kl[:-2]
        if not prev:
            return None
        last_vol = _safe_float(last_closed[5], 0.0)
        vols = [_safe_float(x[5], 0.0) for x in prev if _safe_float(x[5], 0.0) > 0.0]
        if not vols:
            return None
        avg = sum(vols) / len(vols)
        if avg <= 0:
            return None
        return last_vol / avg

    def _oi_change_5m_pct(self, symbol: str) -> Optional[float]:
        oi = self.api.open_interest_hist_5m(symbol=symbol, limit=2)
        if len(oi) < 2:
            return None
        old = _safe_float((oi[0] or {}).get("sumOpenInterestValue"), 0.0)
        new = _safe_float((oi[-1] or {}).get("sumOpenInterestValue"), 0.0)
        if old <= 0:
            return None
        return ((new - old) / old) * 100.0

    def _microstructure_metrics(self, symbols: List[str], slippage_notional: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        spreads: List[float] = []
        depths: List[float] = []
        slippages: List[float] = []

        for sym in symbols:
            try:
                bt = self.api.book_ticker(sym)
                depth = self.api.depth(sym, limit=5)
            except Exception:
                continue

            bid = _safe_float(bt.get("bidPrice"), 0.0)
            ask = _safe_float(bt.get("askPrice"), 0.0)
            if bid <= 0 or ask <= 0 or ask < bid:
                continue
            mid = (bid + ask) / 2.0
            spread_pct = ((ask - bid) / mid) * 100.0
            spreads.append(spread_pct)

            bids = depth.get("bids") if isinstance(depth.get("bids"), list) else []
            asks = depth.get("asks") if isinstance(depth.get("asks"), list) else []
            bid_depth = 0.0
            ask_depth = 0.0
            for row in bids:
                if isinstance(row, list) and len(row) >= 2:
                    px = _safe_float(row[0], 0.0)
                    qty = _safe_float(row[1], 0.0)
                    if px > 0 and qty > 0:
                        bid_depth += px * qty
            for row in asks:
                if isinstance(row, list) and len(row) >= 2:
                    px = _safe_float(row[0], 0.0)
                    qty = _safe_float(row[1], 0.0)
                    if px > 0 and qty > 0:
                        ask_depth += px * qty
            top5_depth = bid_depth + ask_depth
            depths.append(top5_depth)

            side_depth = min(bid_depth, ask_depth)
            if side_depth > 0:
                slip_bps = (slippage_notional / side_depth) * 10000.0
                slippages.append(slip_bps)

        if not spreads or not depths or not slippages:
            return None, None, None
        return sum(spreads) / len(spreads), sum(depths) / len(depths), sum(slippages) / len(slippages)

    def _correlation_with_btc(self, base_symbol: str, symbols: List[str], window_minutes: int) -> Optional[float]:
        k_base = self.api.klines(base_symbol, interval="1m", limit=max(window_minutes + 2, 20))
        if len(k_base) < 10:
            return None
        base_closes = [_safe_float(x[4], 0.0) for x in k_base if _safe_float(x[4], 0.0) > 0.0]
        base_ret = _returns_from_closes(base_closes)
        if len(base_ret) < 5:
            return None

        corrs: List[float] = []
        for sym in symbols:
            if sym.upper() == base_symbol.upper():
                continue
            try:
                k_sym = self.api.klines(sym, interval="1m", limit=max(window_minutes + 2, 20))
            except Exception:
                continue
            closes = [_safe_float(x[4], 0.0) for x in k_sym if _safe_float(x[4], 0.0) > 0.0]
            rets = _returns_from_closes(closes)
            n = min(len(base_ret), len(rets))
            if n < 5:
                continue
            c = _pearson_corr(base_ret[-n:], rets[-n:])
            if c is None:
                continue
            corrs.append(c)

        if not corrs:
            return None
        return sum(corrs) / len(corrs)

    def _normalize_ratio(self, value: Optional[float], threshold: float, cap: float = 2.5) -> float:
        if value is None or threshold <= 0:
            return 0.0
        return _clamp((value / threshold) / cap, 0.0, 1.0)

    def _normalize_pct(self, value: Optional[float], threshold: float, cap_mult: float = 3.0) -> float:
        if value is None or threshold <= 0:
            return 0.0
        return _clamp((value / threshold) / cap_mult, 0.0, 1.0)

    def _normalize_inverse(self, value: Optional[float], threshold: float) -> float:
        if value is None or threshold <= 0:
            return 0.0
        # Lower is better: score=1 when value<=threshold; decays afterwards.
        if value <= threshold:
            return 1.0
        return _clamp(threshold / value, 0.0, 1.0)

    def _analysis_text(self, snap: RegimeSnapshot) -> str:
        if snap.regime == "ACTIVE":
            return "Մասնակցությունը բարձր է, շուկան ակտիվ է (volume/OI/liquidity համադրված):"
        if snap.regime == "QUIET":
            return "Մասնակցությունը ցածր է, շուկան բարակ է (thin/liquidity risk):"
        return "Միջանկյալ ռեժիմ է՝ ակտիվությունը խառը/չեզոք է:"

    def _build_message(self, cfg: Dict[str, Any], snap: RegimeSnapshot) -> str:
        tele = cfg.get("telegram") if isinstance(cfg.get("telegram"), dict) else {}
        include_emoji = bool(tele.get("include_emoji", True))

        icon = ""
        if include_emoji:
            icon = "🟢 " if snap.regime == "ACTIVE" else ("🟡 " if snap.regime == "NEUTRAL" else "🔴 ")

        lines = [
            f"{icon}MARKET REGIME: {snap.regime}".strip(),
            f"Score: {snap.score:.3f}",
            f"Confidence: {snap.confidence:.2f}",
            f"Analysis: {self._analysis_text(snap)}",
            "",
            f"BTC 5m Vol Ratio: x{snap.btc_volume_ratio:.2f}",
            f"AI 5m Vol Ratio: x{snap.ai_volume_ratio:.2f}",
            f"AI OI Change (5m): {_fmt_pct(snap.ai_oi_change_pct)}",
            f"Avg Spread: {_fmt_pct(snap.avg_spread_pct)}",
            f"Avg Top5 Depth: {snap.avg_depth_usdt:,.0f} USDT",
            f"Est. Slippage: {snap.slippage_bps:.2f} bps",
            f"Corr(AI,BTC): {snap.correlation:.3f}",
        ]

        return "\n".join(lines)

    def _compute_snapshot(self, cfg: Dict[str, Any]) -> RegimeSnapshot:
        thresholds = cfg.get("thresholds") if isinstance(cfg.get("thresholds"), dict) else {}
        weights = cfg.get("weights") if isinstance(cfg.get("weights"), dict) else {}
        windows = cfg.get("windows") if isinstance(cfg.get("windows"), dict) else {}
        limits = cfg.get("limits") if isinstance(cfg.get("limits"), dict) else {}
        base_symbol, all_symbols = self._resolve_market_symbols(cfg)
        ai_symbols = self._select_symbols_for_cycle(cfg, all_symbols, "analysis_symbols_per_cycle", 35)
        micro_symbols = self._select_symbols_for_cycle(cfg, ai_symbols, "microstructure_symbols_per_cycle", 25)
        corr_symbols = self._select_symbols_for_cycle(cfg, ai_symbols, "correlation_symbols_per_cycle", 20)

        lookback = int(_safe_float(windows.get("volume_5m_lookback_candles"), 288))
        corr_win = int(_safe_float(windows.get("correlation_window_minutes"), 30))
        slip_notional = _safe_float(limits.get("slippage_estimate_notional_usdt"), 1000.0)

        btc_vol_ratio = self._volume_ratio_5m(base_symbol, lookback) or 0.0

        ai_vol_ratios: List[float] = []
        ai_oi_changes: List[float] = []
        for sym in ai_symbols:
            try:
                v = self._volume_ratio_5m(sym, lookback)
                if v is not None:
                    ai_vol_ratios.append(v)
            except Exception:
                pass
            try:
                oi = self._oi_change_5m_pct(sym)
                if oi is not None:
                    ai_oi_changes.append(oi)
            except Exception:
                pass

        ai_vol_ratio = (sum(ai_vol_ratios) / len(ai_vol_ratios)) if ai_vol_ratios else 0.0
        ai_oi_change = (sum(ai_oi_changes) / len(ai_oi_changes)) if ai_oi_changes else 0.0

        avg_spread, avg_depth, slippage_bps = self._microstructure_metrics(micro_symbols, slip_notional)
        if avg_spread is None:
            avg_spread = 999.0
        if avg_depth is None:
            avg_depth = 0.0
        if slippage_bps is None:
            slippage_bps = 999.0

        corr = self._correlation_with_btc(base_symbol, corr_symbols, corr_win)
        if corr is None:
            corr = 0.0

        vol_score_btc = self._normalize_ratio(
            btc_vol_ratio, _safe_float(thresholds.get("btc_volume_ratio_active"), 1.5), cap=2.0
        )
        vol_score_ai = self._normalize_ratio(
            ai_vol_ratio, _safe_float(thresholds.get("ai_volume_ratio_active"), 1.3), cap=2.0
        )
        volume_signal = (vol_score_btc + vol_score_ai) / 2.0

        oi_signal = self._normalize_pct(
            ai_oi_change, _safe_float(thresholds.get("ai_oi_change_pct_active"), 0.7), cap_mult=3.0
        )

        spread_score = self._normalize_inverse(avg_spread, _safe_float(thresholds.get("spread_pct_max"), 0.04))
        depth_min = _safe_float(thresholds.get("top5_depth_usdt_min"), 250000.0)
        depth_score = _clamp(avg_depth / max(depth_min, 1.0), 0.0, 1.0)
        slip_score = self._normalize_inverse(slippage_bps, _safe_float(thresholds.get("slippage_bps_max"), 6.0))
        micro_signal = (spread_score + depth_score + slip_score) / 3.0

        corr_thr = _safe_float(thresholds.get("correlation_min"), 0.35)
        corr_signal = _clamp((corr + 1.0) / 2.0, 0.0, 1.0) if corr >= corr_thr else _clamp(corr / max(corr_thr, 1e-9), 0.0, 1.0) * 0.5

        w_vol = _safe_float(weights.get("volume"), 0.4)
        w_oi = _safe_float(weights.get("oi"), 0.3)
        w_micro = _safe_float(weights.get("microstructure"), 0.2)
        w_corr = _safe_float(weights.get("correlation"), 0.1)
        w_sum = w_vol + w_oi + w_micro + w_corr
        if w_sum <= 0:
            w_vol, w_oi, w_micro, w_corr, w_sum = 0.4, 0.3, 0.2, 0.1, 1.0

        score = ((w_vol * volume_signal) + (w_oi * oi_signal) + (w_micro * micro_signal) + (w_corr * corr_signal)) / w_sum
        score = _clamp(score, 0.0, 1.0)

        active_thr = _safe_float(thresholds.get("score_active"), 0.60)
        quiet_thr = _safe_float(thresholds.get("score_quiet"), 0.40)
        if score >= active_thr:
            regime = "ACTIVE"
        elif score <= quiet_thr:
            regime = "QUIET"
        else:
            regime = "NEUTRAL"

        confidence = abs(score - 0.5) * 2.0

        return RegimeSnapshot(
            regime=regime,
            score=score,
            confidence=confidence,
            btc_volume_ratio=btc_vol_ratio,
            ai_volume_ratio=ai_vol_ratio,
            ai_oi_change_pct=ai_oi_change,
            avg_spread_pct=avg_spread,
            avg_depth_usdt=avg_depth,
            slippage_bps=slippage_bps,
            correlation=corr,
            ts=_now_ts(),
        )

    def run_once(self, force_send: bool = False) -> None:
        cfg = self.cfg_mgr.load_live()
        if not bool(cfg.get("enabled", True)):
            print("[MARKET_REGIME] disabled=true in config; skipping.")
            return

        snap = self._compute_snapshot(cfg)
        msg = self._build_message(cfg, snap)

        tele = cfg.get("telegram") if isinstance(cfg.get("telegram"), dict) else {}
        only_on_change = bool(tele.get("only_on_regime_change", True))
        last_regime = self._state.get("last_regime")
        should_send = bool(force_send) or (not only_on_change) or (last_regime != snap.regime)

        print(
            f"[MARKET_REGIME] regime={snap.regime} score={snap.score:.3f} "
            f"btc_vol=x{snap.btc_volume_ratio:.2f} ai_vol=x{snap.ai_volume_ratio:.2f} "
            f"oi={snap.ai_oi_change_pct:.2f}% corr={snap.correlation:.3f}"
        )

        if should_send and self._acquire_dispatch_slot(cfg):
            self._send_telegram(cfg, msg)

        self._state["last_regime"] = snap.regime
        self._state["last_score"] = snap.score
        self._save_state(cfg)

    def run_forever(self) -> None:
        startup_sent = False
        first_cycle = True
        while True:
            cycle_started = _now_ts()
            cfg = self.cfg_mgr.load_live()
            interval_min = _safe_float(cfg.get("interval_minutes"), 30.0)
            interval_sec = max(30.0, interval_min * 60.0)

            if not startup_sent:
                tele = cfg.get("telegram") if isinstance(cfg.get("telegram"), dict) else {}
                if bool(tele.get("send_startup_message", True)):
                    self._send_telegram(
                        cfg,
                        f"🧭 Market Regime Detector started\nInterval: {interval_min:.1f}m\nIsolation: signal-only (no trade impact)",
                    )
                startup_sent = True

            try:
                tele = cfg.get("telegram") if isinstance(cfg.get("telegram"), dict) else {}
                force_first = bool(tele.get("send_first_analysis_on_startup", True))
                self.run_once(force_send=bool(first_cycle and force_first))
            except Exception as e:
                print(f"[MARKET_REGIME] cycle failed: {e}")
            first_cycle = False
            elapsed = _now_ts() - cycle_started
            sleep_sec = max(5.0, interval_sec - elapsed)
            time.sleep(sleep_sec)


def main() -> None:
    cfg_path = Path(os.environ.get("MARKET_REGIME_CONFIG", str(DEFAULT_CONFIG_PATH)))
    cfg_mgr = ConfigManager(cfg_path=cfg_path, trading_cfg_path=DEFAULT_TRADING_CONFIG_PATH)
    cfg_for_pid = cfg_mgr.load_live()
    runtime = cfg_for_pid.get("runtime") if isinstance(cfg_for_pid.get("runtime"), dict) else {}
    pid_path = Path(str(runtime.get("pid_path") or DEFAULT_PID_PATH))
    if not _acquire_singleton_pidfile(pid_path):
        return
    detector = MarketRegimeDetector(cfg_mgr)
    run_once = str(os.environ.get("MARKET_REGIME_RUN_ONCE", "")).strip().lower() in {"1", "true", "yes", "on"}
    if run_once:
        detector.run_once()
        return
    detector.run_forever()


if __name__ == "__main__":
    main()
