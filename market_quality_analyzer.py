"""
Market Quality Analyzer — Independent background process.

Analyzes every tradeable symbol every hour using:
  1. Fibonacci adherence (did price respect natural levels?)
  2. 5 ML models consensus (did models correctly predict movement?)
  3. Order book health (thin / spoofed / real?)

Results saved to: data/market_quality_cache.json
Telegram alerts sent for SUSPECT / MANIPULATED symbols.

BLOCKING_ENABLED = False by default.
When you are ready to activate trade blocking, set it to True.
The cache file already contains the "blocked_until" field — main.py
can read it at any time to gate trades.

Run:
  python market_quality_analyzer.py
  python market_quality_analyzer.py --once   # single pass, then exit
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ─────────────────────────────────────────────────────────────
#  MASTER SWITCH — set True when you want to actually block trades
# ─────────────────────────────────────────────────────────────
BLOCKING_ENABLED = True

# ─────────────────────────────────────────────────────────────
#  PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent
DATA_DIR          = BASE_DIR / "data"
CACHE_PATH        = DATA_DIR / "market_quality_cache.json"
PID_PATH          = DATA_DIR / "market_quality_analyzer.pid"
TRADING_CFG_PATH  = BASE_DIR / "config" / "trading.yaml"
ML_MODELS_DIR     = BASE_DIR / "ml" / "models"

BINANCE_FAPI      = "https://fapi.binance.com"
REQUEST_TIMEOUT   = 10
CANDLES_LIMIT     = 90          # 90 x 1-min = last 90 minutes
ANALYSIS_INTERVAL = 3600        # re-analyze every 60 minutes
BLOCK_DURATION    = 3600        # block for 60 minutes when BLOCKING_ENABLED
NUM_MODELS        = 5           # how many ML models to use
FIB_TOUCH_PCT     = 0.003       # 0.3% proximity counts as "touching" a Fib level
SUSPECT_THRESHOLD = 40          # score < 40 → SUSPECT
MANIP_THRESHOLD   = 30          # score < 30 → MANIPULATED
SYMBOL_DELAY      = 0.5         # seconds between symbols (rate-limit safety)
MAX_SYMBOLS       = 9999        # analyze all available symbols (no practical cap)


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _now() -> float:
    return time.time()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[MQA {ts}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────
#  SINGLETON PID GUARD
# ─────────────────────────────────────────────────────────────

def _acquire_pidfile() -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    try:
        if PID_PATH.exists():
            existing = int(PID_PATH.read_text().strip() or "0")
            if existing and existing != pid:
                try:
                    os.kill(existing, 0)
                    _log(f"Another instance already running (pid={existing}). Exiting.")
                    return False
                except OSError:
                    pass
        PID_PATH.write_text(str(pid))
    except Exception:
        pass

    def _cleanup():
        try:
            raw = PID_PATH.read_text().strip()
            if raw and int(raw) == pid:
                PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_cleanup)
    return True


# ─────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────

def _load_telegram_config() -> Tuple[str, int]:
    """Return (token, chat_id) from trading.yaml."""
    cfg = _load_yaml(TRADING_CFG_PATH)
    token   = str(cfg.get("telegram_token") or "").strip()
    chat_id = int(_safe_float(cfg.get("telegram_chat_id"), 0))
    return token, chat_id


def _send_telegram(message: str) -> None:
    token, chat_id = _load_telegram_config()
    if not token or chat_id <= 0:
        _log("Telegram not configured, skipping notification.")
        return
    try:
        from monitoring.telegram import send_telegram
        from monitoring.subscribers import get_subscribers, remove_subscriber
        targets = get_subscribers(chat_id, token=token) or [chat_id]
        for cid in targets:
            try:
                send_telegram(message, token, int(cid))
            except Exception as e:
                if "403" in str(e):
                    try:
                        remove_subscriber(int(cid), token=token)
                    except Exception:
                        pass
    except Exception as e:
        _log(f"Telegram send error: {e}")


# ─────────────────────────────────────────────────────────────
#  BINANCE API
# ─────────────────────────────────────────────────────────────

def _get(url: str, params: Dict = None) -> Any:
    resp = requests.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_usdt_futures_symbols() -> List[str]:
    """Return all active USDT perpetual futures symbols."""
    try:
        data = _get(f"{BINANCE_FAPI}/fapi/v1/exchangeInfo")
        symbols = []
        for s in data.get("symbols", []):
            if (
                s.get("quoteAsset") == "USDT"
                and s.get("contractType") == "PERPETUAL"
                and s.get("status") == "TRADING"
            ):
                symbols.append(s["symbol"])
        return sorted(symbols)
    except Exception as e:
        _log(f"Failed to fetch symbol universe: {e}")
        return []


def fetch_candles(symbol: str, interval: str = "1m", limit: int = CANDLES_LIMIT) -> List[Dict]:
    """Return list of OHLCV candle dicts."""
    try:
        raw = _get(f"{BINANCE_FAPI}/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        })
        candles = []
        for row in raw:
            candles.append({
                "open":   float(row[1]),
                "high":   float(row[2]),
                "low":    float(row[3]),
                "close":  float(row[4]),
                "volume": float(row[5]),
            })
        return candles
    except Exception as e:
        _log(f"  [{symbol}] candles error: {e}")
        return []


def fetch_order_book(symbol: str, limit: int = 20) -> Dict:
    """Return order book snapshot with bids and asks."""
    try:
        return _get(f"{BINANCE_FAPI}/fapi/v1/depth", {
            "symbol": symbol,
            "limit": limit,
        })
    except Exception as e:
        _log(f"  [{symbol}] order book error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
#  ML MODELS — load the 5 highest-accuracy models
# ─────────────────────────────────────────────────────────────

def _parse_model_score(path: Path) -> float:
    """Extract accuracy score from filename like model_1234567890_0.8321.pkl"""
    try:
        stem = path.stem  # model_1234567890_0.8321
        parts = stem.split("_")
        return float(parts[-1])
    except Exception:
        return 0.0


def load_top_models(n: int = NUM_MODELS) -> List[Any]:
    """Load the N highest-accuracy models from ml/models/."""
    try:
        all_models = sorted(
            ML_MODELS_DIR.glob("model_*.pkl"),
            key=_parse_model_score,
            reverse=True,
        )
        top = all_models[:n]
        loaded = []
        for path in top:
            try:
                try:
                    import joblib
                    model = joblib.load(path)
                except ImportError:
                    with open(path, "rb") as f:
                        model = pickle.load(f)
                loaded.append(model)
                _log(f"  Loaded model: {path.name} (score={_parse_model_score(path):.4f})")
            except Exception as e:
                _log(f"  Failed to load {path.name}: {e}")
        return loaded
    except Exception as e:
        _log(f"Failed to load models: {e}")
        return []


def _build_feature_vector(candles: List[Dict]) -> List[float]:
    """Build the same 7-feature vector used by InferenceEngine."""
    try:
        from features.feature_store import build_features
        feats = build_features(candles)
        structure_map = {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}
        range_map = {"RANGE": 0, "TREND": 1}
        return [[
            feats["rsi"],
            feats["momentum"],
            feats["acceleration"],
            feats["volatility"],
            feats.get("atr", 0.0),
            structure_map.get(feats.get("structure", "NEUTRAL"), 0),
            range_map.get(feats.get("range", {}).get("type", "RANGE"), 0),
        ]]
    except Exception:
        return None


def _model_predict_up(model: Any, candles: List[Dict]) -> Optional[bool]:
    """Return True if model predicts price will go UP, False for DOWN, None on error."""
    try:
        x = _build_feature_vector(candles)
        if x is None:
            return None
        probs = model.predict_proba(x)
        row = probs[0]
        if len(row) >= 2:
            p_up = float(row[1])
        else:
            p_up = float(row[0])
        return p_up >= 0.5
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
#  ANALYSIS COMPONENTS
# ─────────────────────────────────────────────────────────────

def _analyze_fibonacci(candles: List[Dict]) -> Tuple[float, str]:
    """
    Score 0-40: how well did price respect Fibonacci retracement levels
    in the last 90 candles?

    We look for "Fibonacci touches" — candles where the price came within
    FIB_TOUCH_PCT of a Fib level AND then reversed direction.
    More bounces = more natural market behavior.
    """
    if len(candles) < 30:
        return 20.0, "Not enough data"

    try:
        from features.fibonacci import fibonacci_levels
    except Exception:
        return 20.0, "Fibonacci module unavailable"

    fib_levels_list = [0.236, 0.382, 0.5, 0.618, 0.786]
    total_checks = 0
    bounces = 0
    details_parts = []

    # Analyze in 3 windows of 30 candles
    segments = [candles[0:30], candles[30:60], candles[60:90]] if len(candles) >= 90 else [candles]

    for seg_idx, seg in enumerate(segments):
        if len(seg) < 10:
            continue

        try:
            fib_data = fibonacci_levels(seg)
        except Exception:
            continue

        levels = fib_data.get("levels", {})
        fib_values = [float(v) for v in levels.values()]

        if not fib_values:
            continue

        # Check each candle (except first 3 and last 3) for Fib touch + bounce
        for i in range(3, len(seg) - 3):
            c = seg[i]
            candle_range = c["high"] - c["low"]
            if candle_range <= 0:
                continue

            for fib_val in fib_values:
                # Check if candle body or wick touched the Fib level
                proximity_low  = abs(c["low"]  - fib_val) / fib_val
                proximity_high = abs(c["high"] - fib_val) / fib_val

                if proximity_low > FIB_TOUCH_PCT and proximity_high > FIB_TOUCH_PCT:
                    continue  # didn't touch this level

                total_checks += 1

                # Check bounce: next 3 candles moved away from the touch
                touch_from_below = c["low"] <= fib_val <= c["high"] and c["close"] > c["open"]
                touch_from_above = c["low"] <= fib_val <= c["high"] and c["close"] < c["open"]

                next_closes = [seg[j]["close"] for j in range(i+1, min(i+4, len(seg)))]
                if not next_closes:
                    continue

                avg_next = sum(next_closes) / len(next_closes)

                if touch_from_below and avg_next > fib_val:
                    bounces += 1  # Touched from below, bounced up — natural
                elif touch_from_above and avg_next < fib_val:
                    bounces += 1  # Touched from above, bounced down — natural
                break  # only count one level per candle

    if total_checks == 0:
        # No Fibonacci touches detected — ambiguous, give neutral score
        return 20.0, "No Fib touches detected (ambiguous)"

    ratio = bounces / total_checks
    score = round(ratio * 40.0, 1)
    pct = round(ratio * 100, 1)
    detail = f"Fib bounces: {bounces}/{total_checks} ({pct}%)"
    details_parts.append(detail)

    return score, " | ".join(details_parts) or detail


def _analyze_models(candles: List[Dict], models: List[Any]) -> Tuple[float, str]:
    """
    Score 0-40: did the ML models correctly predict what happened
    in the last 90 minutes?

    Method:
      - Split 90 candles into 3 windows of 30
      - For each window: use first 20 candles to predict, check next 10
      - Run all 5 models, count how many predicted correctly
    Score = (correct / total_predictions) * 40
    """
    if not models or len(candles) < 30:
        return 20.0, "No models or insufficient data"

    total_predictions = 0
    correct_predictions = 0

    segments = []
    if len(candles) >= 90:
        segments = [(candles[0:30], 20), (candles[30:60], 20), (candles[60:90], 20)]
    elif len(candles) >= 30:
        segments = [(candles, 20)]

    for seg, split_at in segments:
        if len(seg) <= split_at:
            continue

        train_part = seg[:split_at]
        future_part = seg[split_at:]

        if len(train_part) < 20 or not future_part:
            continue

        # Actual direction: did price go up in the "future" part?
        entry_close = train_part[-1]["close"]
        exit_close  = future_part[-1]["close"]
        actually_went_up = exit_close > entry_close

        for model in models:
            pred_up = _model_predict_up(model, train_part)
            if pred_up is None:
                continue
            total_predictions += 1
            if pred_up == actually_went_up:
                correct_predictions += 1

    if total_predictions == 0:
        return 20.0, "Models could not produce predictions"

    ratio = correct_predictions / total_predictions
    score = round(ratio * 40.0, 1)
    pct = round(ratio * 100, 1)
    detail = f"Model accuracy on last 90min: {correct_predictions}/{total_predictions} ({pct}%)"
    return score, detail


def _analyze_order_book(order_book: Dict) -> Tuple[float, str]:
    """
    Score 0-20: is the order book healthy (real participants) or thin/spoofed?

    Checks:
      - Total depth in USDT (thin = low liquidity)
      - Largest single order as % of total depth (high = possible wall/spoof)
      - Bid/ask imbalance (severe imbalance = manipulation signal)
    """
    if not order_book:
        return 5.0, "Order book unavailable"

    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    if not bids or not asks:
        return 5.0, "Empty order book"

    def side_stats(orders):
        total_usdt = 0.0
        max_single = 0.0
        for price_str, qty_str in orders:
            try:
                price = float(price_str)
                qty   = float(qty_str)
                usdt  = price * qty
                total_usdt += usdt
                if usdt > max_single:
                    max_single = usdt
            except Exception:
                pass
        return total_usdt, max_single

    bid_total, bid_max = side_stats(bids)
    ask_total, ask_max = side_stats(asks)
    total_depth = bid_total + ask_total

    if total_depth <= 0:
        return 5.0, "Zero depth"

    score = 20.0
    issues = []

    # 1) Thin market penalty
    if total_depth < 50_000:
        score -= 10.0
        issues.append(f"Thin depth ${total_depth:,.0f}")
    elif total_depth < 200_000:
        score -= 5.0
        issues.append(f"Low depth ${total_depth:,.0f}")

    # 2) Wall/spoofing detection — largest single order dominates one side
    bid_wall_ratio = bid_max / bid_total if bid_total > 0 else 0
    ask_wall_ratio = ask_max / ask_total if ask_total > 0 else 0
    wall_ratio = max(bid_wall_ratio, ask_wall_ratio)

    if wall_ratio > 0.6:
        score -= 8.0
        issues.append(f"Spoof wall detected ({wall_ratio*100:.0f}% in 1 order)")
    elif wall_ratio > 0.4:
        score -= 4.0
        issues.append(f"Large wall ({wall_ratio*100:.0f}% in 1 order)")

    # 3) Severe bid/ask imbalance
    if bid_total > 0 and ask_total > 0:
        imbalance = abs(bid_total - ask_total) / total_depth
        if imbalance > 0.7:
            score -= 4.0
            issues.append(f"Severe imbalance ({imbalance*100:.0f}%)")
        elif imbalance > 0.5:
            score -= 2.0
            issues.append(f"Imbalance ({imbalance*100:.0f}%)")

    score = max(0.0, round(score, 1))
    summary = f"Depth ${total_depth:,.0f}"
    if issues:
        summary += " | " + " | ".join(issues)
    return score, summary


# ─────────────────────────────────────────────────────────────
#  CACHE
# ─────────────────────────────────────────────────────────────

def load_cache() -> Dict[str, Any]:
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def save_cache(cache: Dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        _log(f"Cache save error: {e}")


# ─────────────────────────────────────────────────────────────
#  BLOCKING GATE (disabled by default)
# ─────────────────────────────────────────────────────────────

def is_symbol_blocked(symbol: str) -> bool:
    """
    Returns True if this symbol should NOT be traded right now.
    Always returns False while BLOCKING_ENABLED = False.

    When you activate BLOCKING_ENABLED = True, this function reads
    the cache and blocks symbols flagged as SUSPECT/MANIPULATED
    for the next BLOCK_DURATION seconds.
    """
    if not BLOCKING_ENABLED:
        return False

    try:
        cache = load_cache()
        entry = cache.get(symbol)
        if not entry:
            return False
        blocked_until = _safe_float(entry.get("blocked_until") or 0)
        if blocked_until > _now():
            return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────────────────────
#  CORE ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_symbol(symbol: str, models: List[Any], cache: Dict[str, Any]) -> Optional[Dict]:
    """
    Full analysis of one symbol.
    Returns result dict or None on failure.
    """
    candles = fetch_candles(symbol)
    if len(candles) < 30:
        return None

    order_book = fetch_order_book(symbol)

    fib_score,   fib_detail   = _analyze_fibonacci(candles)
    model_score, model_detail = _analyze_models(candles, models)
    ob_score,    ob_detail    = _analyze_order_book(order_book)

    total_score = round(fib_score + model_score + ob_score, 1)

    if total_score >= SUSPECT_THRESHOLD:
        verdict = "GOOD"
    elif total_score >= MANIP_THRESHOLD:
        verdict = "SUSPECT"
    else:
        verdict = "MANIPULATED"

    blocked_until = None
    if BLOCKING_ENABLED and verdict in ("SUSPECT", "MANIPULATED"):
        blocked_until = _now() + BLOCK_DURATION

    result = {
        "symbol":        symbol,
        "score":         total_score,
        "verdict":       verdict,
        "fib_score":     fib_score,
        "model_score":   model_score,
        "ob_score":      ob_score,
        "fib_detail":    fib_detail,
        "model_detail":  model_detail,
        "ob_detail":     ob_detail,
        "blocked_until": blocked_until,
        "last_analyzed": _now(),
    }

    return result


# ─────────────────────────────────────────────────────────────
#  TELEGRAM REPORT
# ─────────────────────────────────────────────────────────────

def _verdict_icon(verdict: str) -> str:
    return {"GOOD": "✅", "SUSPECT": "⚠️", "MANIPULATED": "🚫"}.get(verdict, "❓")


def build_cycle_report(results: List[Dict], symbols_analyzed: int, duration_sec: float) -> str:
    """Build a Telegram summary message for the completed analysis cycle."""
    good       = [r for r in results if r["verdict"] == "GOOD"]
    suspect    = [r for r in results if r["verdict"] == "SUSPECT"]
    manipulated = [r for r in results if r["verdict"] == "MANIPULATED"]

    lines = [
        "═══ MARKET QUALITY REPORT ═══",
        f"Analyzed: {symbols_analyzed} symbols in {duration_sec/60:.1f}min",
        f"✅ GOOD: {len(good)}  ⚠️ SUSPECT: {len(suspect)}  🚫 MANIPULATED: {len(manipulated)}",
        f"Blocking active: {'YES' if BLOCKING_ENABLED else 'NO (observation mode)'}",
        "",
    ]

    # Show worst symbols
    bad = sorted(suspect + manipulated, key=lambda r: r["score"])
    if bad:
        lines.append("--- Worst symbols ---")
        for r in bad[:15]:
            icon = _verdict_icon(r["verdict"])
            lines.append(
                f"{icon} {r['symbol']:12s}  score={r['score']:.0f}/100"
            )
            lines.append(f"    Fib={r['fib_score']:.0f}  Models={r['model_score']:.0f}  Book={r['ob_score']:.0f}")
            lines.append(f"    {r['model_detail']}")
            lines.append(f"    {r['ob_detail']}")
        lines.append("")

    # Show top 5 best symbols
    best = sorted(good, key=lambda r: r["score"], reverse=True)[:5]
    if best:
        lines.append("--- Best quality symbols ---")
        for r in best:
            lines.append(f"✅ {r['symbol']:12s}  score={r['score']:.0f}/100")
        lines.append("")

    lines.append(f"Next analysis in ~60min")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────

def run_cycle(models: List[Any]) -> List[Dict]:
    """Analyze all symbols once. Returns list of result dicts."""
    _log("Starting analysis cycle...")
    cycle_start = _now()

    symbols = fetch_usdt_futures_symbols()
    if not symbols:
        _log("No symbols fetched, skipping cycle.")
        return []

    # Cap to avoid very long cycles
    if len(symbols) > MAX_SYMBOLS:
        symbols = symbols[:MAX_SYMBOLS]

    _log(f"Analyzing {len(symbols)} symbols...")

    cache = load_cache()
    results = []

    for i, symbol in enumerate(symbols):
        try:
            result = analyze_symbol(symbol, models, cache)
            if result:
                cache[symbol] = result
                results.append(result)
                if result["verdict"] != "GOOD":
                    _log(
                        f"  [{i+1}/{len(symbols)}] {symbol}: {result['verdict']} "
                        f"score={result['score']:.0f} "
                        f"(fib={result['fib_score']:.0f} "
                        f"model={result['model_score']:.0f} "
                        f"book={result['ob_score']:.0f})"
                    )
        except Exception as e:
            _log(f"  [{symbol}] Analysis error: {e}")

        # Save cache incrementally so results survive interruption
        if (i + 1) % 20 == 0:
            save_cache(cache)

        time.sleep(SYMBOL_DELAY)

    save_cache(cache)

    duration = _now() - cycle_start
    _log(f"Cycle complete: {len(results)} analyzed in {duration/60:.1f}min")

    # Build and send Telegram report
    if results:
        report = build_cycle_report(results, len(results), duration)
        _log("Sending Telegram report...")
        _send_telegram(report)

    return results


def main(run_once: bool = False) -> None:
    if not _acquire_pidfile():
        return

    _log("Market Quality Analyzer starting...")
    _log(f"BLOCKING_ENABLED = {BLOCKING_ENABLED}")
    _log(f"Cache path: {CACHE_PATH}")

    _send_telegram(
        f"Market Quality Analyzer started\n"
        f"Blocking: {'ACTIVE' if BLOCKING_ENABLED else 'OFF (observation mode)'}\n"
        f"Analyzing ALL USDT futures symbols every 60min"
    )

    # Load ML models once at startup
    _log(f"Loading top {NUM_MODELS} ML models...")
    models = load_top_models(NUM_MODELS)
    if not models:
        _log("WARNING: No ML models loaded. Model score will be neutral (20/40).")

    while True:
        try:
            run_cycle(models)
        except Exception as e:
            _log(f"Cycle error: {e}")

        if run_once:
            _log("--once flag set, exiting.")
            break

        _log(f"Sleeping {ANALYSIS_INTERVAL//60} minutes until next cycle...")
        time.sleep(ANALYSIS_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Quality Analyzer")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single analysis cycle then exit",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    main(run_once=args.once)
