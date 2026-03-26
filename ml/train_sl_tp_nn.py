import sys
import csv
import json
import pickle
import math
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from execution.sl_tp_nn import _LOG_PATH, _load_weights, _save_weights  # type: ignore
from execution.sl_tp_ai_engine import train_ensemble_from_log, analyze_due_snapshots  # type: ignore

try:  # optional, only needed for the supervised model
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor  # fallback, always available
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    RandomForestRegressor = None  # type: ignore


_MODEL_PATH = ROOT_DIR / "data" / "sl_tp_nn_model.pkl"
_SYMBOL_STATS_PATH = ROOT_DIR / "data" / "sl_tp_nn_symbol_stats.json"


def _build_training_pairs() -> List[Tuple[Dict[str, str], float, str]]:
    """Կառուցել training samples sl_tp_nn_log.csv-ից.

    CSV schema (header):
      timestamp_ms,symbol,side,entry_price,base_sl_pct,base_tp_pcts,
      ai_sl_pct,ai_tp_pcts,pnl_pct,reason

    suggest_sl_tp() գրառումները ունեն լցված base_*/ai_* դաշտեր և դատարկ pnl_pct,
    իսկ on_trade_exit() գրառումները՝ միայն pnl_pct/reason:
      - suggestion row:    pnl_pct == ""
      - exit row:          pnl_pct != ""

    Այստեղ մոտավորում ենք հետևյալ կերպ.
      - Տողերը սորտավորում ենք ըստ timestamp_ms
      - Յուրաքանչյուր symbol/side զույգի համար պահում ենք վերջին suggestion տողը
      - Յուրաքանչյուր exit տողի համար որոնում ենք նույն symbol/side-ի վերջին
        suggestion-ը, որը ժամանակով ավելի վաղ է (կամ հավասար), և ստեղծում
        train pair (suggest_row, pnl, reason)
    """

    if not _LOG_PATH.exists():
        raise FileNotFoundError(f"sl_tp_nn_log.csv not found at {_LOG_PATH}")

    # Defensive read:
    # sl_tp_nn_log.csv may occasionally contain NUL bytes due to abrupt writes.
    # We sanitize those bytes so one bad line does not disable the whole trainer.
    raw = _LOG_PATH.read_bytes()
    had_nul = b"\x00" in raw
    if had_nul:
        raw = raw.replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise RuntimeError("sl_tp_nn_log.csv is empty (no rows).")
    reader = csv.DictReader(lines)
    rows = [row for row in reader]

    if not rows:
        raise RuntimeError("sl_tp_nn_log.csv is empty (no rows).")
    if had_nul:
        try:
            print("[SLTP-NN TRAIN] warning: sanitized NUL bytes in sl_tp_nn_log.csv")
        except Exception:
            pass

    # sort by timestamp_ms ascending so that "last suggestion before exit" իմաստ ունենա
    def _ts(row: Dict[str, str]) -> int:
        try:
            return int(row.get("timestamp_ms") or 0)
        except (TypeError, ValueError):
            return 0

    rows.sort(key=_ts)

    last_suggestion: Dict[Tuple[str, str], Dict[str, str]] = {}
    pairs: List[Tuple[Dict[str, str], float, str]] = []

    for row in rows:
        symbol = (row.get("symbol") or "").strip()
        raw_side = (row.get("side") or "").strip().upper()
        if not symbol or not raw_side:
            continue

        # Նորմալացնում ենք side-ը այնպես, որ LONG/BUY համարվեն նույն ուղղությունը,
        # իսկ SHORT/SELL մյուսը. Սա ապահովում է, որ suggest_sl_tp()-ի LONG/SHORT
        # տողերը ճիշտ զույգվեն on_trade_exit()-ի BUY/SELL տողերի հետ:
        if raw_side in ("BUY", "LONG"):
            side_norm = "LONG"
        elif raw_side in ("SELL", "SHORT"):
            side_norm = "SHORT"
        else:
            side_norm = raw_side

        key = (symbol, side_norm)
        pnl_raw = row.get("pnl_pct")
        ai_sl_raw = row.get("ai_sl_pct")

        # Suggestion row: ai_sl_pct not empty (մեր suggest_sl_tp logging)
        if ai_sl_raw not in (None, ""):
            last_suggestion[key] = row

        # Exit row: pnl_pct not empty (on_trade_exit logging)
        if pnl_raw in (None, ""):
            continue

        try:
            pnl_val = float(pnl_raw)
        except (TypeError, ValueError):
            continue

        reason = (row.get("reason") or "").strip()
        sugg = last_suggestion.get(key)
        if not sugg:
            # Չկա համապատասխան suggestion՝ skip
            continue

        pairs.append((sugg, pnl_val, reason))

    if not pairs:
        raise RuntimeError(
            "No training pairs could be built from sl_tp_nn_log.csv. "
            "Make sure there are both suggestion and exit rows for the same symbol/side."
        )

    return pairs


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_tp_list(raw: str) -> List[float]:
    if not raw:
        return []
    parts = str(raw).split(";")
    out: List[float] = []
    for p in parts:
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if v > 0:
            out.append(v)
    return out


def _extract_features_from_suggestion(sugg: Dict[str, str]) -> List[float]:
    base_sl = _safe_float(sugg.get("base_sl_pct"), 0.0)
    ai_sl = _safe_float(sugg.get("ai_sl_pct"), base_sl)

    base_tp_list = _parse_tp_list(sugg.get("base_tp_pcts") or "")
    ai_tp_list = _parse_tp_list(sugg.get("ai_tp_pcts") or "")

    if base_tp_list:
        base_tp_mean = sum(base_tp_list) / float(len(base_tp_list))
        base_tp_max = max(base_tp_list)
    else:
        base_tp_mean = 0.0
        base_tp_max = 0.0

    if ai_tp_list:
        ai_tp_mean = sum(ai_tp_list) / float(len(ai_tp_list))
        ai_tp_max = max(ai_tp_list)
    else:
        ai_tp_mean = base_tp_mean
        ai_tp_max = base_tp_max

    raw_side = (sugg.get("side") or "").strip().upper()
    if raw_side in ("BUY", "LONG"):
        side_sign = 1.0
    elif raw_side in ("SELL", "SHORT"):
        side_sign = -1.0
    else:
        side_sign = 0.0

    atr_val = 0.0
    conf_val = 0.0
    raw_reason = sugg.get("reason") or ""
    if raw_reason:
        try:
            obj = json.loads(raw_reason)
            if isinstance(obj, dict):
                atr_val = _safe_float(obj.get("atr"), 0.0)
                conf_val = _safe_float(obj.get("confidence"), 0.0)
        except Exception:
            atr_val = 0.0
            conf_val = 0.0

    return [
        atr_val,
        conf_val,
        side_sign,
        base_sl,
        ai_sl,
        base_tp_mean,
        ai_tp_mean,
        base_tp_max,
        ai_tp_max,
    ]


def _build_dataset_from_pairs(
    pairs: List[Tuple[Dict[str, str], float, str]]
) -> Tuple[List[List[float]], List[float], List[int]]:
    X: List[List[float]] = []
    y_reg: List[float] = []
    y_cls: List[int] = []

    for sugg, pnl, _reason in pairs:
        feats = _extract_features_from_suggestion(sugg)
        X.append(feats)
        y_reg.append(float(pnl))
        y_cls.append(1 if pnl > 0 else 0)

    return X, y_reg, y_cls


def _compute_stats(pairs: List[Tuple[Dict[str, str], float, str]]) -> Dict[str, float]:
    """Հաշվում ենք ընդհանուր win/loss վիճակագրությունը ծրարային update-ի համար."""

    n = len(pairs)
    wins: List[float] = []
    losses: List[float] = []

    for _sugg, pnl, _reason in pairs:
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(pnl)

    mean_pnl = sum(p for _s, p, _r in pairs) / float(n) if n > 0 else 0.0
    avg_win = sum(wins) / float(len(wins)) if wins else 0.0
    avg_loss = sum(losses) / float(len(losses)) if losses else 0.0

    win_rate = float(len(wins)) / float(n) if n > 0 else 0.0

    return {
        "n": float(n),
        "mean_pnl": float(mean_pnl),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "win_rate": float(win_rate),
    }


def _compute_symbol_level_stats(
    pairs: List[Tuple[Dict[str, str], float, str]]
) -> Dict[str, Dict[str, float]]:
    """Aggregate performance stats per (symbol, side) over all training pairs.

    Returns a dict keyed by "SYMBOL:SIDE" (e.g. "BTCUSDT:LONG") with fields:
      - n: number of trades
      - win_rate: fraction of trades with pnl > 0
      - mean_pnl: average pnl_pct across trades
    """

    agg: Dict[Tuple[str, str], Dict[str, float]] = {}

    for sugg, pnl, _reason in pairs:
        symbol = (sugg.get("symbol") or "").strip()
        side = (sugg.get("side") or "").strip().upper()
        if not symbol or not side:
            continue

        key = (symbol, side)
        entry = agg.get(key)
        if not isinstance(entry, dict):
            entry = {"n": 0.0, "sum_pnl": 0.0, "wins": 0.0}

        try:
            pnl_val = float(pnl)
        except (TypeError, ValueError):
            pnl_val = 0.0

        entry["n"] = float(entry.get("n", 0.0)) + 1.0
        entry["sum_pnl"] = float(entry.get("sum_pnl", 0.0)) + pnl_val
        if pnl_val > 0.0:
            entry["wins"] = float(entry.get("wins", 0.0)) + 1.0

        agg[key] = entry

    out: Dict[str, Dict[str, float]] = {}
    for (symbol, side), entry in agg.items():
        n = float(entry.get("n", 0.0))
        if n <= 0.0:
            continue
        sum_pnl = float(entry.get("sum_pnl", 0.0))
        wins = float(entry.get("wins", 0.0))
        mean_pnl = sum_pnl / n
        win_rate = wins / n if n > 0 else 0.0

        key = f"{symbol}:{side}"
        out[key] = {
            "n": float(n),
            "win_rate": float(win_rate),
            "mean_pnl": float(mean_pnl),
        }

    return out


def _save_symbol_stats(stats: Dict[str, Dict[str, float]]) -> None:
    """Persist per-symbol/per-side stats to JSON for online inference use."""

    try:
        _SYMBOL_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _SYMBOL_STATS_PATH.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)
    except Exception:
        return


def _compute_risk_metrics(pairs: List[Tuple[Dict[str, str], float, str]]) -> Dict[str, float]:
    """Compute equity-curve-based risk stats from realised pnl_pct.

    We treat each pnl_pct as a per-trade return and build a simple
    cumulative equity curve in percentage points. From this we derive:

      - max_drawdown: maximum peak-to-trough drop of the equity curve
      - pnl_std: standard deviation of per-trade returns
      - sharpe: mean / std * sqrt(n) (per-trade Sharpe-like ratio)
      - sortino: mean / downside_std * sqrt(n) (penalising only losses)
    """

    if not pairs:
        return {
            "max_drawdown": 0.0,
            "pnl_std": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    pnls: List[float] = []
    for _s, pnl, _r in pairs:
        try:
            pnls.append(float(pnl))
        except (TypeError, ValueError):
            continue

    if not pnls:
        return {
            "max_drawdown": 0.0,
            "pnl_std": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    # Equity curve in percentage points.
    eq: List[float] = []
    cur = 0.0
    for p in pnls:
        cur += p
        eq.append(cur)

    max_eq = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > max_eq:
            max_eq = v
        dd = max_eq - v
        if dd > max_dd:
            max_dd = dd

    # max_drawdown is reported as a negative percentage drop.
    max_drawdown = -float(max_dd) if max_dd > 0 else 0.0

    if np is None:
        return {
            "max_drawdown": max_drawdown,
            "pnl_std": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    try:
        arr = np.array(pnls, dtype=float)
    except Exception:
        return {
            "max_drawdown": max_drawdown,
            "pnl_std": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    n = arr.size
    if n <= 1:
        return {
            "max_drawdown": max_drawdown,
            "pnl_std": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    mean_pnl = float(arr.mean())
    std_pnl = float(arr.std(ddof=1)) if n > 1 else 0.0

    sharpe = 0.0
    if std_pnl > 0:
        sharpe = (mean_pnl / std_pnl) * math.sqrt(float(n))

    # Downside deviation for Sortino (only negative returns).
    neg = arr[arr < 0.0]
    downside_std = float(neg.std(ddof=1)) if neg.size > 1 else 0.0
    sortino = 0.0
    if downside_std > 0:
        sortino = (mean_pnl / downside_std) * math.sqrt(float(n))

    return {
        "max_drawdown": float(max_drawdown),
        "pnl_std": float(std_pnl),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
    }


def _train_supervised_model(X: List[List[float]], y_reg: List[float], y_cls: List[int]) -> None:
    """Train a small supervised model on the SL/TP dataset and persist it.

    This step is intentionally offline-only and does not affect live trading
    decisions directly. It produces a RandomForestRegressor that approximates
    pnl_pct from the engineered features so that future iterations can use it
    as a value function or policy component.
    """

    if np is None or RandomForestRegressor is None:
        print("[SLTP-NN TRAIN] sklearn/numpy not available, skipping supervised model training")
        return

    try:
        import numpy as _np  # local alias to satisfy type checkers
    except Exception:
        print("[SLTP-NN TRAIN] numpy import failed at runtime, skipping supervised model training")
        return

    m = len(X)
    if m < 20:
        # Too few samples for a meaningful supervised model.
        print(f"[SLTP-NN TRAIN] not enough samples for supervised model: {m}")
        return

    try:
        X_arr = _np.array(X, dtype=float)
        y_arr = _np.array(y_reg, dtype=float)
    except Exception as e:
        print(f"[SLTP-NN TRAIN] failed to build numpy arrays: {e}")
        return

    if X_arr.ndim != 2 or y_arr.ndim != 1 or X_arr.shape[0] != y_arr.shape[0]:
        print(
            f"[SLTP-NN TRAIN] invalid dataset shapes for supervised model: X={X_arr.shape}, y={y_arr.shape}"
        )
        return

    n_total = X_arr.shape[0]
    # Simple time-based split: first 80% train, last 20% validation.
    split = int(n_total * 0.8)
    if split <= 0 or split >= n_total:
        print(f"[SLTP-NN TRAIN] degenerate split for supervised model: n={n_total}, split={split}")
        return

    X_train, X_val = X_arr[:split], X_arr[split:]
    y_train, y_val = y_arr[:split], y_arr[split:]

    try:
        try:
            from cuml.ensemble import RandomForestRegressor as _RFR_GPU
            model = _RFR_GPU(n_estimators=500, max_depth=8, random_state=42)
            print("[SLTP-NN TRAIN] GPU mode: cuML RandomForestRegressor (RTX 5070 Ti)")
        except Exception:
            model = RandomForestRegressor(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42)
            print("[SLTP-NN TRAIN] CPU mode: sklearn RandomForestRegressor")
    except Exception as e:
        print(f"[SLTP-NN TRAIN] failed to construct RandomForestRegressor: {e}")
        return

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"[SLTP-NN TRAIN] supervised model fit failed: {e}")
        return

    try:
        y_pred = model.predict(X_val)
        mae = float(_np.mean(_np.abs(y_pred - y_val))) if y_val.size > 0 else 0.0
        mse = float(_np.mean((y_pred - y_val) ** 2)) if y_val.size > 0 else 0.0

        # Classification-style accuracy on the sign of pnl.
        sign_true = _np.where(y_val > 0.0, 1, 0)
        sign_pred = _np.where(y_pred > 0.0, 1, 0)
        acc = float((sign_true == sign_pred).mean()) if y_val.size > 0 else 0.0

        print(
            "[SLTP-NN TRAIN] supervised model val: MAE={mae:.4f} MSE={mse:.4f} sign_acc={acc:.2%}"
        )
    except Exception as e:
        print(f"[SLTP-NN TRAIN] validation metrics failed: {e}")

    # Persist the model for future online use.
    try:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _MODEL_PATH.open("wb") as f:
            pickle.dump(model, f)
        print(f"[SLTP-NN TRAIN] saved supervised model to {_MODEL_PATH}")
    except Exception as e:
        print(f"[SLTP-NN TRAIN] failed to save supervised model: {e}")


def _update_weights(stats: Dict[str, float]) -> Dict[str, float]:
    """Թարմացնում է sl_tp_nn քաշերը պարզ heuristic-ի միջոցով.

    - sl_bias:   meta-կարգավորում է stop-loss տոկոսը base_sl_pct-ի նկատմամբ
      * Եթե ունենք կորուստներ, օգտագործում ենք average loss → aim tighter SL
        (ավելի փոքր տոկոս), այսինքն sl_bias->ավելի negative

    - tp_scale:  մասշտաբ է take-profit տոկոսների համար
      * Եթե միջին շահույթը մեծ է, թույլ ենք տալիս մի փոքր ավելի լայն TP
      * Եթե շահույթ գրեթե չկա և ընդհանուր mean_pnl-ը negative է, փոքր-ինչ
        կրճատում ենք tp_scale-ը, որպեսզի TP-ները ավելի մոտ լինեն

    Սա offline, զգուշավոր update է, որը չի դիպչում live exit manager-ին.
    """

    weights = _load_weights() or {}
    old_sl_bias = float(weights.get("sl_bias", 0.0))
    old_tp_scale = float(weights.get("tp_scale", 1.0))

    avg_win = float(stats.get("avg_win", 0.0))
    avg_loss = float(stats.get("avg_loss", 0.0))
    mean_pnl = float(stats.get("mean_pnl", 0.0))

    # Թիրախային sl_bias՝ հիմնված average loss-ի վրա.
    # Օրինակ՝ avg_loss ≈ -3.0% → sl_bias_target ≈ -0.3 (կրճատել ռիսկը).
    if avg_loss < 0.0:
        sl_bias_target = max(min(avg_loss / 10.0, 3.0), -3.0)
    else:
        sl_bias_target = old_sl_bias

    # Թիրախային tp_scale՝ հիմնված average win-ի և ընդհանուր mean_pnl-ի վրա.
    if avg_win > 0.0:
        # Օրինակ՝ avg_win ≈ 6.0% → tp_scale_target ≈ 1.3
        tp_scale_target = 1.0 + max(min(avg_win / 20.0, 1.0), -0.5)
    else:
        if mean_pnl < 0.0:
            # Եթե հաղթող գործարք գրեթե չկա և միջին արդյունքը բացասական է,
            # փոքր-ինչ կրճատում ենք TP-ները, որ ավելի շուտ փակվեն.
            tp_scale_target = 0.9
        else:
            tp_scale_target = old_tp_scale

    # Exponential moving average update՝ չափավոր քայլով.
    lr = 0.3
    new_sl_bias = (1.0 - lr) * old_sl_bias + lr * sl_bias_target
    new_tp_scale = (1.0 - lr) * old_tp_scale + lr * tp_scale_target

    # Պարզ անվտանգության սահմաններ.
    if new_tp_scale < 0.3:
        new_tp_scale = 0.3
    if new_tp_scale > 3.0:
        new_tp_scale = 3.0

    weights["sl_bias"] = float(f"{new_sl_bias:.6f}")
    weights["tp_scale"] = float(f"{new_tp_scale:.6f}")

    _save_weights(weights)
    return {
        "sl_bias": new_sl_bias,
        "tp_scale": new_tp_scale,
    }


def run_training() -> None:
    try:
        analyzed = int(analyze_due_snapshots(lookback_minutes=180))
        if analyzed > 0:
            print(f"[SLTP-NN TRAIN] post-trade 2h analysis processed: {analyzed}")
    except Exception as e:
        print(f"[SLTP-NN TRAIN] post-trade analysis failed: {e}")

    pairs = _build_training_pairs()
    stats = _compute_stats(pairs)

    n = int(stats["n"]) if "n" in stats else len(pairs)
    if n < 5:
        raise RuntimeError(f"Not enough training samples for sl_tp_nn: {n}")

    print("[SLTP-NN TRAIN] samples=", n)
    print(
        "[SLTP-NN TRAIN] mean_pnl={mean:.4f}% win_rate={wr:.2%} avg_win={aw:.4f}% avg_loss={al:.4f}%".format(
            mean=stats.get("mean_pnl", 0.0),
            wr=stats.get("win_rate", 0.0),
            aw=stats.get("avg_win", 0.0),
            al=stats.get("avg_loss", 0.0),
        )
    )

    # Additional risk metrics based on the realised equity curve.
    try:
        risk = _compute_risk_metrics(pairs)
        print(
            "[SLTP-NN TRAIN] risk: max_drawdown={dd:.4f} pnl_std={std:.4f} sharpe={sh:.4f} sortino={so:.4f}".format(
                dd=risk.get("max_drawdown", 0.0),
                std=risk.get("pnl_std", 0.0),
                sh=risk.get("sharpe", 0.0),
                so=risk.get("sortino", 0.0),
            )
        )
    except Exception as e:
        print(f"[SLTP-NN TRAIN] risk metrics failed: {e}")

    try:
        X, y_reg, y_cls = _build_dataset_from_pairs(pairs)
        m = len(X)
        pos = sum(1 for v in y_cls if v == 1)
        neg = m - pos
        mean_pnl_reg = sum(y_reg) / float(m) if m > 0 else 0.0
        print(
            "[SLTP-NN TRAIN] supervised dataset: samples={m} pos={p} neg={n} mean_pnl_reg={r:.4f}%".format(
                m=m,
                p=pos,
                n=neg,
                r=mean_pnl_reg,
            )
        )

        # Train and persist a supervised model (RandomForestRegressor) on the
        # same dataset. This remains offline-only and does not affect live
        # trading until explicitly wired into suggest_sl_tp.
        _train_supervised_model(X, y_reg, y_cls)
    except Exception as e:
        print(f"[SLTP-NN TRAIN] dataset build failed: {e}")

    # Compute and persist per-symbol/per-side performance statistics so that
    # the online SL/TP NN wrapper can adapt slightly by symbol and direction
    # without needing to re-scan the log file.
    try:
        sym_stats = _compute_symbol_level_stats(pairs)
        _save_symbol_stats(sym_stats)
        print(
            f"[SLTP-NN TRAIN] saved symbol stats for {len(sym_stats)} symbol/side keys to {_SYMBOL_STATS_PATH}"
        )
    except Exception as e:
        print(f"[SLTP-NN TRAIN] symbol-level stats save failed: {e}")

    new_weights = _update_weights(stats)
    print(
        "[SLTP-NN TRAIN] updated weights: sl_bias={sl:.6f}, tp_scale={tp:.6f}".format(
            sl=new_weights["sl_bias"],
            tp=new_weights["tp_scale"],
        )
    )

    # Train production SL/TP ensemble (5-model system) asynchronously in main
    # loop schedule; this function is invoked periodically.
    try:
        ens = train_ensemble_from_log(_LOG_PATH, min_samples=80)
        if float(ens.get("trained", 0.0)) > 0.0:
            print(f"[SLTP-NN TRAIN] ensemble trained; samples={int(ens.get('samples', 0.0))}")
        else:
            print(f"[SLTP-NN TRAIN] ensemble skipped; samples={int(ens.get('samples', 0.0))}")
    except Exception as e:
        print(f"[SLTP-NN TRAIN] ensemble training failed: {e}")


if __name__ == "__main__":
    run_training()
