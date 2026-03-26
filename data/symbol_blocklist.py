import json
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple


_ROOT_DIR = Path(__file__).resolve().parents[1]
_STATE_PATH = _ROOT_DIR / "data" / "symbol_blocklist.json"
_MQA_CACHE_PATH = _ROOT_DIR / "data" / "market_quality_cache.json"
_CACHE_MTIME: float | None = None
_CACHE_SYMBOLS: Set[str] = set()

# MQA cache — reload at most once per 60 seconds
_MQA_CACHE_LOADED_AT: float = 0.0
_MQA_CACHE_DATA: dict = {}
_MQA_RELOAD_INTERVAL = 60.0


def _load_mqa_cache() -> dict:
    global _MQA_CACHE_LOADED_AT, _MQA_CACHE_DATA
    now = time.time()
    if now - _MQA_CACHE_LOADED_AT < _MQA_RELOAD_INTERVAL:
        return _MQA_CACHE_DATA
    try:
        raw = _MQA_CACHE_PATH.read_text(encoding="utf-8")
        _MQA_CACHE_DATA = json.loads(raw)
    except Exception:
        _MQA_CACHE_DATA = {}
    _MQA_CACHE_LOADED_AT = now
    return _MQA_CACHE_DATA


def _is_mqa_blocked(symbol: str) -> bool:
    """Return True if the MQA cache shows this symbol is NOT GOOD (score < 40).
    Checks the current verdict directly — no timer, no expiry.
    Only unblocked when MQA gives the symbol a GOOD verdict (score >= 40).
    If the symbol has never been analyzed, allow it (return False).
    """
    try:
        cache = _load_mqa_cache()
        entry = cache.get(symbol)
        if not entry:
            return False  # not yet analyzed → allow
        verdict = (entry.get("verdict") or "").upper()
        # Block anything that is not explicitly GOOD
        # Block SUSPECT and MANIPULATED — signal still comes through (main.py signal path no longer calls this)
        # Only GOOD coins open real trades
        if verdict in ("SUSPECT", "MANIPULATED"):
            return True
        return False
    except Exception:
        return False


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _ensure_file() -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _STATE_PATH.exists():
        _STATE_PATH.write_text("[]", encoding="utf-8")


def _read_symbols_from_disk() -> Set[str]:
    _ensure_file()
    try:
        raw = _STATE_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return set()
        data = json.loads(raw)
        if not isinstance(data, list):
            return set()
        out: Set[str] = set()
        for item in data:
            sym = _normalize_symbol(str(item))
            if sym:
                out.add(sym)
        return out
    except Exception:
        return set()


def _write_symbols_to_disk(symbols: Iterable[str]) -> None:
    clean = sorted({_normalize_symbol(s) for s in symbols if _normalize_symbol(s)})
    _ensure_file()
    _STATE_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    global _CACHE_MTIME, _CACHE_SYMBOLS
    try:
        _CACHE_MTIME = _STATE_PATH.stat().st_mtime
    except Exception:
        _CACHE_MTIME = None
    _CACHE_SYMBOLS = set(clean)


def _load_symbols_cached() -> Set[str]:
    global _CACHE_MTIME, _CACHE_SYMBOLS
    _ensure_file()
    try:
        current_mtime = _STATE_PATH.stat().st_mtime
    except Exception:
        current_mtime = None
    if _CACHE_MTIME is not None and current_mtime == _CACHE_MTIME:
        return set(_CACHE_SYMBOLS)
    symbols = _read_symbols_from_disk()
    _CACHE_SYMBOLS = set(symbols)
    _CACHE_MTIME = current_mtime
    return symbols


def parse_symbols_argument(raw: str) -> List[str]:
    if raw is None:
        return []
    text = str(raw).replace("\n", ",")
    parts = [p.strip() for p in text.split(",")]
    out: List[str] = []
    seen: Set[str] = set()
    for p in parts:
        sym = _normalize_symbol(p)
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def list_blocked_symbols() -> List[str]:
    return sorted(_load_symbols_cached())


def is_symbol_blocked(symbol: str) -> bool:
    sym = _normalize_symbol(symbol)
    if not sym:
        return False
    if sym in _load_symbols_cached():
        return True
    return _is_mqa_blocked(sym)


def add_blocked_symbols(symbols: Sequence[str]) -> Tuple[List[str], List[str]]:
    current = _load_symbols_cached()
    added: List[str] = []
    already: List[str] = []
    for s in symbols:
        sym = _normalize_symbol(s)
        if not sym:
            continue
        if sym in current:
            already.append(sym)
        else:
            current.add(sym)
            added.append(sym)
    _write_symbols_to_disk(current)
    return sorted(added), sorted(already)


def remove_blocked_symbols(symbols: Sequence[str]) -> Tuple[List[str], List[str]]:
    current = _load_symbols_cached()
    removed: List[str] = []
    missing: List[str] = []
    for s in symbols:
        sym = _normalize_symbol(s)
        if not sym:
            continue
        if sym in current:
            current.remove(sym)
            removed.append(sym)
        else:
            missing.append(sym)
    _write_symbols_to_disk(current)
    return sorted(removed), sorted(missing)

def get_mqa_info(symbol: str) -> dict:
    """Return MQ verdict and score for display in signal messages.
    Returns dict with keys: verdict (GOOD/SUSPECT/MANIPULATED or None), score (int or None).
    """
    try:
        cache = _load_mqa_cache()
        sym = _normalize_symbol(symbol)
        entry = cache.get(sym)
        if not entry:
            return {"verdict": None, "score": None}
        verdict = (entry.get("verdict") or "").upper()
        score = entry.get("score")
        if score is not None:
            try:
                score = int(score)
            except Exception:
                score = None
        return {"verdict": verdict or None, "score": score}
    except Exception:
        return {"verdict": None, "score": None}
