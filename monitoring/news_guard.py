from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from monitoring.logger import log

try:
    import tradingeconomics as te  # type: ignore[import]

    _TE_AVAILABLE = True
except ImportError:
    te = None  # type: ignore[assignment]
    _TE_AVAILABLE = False

try:
    import investpy  # type: ignore[import]

    _INVESTPY_AVAILABLE = True
except ImportError:
    investpy = None  # type: ignore[assignment]
    _INVESTPY_AVAILABLE = False


# Minutes before and after a high-impact event during which we pause trading.
_BUFFER_BEFORE_MIN = 20
_BUFFER_AFTER_MIN = 20

# Cache economic-calendar responses to avoid hammering the API.
_EVENTS_CACHE: List[Dict[str, Any]] = []
_LAST_EVENTS_FETCH_TS: Optional[float] = None
_EVENTS_CACHE_TTL_SECONDS = 300.0
_EVENTS_ERROR_TTL_SECONDS = 3600.0
_LAST_EVENTS_ERROR_TS: Optional[float] = None
_LAST_EVENTS_ERROR_LOG_TS: Optional[float] = None

_US_EVENTS_CACHE: List[Dict[str, Any]] = []
_LAST_US_EVENTS_FETCH_TS: Optional[float] = None
_US_EVENTS_CACHE_TTL_SECONDS = 300.0
_US_EVENTS_ERROR_TTL_SECONDS = 3600.0
_LAST_US_EVENTS_ERROR_TS: Optional[float] = None
_LAST_US_EVENTS_ERROR_LOG_TS: Optional[float] = None

# Only consider news from these countries/regions. Extend as needed.
_IMPORTANT_COUNTRIES = {
    "United States",
    "Euro Zone",
    "Germany",
    "United Kingdom",
}

# Internal flags so that we do not spam logs.
_TE_IMPORT_ERROR_LOGGED = False
_TE_LOGIN_OK = False
_TE_LOGIN_FAILED_LOGGED = False
_INVESTPY_IMPORT_ERROR_LOGGED = False


def _log_import_warning_once() -> None:
    global _TE_IMPORT_ERROR_LOGGED
    if _TE_AVAILABLE:
        return
    if _TE_IMPORT_ERROR_LOGGED:
        return
    _TE_IMPORT_ERROR_LOGGED = True
    try:
        log(
            "[NEWS GUARD] tradingeconomics package not installed; "
            "news-based trading pause is DISABLED. Install it with 'pip install tradingeconomics'."
        )
    except Exception:
        pass


def _log_investpy_import_warning_once() -> None:
    global _INVESTPY_IMPORT_ERROR_LOGGED
    if _INVESTPY_AVAILABLE:
        return
    if _INVESTPY_IMPORT_ERROR_LOGGED:
        return
    _INVESTPY_IMPORT_ERROR_LOGGED = True
    try:
        log(
            "[NEWS GUARD] investpy package not installed; "
            "news-based trading pause is DISABLED. Install it with 'pip install investpy'."
        )
    except Exception:
        pass


def _ensure_login() -> bool:
    global _TE_LOGIN_OK, _TE_LOGIN_FAILED_LOGGED

    if not _TE_AVAILABLE or te is None:  # type: ignore[truthy-function]
        _log_import_warning_once()
        return False

    if _TE_LOGIN_OK:
        return True

    try:
        # Public demo credentials documented by TradingEconomics.
        te.login("guest:guest")  # type: ignore[call-arg]
        _TE_LOGIN_OK = True
        return True
    except Exception as e:  # pragma: no cover - best-effort logging
        if not _TE_LOGIN_FAILED_LOGGED:
            _TE_LOGIN_FAILED_LOGGED = True
            try:
                log(f"[NEWS GUARD] TradingEconomics login failed; news guard disabled: {e}")
            except Exception:
                pass
        return False


def _parse_event_time(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        dt = raw
    else:
        if raw is None:
            return None
        try:
            s = str(raw).strip()
            if not s:
                return None
            # Normalise potential trailing Z to an explicit UTC offset.
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
        except Exception:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _get_today_high_impact_events() -> List[Dict[str, Any]]:
    """Fetch and cache today's high-impact macro events from TradingEconomics.

    Returns a list of dicts with keys: time (UTC datetime), country, event.
    If the API or library is unavailable, returns an empty list.
    """

    global _EVENTS_CACHE, _LAST_EVENTS_FETCH_TS, _LAST_EVENTS_ERROR_TS, _LAST_EVENTS_ERROR_LOG_TS

    if not _INVESTPY_AVAILABLE or investpy is None:  # type: ignore[truthy-function]
        _log_investpy_import_warning_once()
        return []

    now_ts = time.time()
    if (
        _EVENTS_CACHE
        and _LAST_EVENTS_FETCH_TS is not None
        and (now_ts - _LAST_EVENTS_FETCH_TS) < _EVENTS_CACHE_TTL_SECONDS
    ):
        return _EVENTS_CACHE

    if (
        _LAST_EVENTS_ERROR_TS is not None
        and (now_ts - _LAST_EVENTS_ERROR_TS) < _EVENTS_ERROR_TTL_SECONDS
    ):
        return []

    try:
        today = datetime.now(timezone.utc).date()
        from_date_str = today.strftime("%d/%m/%Y")
        # investpy.economic_calendar պահանջում է, որ to_date > from_date,
        # ուստի օգտագործում ենք հաջորդ օրվա ամսաթիվը որպես to_date:
        to_date_str = (today + timedelta(days=1)).strftime("%d/%m/%Y")

        countries = [c.lower() for c in _IMPORTANT_COUNTRIES]

        df = investpy.economic_calendar(  # type: ignore[call-arg]
            time_zone="GMT",
            time_filter="time_only",
            countries=countries,
            importances=["high"],
            from_date=from_date_str,
            to_date=to_date_str,
        )
    except Exception as e:  # pragma: no cover - network/API failure path
        _LAST_EVENTS_ERROR_TS = now_ts
        try:
            should_log = True
            if _LAST_EVENTS_ERROR_LOG_TS is not None and (now_ts - _LAST_EVENTS_ERROR_LOG_TS) < _EVENTS_ERROR_TTL_SECONDS:
                should_log = False
            if should_log:
                _LAST_EVENTS_ERROR_LOG_TS = now_ts
                log(f"[NEWS GUARD] Failed to fetch economic calendar from Investing.com: {e}")
        except Exception:
            pass
        return []

    if df is None:
        return []

    try:
        rows = df.to_dict("records")  # type: ignore[assignment]
    except Exception:
        rows = []

    results: List[Dict[str, Any]] = []
    for e in rows or []:  # type: ignore[union-attr]
        try:
            if not isinstance(e, dict):
                continue

            importance = (e.get("importance") or e.get("Importance") or "").strip().lower()
            if importance != "high":
                continue

            zone_val = e.get("zone") or e.get("country") or e.get("Country")
            country = (zone_val or "").strip()
            if _IMPORTANT_COUNTRIES and country not in _IMPORTANT_COUNTRIES:
                continue

            date_str = (e.get("date") or e.get("Date") or "").strip()
            time_str = (e.get("time") or e.get("Time") or "").strip()
            if not date_str:
                continue

            try:
                if time_str and time_str not in ("All Day", "Tentative"):
                    dt_naive = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
                else:
                    dt_naive = datetime.strptime(date_str, "%d/%m/%Y")
            except Exception:
                continue

            dt = dt_naive.replace(tzinfo=timezone.utc)

            results.append(
                {
                    "time": dt,
                    "country": country,
                    "event": e.get("event") or e.get("Event") or "",
                }
            )
        except Exception:
            continue

    _EVENTS_CACHE = results
    _LAST_EVENTS_FETCH_TS = now_ts
    return results


def get_active_news_events() -> List[Dict[str, Any]]:
    """Return high-impact events that are currently inside the guard window.

    Guard window is defined as [event_time - BUFFER_BEFORE, event_time + BUFFER_AFTER].
    """

    events = _get_today_high_impact_events()
    if not events:
        return []

    now = datetime.now(timezone.utc)
    before = timedelta(minutes=_BUFFER_BEFORE_MIN)
    after = timedelta(minutes=_BUFFER_AFTER_MIN)

    active: List[Dict[str, Any]] = []
    for ev in events:
        t = ev.get("time")
        if not isinstance(t, datetime):
            continue
        if now - before <= t <= now + after:
            active.append(ev)

    return active


def is_news_window_active() -> bool:
    """Convenience helper: True if at least one active high-impact event is near now."""

    return bool(get_active_news_events())


def get_us_high_impact_events_for_today_window() -> List[Dict[str, Any]]:
    global _US_EVENTS_CACHE, _LAST_US_EVENTS_FETCH_TS, _LAST_US_EVENTS_ERROR_TS, _LAST_US_EVENTS_ERROR_LOG_TS

    events: List[Dict[str, Any]] = []

    if not _INVESTPY_AVAILABLE or investpy is None:  # type: ignore[truthy-function]
        _log_investpy_import_warning_once()
        return events

    now_ts = time.time()
    if (
        _US_EVENTS_CACHE
        and _LAST_US_EVENTS_FETCH_TS is not None
        and (now_ts - _LAST_US_EVENTS_FETCH_TS) < _US_EVENTS_CACHE_TTL_SECONDS
    ):
        return _US_EVENTS_CACHE

    if (
        _LAST_US_EVENTS_ERROR_TS is not None
        and (now_ts - _LAST_US_EVENTS_ERROR_TS) < _US_EVENTS_ERROR_TTL_SECONDS
    ):
        return []

    try:
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        start_dt = datetime(today.year, today.month, today.day, 0, 0, tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=1, hours=12)

        from_date_str = start_dt.strftime("%d/%m/%Y")
        to_date_str = (start_dt + timedelta(days=1)).strftime("%d/%m/%Y")

        df = investpy.economic_calendar(  # type: ignore[call-arg]
            time_zone="GMT",
            time_filter="time_only",
            countries=["united states"],
            importances=["high"],
            from_date=from_date_str,
            to_date=to_date_str,
        )
    except Exception as e:
        _LAST_US_EVENTS_ERROR_TS = now_ts
        try:
            should_log = True
            if _LAST_US_EVENTS_ERROR_LOG_TS is not None and (now_ts - _LAST_US_EVENTS_ERROR_LOG_TS) < _US_EVENTS_ERROR_TTL_SECONDS:
                should_log = False
            if should_log:
                _LAST_US_EVENTS_ERROR_LOG_TS = now_ts
                log(f"[NEWS GUARD] Failed to fetch US high-impact events from Investing.com: {e}")
        except Exception:
            pass
        return []

    if df is None:
        return events

    try:
        rows = df.to_dict("records")  # type: ignore[assignment]
    except Exception:
        rows = []

    for row in rows or []:  # type: ignore[union-attr]
        try:
            date_str = (row.get("date") or row.get("Date") or "").strip()
            time_str = (row.get("time") or row.get("Time") or "").strip()
            if not date_str:
                continue

            try:
                if time_str and time_str not in ("All Day", "Tentative"):
                    dt_naive = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
                else:
                    dt_naive = datetime.strptime(date_str, "%d/%m/%Y")
            except Exception:
                continue

            dt = dt_naive.replace(tzinfo=timezone.utc)

            # Պահպանում ենք միայն այն իրադարձությունները, որոնք ընկած են
            # [այսօրվա 00:00, վաղվա 12:00] UTC window-ում
            if dt < start_dt or dt > end_dt:
                continue

            # Եվ բացի այդ, հեռացնում ենք արդեն անցած իրադարձությունները,
            # որպեսզի summary-ում մնան միայն ապագա event-ները:
            if dt < now_utc:
                continue

            events.append(
                {
                    "time": dt,
                    "country": "United States",
                    "event": row.get("event") or row.get("Event") or "",
                    "importance": (row.get("importance") or row.get("Importance") or "").strip(),
                    "actual": row.get("actual") or row.get("Actual"),
                    "forecast": row.get("forecast") or row.get("Forecast"),
                    "previous": row.get("previous") or row.get("Previous"),
                }
            )
        except Exception:
            continue

    _US_EVENTS_CACHE = events
    _LAST_US_EVENTS_FETCH_TS = now_ts
    return events
