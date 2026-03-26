def compute_sl_tp(entry, atr, direction, rr_levels=(1, 2, 3), sl_pct=None, tp_pcts=None):
    # For very small ATR values (e.g., micro-priced symbols), use a minimum
    # effective ATR so that SL/TP are meaningfully away from entry.
    # This avoids degenerate cases like entry == SL == all TP levels.
    if atr is None:
        atr = 0.0

    # If explicit percent-based config is provided, use that instead of ATR.
    if sl_pct is not None or tp_pcts is not None:
        try:
            base_sl_pct = float(sl_pct) if sl_pct is not None else 2.0
        except (TypeError, ValueError):
            base_sl_pct = 2.0

        raw_tp_pcts = tp_pcts if tp_pcts is not None else rr_levels
        factors = []
        for v in raw_tp_pcts:
            try:
                fv = float(v)
                if fv > 0:
                    factors.append(fv)
            except (TypeError, ValueError):
                continue
        if not factors:
            # Fallback to simple 1,2,3% if tp_pcts are unusable
            factors = [1.0, 2.0, 3.0]

        sl_factor = abs(base_sl_pct) / 100.0
        tp_factors = [abs(f) / 100.0 for f in factors]

        if direction == "LONG":
            sl = entry * (1.0 - sl_factor)
            tps = [entry * (1.0 + f) for f in tp_factors]
        else:
            sl = entry * (1.0 + sl_factor)
            tps = [entry * (1.0 - f) for f in tp_factors]

        return sl, tps

    # Minimum effective ATR: about 2% of price, so that SL/TP are not too tight
    # This avoids degenerate cases like entry == SL == all TP levels.
    min_atr = max(abs(entry) * 0.02, 1e-8)
    eff_atr = atr if atr > min_atr else min_atr

    if direction == "LONG":
        sl = entry - eff_atr
        tps = [entry + eff_atr * rr for rr in rr_levels]
    else:
        sl = entry + eff_atr
        tps = [entry - eff_atr * rr for rr in rr_levels]

    return sl, tps
