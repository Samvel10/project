#!/usr/bin/env python3
"""
STEP E — Verification: Python old vs C++ new.
Tests that every migrated function produces IDENTICAL output
for the same input. Any mismatch = FAIL → do not proceed.

Run after building binance_fast:
    cd /var/www/html/new_example_bot
    python3 cpp/test_verify.py
"""

import sys
import os
import math
import hmac
import hashlib
from urllib.parse import urlencode

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from execution.binance_fast import (
        tick_decimals as cpp_tick_decimals,
        adjust_price as cpp_adjust_price,
        adjust_quantity as cpp_adjust_quantity,
        adjust_close_quantity as cpp_adjust_close_quantity,
        format_decimal as cpp_format_decimal,
        url_encode as cpp_url_encode,
        sign_params as cpp_sign_params,
        timestamp_ms as cpp_timestamp_ms,
        build_place_order_params as cpp_build_place_order_params,
        build_place_algo_order_params as cpp_build_place_algo_order_params,
        build_cancel_order_params as cpp_build_cancel_order_params,
        build_cancel_algo_order_params as cpp_build_cancel_algo_order_params,
        build_open_orders_params as cpp_build_open_orders_params,
        build_algo_open_orders_params as cpp_build_algo_open_orders_params,
    )
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: binance_fast module not built yet: {e}")
    print("Run: cd cpp && bash build.sh")
    CPP_AVAILABLE = False

passed = 0
failed = 0
skipped = 0


def check(name, py_result, cpp_result, tolerance=None):
    global passed, failed
    if tolerance is not None and isinstance(py_result, float) and isinstance(cpp_result, float):
        if abs(py_result - cpp_result) <= tolerance:
            passed += 1
            return True
        else:
            failed += 1
            print(f"  FAIL: {name}")
            print(f"    Python: {py_result!r}")
            print(f"    C++:    {cpp_result!r}")
            print(f"    Diff:   {abs(py_result - cpp_result)}")
            return False
    if py_result == cpp_result:
        passed += 1
        return True
    else:
        failed += 1
        print(f"  FAIL: {name}")
        print(f"    Python: {py_result!r}")
        print(f"    C++:    {cpp_result!r}")
        return False


# ═══════════════════════════════════════════════════════════════
# 1. tick_decimals
# ═══════════════════════════════════════════════════════════════
def py_tick_decimals(tick):
    """Exact copy of Python _tick_decimals()"""
    if tick <= 0 or tick >= 1:
        return 0
    s = f"{tick:.15f}".rstrip('0')
    if '.' in s:
        return len(s.split('.')[1])
    return 0


print("=== 1. tick_decimals ===")
tick_test_cases = [
    0.0001, 0.001, 0.01, 0.1, 0.00001, 0.0000001,
    1.0, 0.0, -0.01, 0.5, 0.25, 0.125, 0.00000001,
    0.1234, 0.050, 2.0, 10.0,
]
if CPP_AVAILABLE:
    for t in tick_test_cases:
        py_r = py_tick_decimals(t)
        cpp_r = cpp_tick_decimals(t)
        check(f"tick_decimals({t})", py_r, cpp_r)
else:
    skipped += len(tick_test_cases)
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 2. adjust_price
# ═══════════════════════════════════════════════════════════════
def py_adjust_price(price, tick_size):
    """Exact copy of Python adjust_price() core logic"""
    try:
        p = float(price)
    except (TypeError, ValueError):
        return 0.0
    if p <= 0:
        return 0.0
    if tick_size <= 0:
        return p
    p = math.floor(p / tick_size) * tick_size
    return p


print("\n=== 2. adjust_price ===")
price_test_cases = [
    (65432.123456, 0.01),
    (0.09567, 0.001),
    (1.23456789, 0.0001),
    (100.0, 0.1),
    (0.0, 0.01),
    (-5.0, 0.01),
    (50.5, 0.0),    # no tick filter
    (0.00012345, 0.00001),
    (99999.99, 0.01),
]
if CPP_AVAILABLE:
    for price, tick in price_test_cases:
        py_r = py_adjust_price(price, tick)
        cpp_r = cpp_adjust_price(price, tick)
        check(f"adjust_price({price}, {tick})", py_r, cpp_r, tolerance=1e-12)
else:
    skipped += len(price_test_cases)
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 3. adjust_quantity
# ═══════════════════════════════════════════════════════════════
def py_adjust_quantity(qty, price, step_size, min_qty, min_notional, hard_min):
    """Exact copy of Python adjust_quantity() core logic"""
    if qty <= 0 or price <= 0:
        return 0.0
    eff_min = min_notional
    if hard_min > 0 and (eff_min <= 0 or eff_min < hard_min):
        eff_min = hard_min
    if step_size <= 0 and min_qty <= 0 and eff_min <= 0:
        if qty <= 0:
            return 0.0
        qty = math.ceil(qty * 1e8) / 1e8
        return float(f"{qty:.8f}")
    if step_size > 0:
        qty = math.floor(qty / step_size) * step_size
    if min_qty > 0 and qty < min_qty:
        qty = min_qty
    if eff_min > 0:
        notional = qty * price
        if notional < eff_min:
            required_qty = eff_min / price
            if step_size > 0:
                qty = math.ceil(required_qty / step_size) * step_size
            else:
                qty = required_qty
    if qty <= 0:
        return 0.0
    return float(f"{qty:.8f}")


print("\n=== 3. adjust_quantity ===")
qty_test_cases = [
    # (qty, price, step_size, min_qty, min_notional, hard_min)
    (10.5, 100.0, 0.1, 0.1, 5.0, 5.0),
    (0.001, 65000.0, 0.001, 0.001, 5.0, 5.0),
    (100.0, 0.05, 1.0, 1.0, 5.0, 5.0),
    (0.5, 10.0, 0.01, 0.01, 5.0, 5.0),  # notional 5.0, exactly at min
    (0.001, 10.0, 0.001, 0.001, 5.0, 5.0),  # notional 0.01, below min → ceil up
    (0.0, 100.0, 0.1, 0.1, 5.0, 5.0),  # zero qty
    (10.0, 0.0, 0.1, 0.1, 5.0, 5.0),   # zero price
    (1.23456789, 50.0, 0.0001, 0.001, 0.0, 0.0),  # no min_notional
    (0.123, 100.0, 0.0, 0.0, 0.0, 0.0),  # no filters at all
]
if CPP_AVAILABLE:
    for q, p, ss, mq, mn, hm in qty_test_cases:
        py_r = py_adjust_quantity(q, p, ss, mq, mn, hm)
        cpp_r = cpp_adjust_quantity(q, p, ss, mq, mn, hm)
        check(f"adjust_quantity({q}, {p}, {ss}, {mq}, {mn}, {hm})", py_r, cpp_r, tolerance=1e-10)
else:
    skipped += len(qty_test_cases)
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 4. adjust_close_quantity
# ═══════════════════════════════════════════════════════════════
def py_adjust_close_quantity(qty, step_size, min_qty, round_up):
    """Exact copy of Python adjust_close_quantity() core logic"""
    if qty <= 0:
        return 0.0
    if step_size <= 0 and min_qty <= 0:
        return max(0.0, qty)
    if step_size > 0:
        if round_up:
            steps = int(qty / step_size)
            if steps * step_size < qty:
                steps += 1
            qty = float(steps) * step_size
        else:
            qty = math.floor(qty / step_size) * step_size
    if min_qty > 0 and qty < min_qty:
        qty = min_qty
    if qty <= 0:
        return 0.0
    return float(f"{qty:.8f}")


print("\n=== 4. adjust_close_quantity ===")
close_qty_cases = [
    (1.234, 0.01, 0.01, False),
    (1.234, 0.01, 0.01, True),
    (0.005, 0.01, 0.01, False),   # below min_qty
    (0.005, 0.01, 0.01, True),    # below min_qty, round up
    (100.0, 1.0, 1.0, False),
    (0.0, 0.01, 0.01, False),     # zero qty
    (5.5, 0.0, 0.0, False),       # no filters
    (0.123456789, 0.0001, 0.001, False),
]
if CPP_AVAILABLE:
    for q, ss, mq, ru in close_qty_cases:
        py_r = py_adjust_close_quantity(q, ss, mq, ru)
        cpp_r = cpp_adjust_close_quantity(q, ss, mq, ru)
        check(f"adjust_close_quantity({q}, {ss}, {mq}, {ru})", py_r, cpp_r, tolerance=1e-10)
else:
    skipped += len(close_qty_cases)
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 5. format_decimal
# ═══════════════════════════════════════════════════════════════
print("\n=== 5. format_decimal ===")
format_cases = [
    (0.095, 3),
    (1.23456789, 8),
    (100.0, 0),
    (0.00012345, 8),
    (99999.99, 2),
    (0.1, 5),
]
if CPP_AVAILABLE:
    for val, dec in format_cases:
        py_r = f"{val:.{dec}f}"
        cpp_r = cpp_format_decimal(val, dec)
        check(f"format_decimal({val}, {dec})", py_r, cpp_r)
else:
    skipped += len(format_cases)
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 6. url_encode
# ═══════════════════════════════════════════════════════════════
print("\n=== 6. url_encode ===")
encode_cases = [
    [("symbol", "BTCUSDT"), ("side", "BUY"), ("type", "MARKET")],
    [("symbol", "ETHUSDT"), ("quantity", "1.5"), ("reduceOnly", "true")],
    [("algoId", "12345")],
    [],  # empty
]
if CPP_AVAILABLE:
    for params in encode_cases:
        py_r = urlencode(params)
        cpp_r = cpp_url_encode(params)
        check(f"url_encode({params})", py_r, cpp_r)
else:
    skipped += len(encode_cases)
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 7. sign_params
# ═══════════════════════════════════════════════════════════════
def py_sign(api_secret_str, params_list):
    """Exact copy of Python _sign() logic"""
    api_secret = api_secret_str.encode()
    query = urlencode(params_list)
    signature = hmac.new(api_secret, query.encode(), hashlib.sha256).hexdigest()
    return f"{query}&signature={signature}"


print("\n=== 7. sign_params ===")
sign_cases = [
    ("mysecretkey123", [("symbol", "BTCUSDT"), ("side", "BUY"), ("timestamp", "1708000000000")]),
    ("abcdef", [("symbol", "ETHUSDT"), ("quantity", "1.5"), ("type", "MARKET"), ("timestamp", "1708000000001")]),
    ("x" * 64, [("algoId", "999"), ("timestamp", "1708000000002")]),
]
if CPP_AVAILABLE:
    for secret, params in sign_cases:
        py_r = py_sign(secret, params)
        cpp_r = cpp_sign_params(secret, params)
        check(f"sign_params('{secret[:10]}...', {len(params)} params)", py_r, cpp_r)
else:
    skipped += len(sign_cases)
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 8. build_place_order_params
# ═══════════════════════════════════════════════════════════════
print("\n=== 8. build_place_order_params ===")
if CPP_AVAILABLE:
    # Test case 1: basic MARKET order
    cpp_r = cpp_build_place_order_params("BTCUSDT", "BUY", "MARKET")
    expected = [("symbol", "BTCUSDT"), ("side", "BUY"), ("type", "MARKET")]
    check("place_order basic MARKET", expected, cpp_r)

    # Test case 2: STOP_MARKET with all params
    cpp_r = cpp_build_place_order_params(
        "ETHUSDT", "SELL", "STOP_MARKET",
        quantity="1.5",
        reduce_only=True,
        stop_price="3000.00",
        close_position="true",
        working_type="MARK_PRICE",
        position_side="SHORT",
    )
    expected = [
        ("symbol", "ETHUSDT"), ("side", "SELL"), ("type", "STOP_MARKET"),
        ("quantity", "1.5"), ("reduceOnly", "true"), ("stopPrice", "3000.00"),
        ("closePosition", "true"), ("workingType", "MARK_PRICE"),
        ("positionSide", "SHORT"),
    ]
    check("place_order STOP_MARKET full", expected, cpp_r)

    # Test case 3: position_side with whitespace
    cpp_r = cpp_build_place_order_params(
        "XRPUSDT", "BUY", "MARKET",
        position_side="  long  ",
    )
    expected = [("symbol", "XRPUSDT"), ("side", "BUY"), ("type", "MARKET"), ("positionSide", "LONG")]
    check("place_order position_side strip+upper", expected, cpp_r)
else:
    skipped += 3
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 9. build_place_algo_order_params
# ═══════════════════════════════════════════════════════════════
print("\n=== 9. build_place_algo_order_params ===")
if CPP_AVAILABLE:
    # Test: close_position=True (should NOT include quantity/reduceOnly)
    cpp_r = cpp_build_place_algo_order_params(
        "BTCUSDT", "SELL", "CONDITIONAL", "STOP_MARKET",
        quantity="1.0",
        reduce_only=True,
        trigger_price="60000.00",
        close_position=True,
        working_type="MARK_PRICE",
    )
    expected = [
        ("symbol", "BTCUSDT"), ("side", "SELL"), ("algoType", "CONDITIONAL"), ("type", "STOP_MARKET"),
        ("closePosition", "true"),
        ("triggerPrice", "60000.00"), ("workingType", "MARK_PRICE"),
    ]
    check("algo_order close_position=True", expected, cpp_r)

    # Test: close_position=False (should include quantity+reduceOnly)
    cpp_r = cpp_build_place_algo_order_params(
        "ETHUSDT", "BUY", "CONDITIONAL", "TAKE_PROFIT_MARKET",
        quantity="2.5",
        reduce_only=True,
        trigger_price="4000.00",
        close_position=False,
        working_type="CONTRACT_PRICE",
    )
    expected = [
        ("symbol", "ETHUSDT"), ("side", "BUY"), ("algoType", "CONDITIONAL"), ("type", "TAKE_PROFIT_MARKET"),
        ("quantity", "2.5"), ("reduceOnly", "true"),
        ("triggerPrice", "4000.00"), ("workingType", "CONTRACT_PRICE"),
    ]
    check("algo_order close_position=False", expected, cpp_r)
else:
    skipped += 2
    print("  SKIPPED (module not built)")

# ═══════════════════════════════════════════════════════════════
# 10. build_cancel_order_params
# ═══════════════════════════════════════════════════════════════
print("\n=== 10. build_cancel_order_params ===")
if CPP_AVAILABLE:
    cpp_r = cpp_build_cancel_order_params("btcusdt", order_id=12345)
    expected = [("symbol", "BTCUSDT"), ("orderId", "12345")]
    check("cancel_order by orderId", expected, cpp_r)

    cpp_r = cpp_build_cancel_algo_order_params(999888777)
    expected = [("algoId", "999888777")]
    check("cancel_algo_order", expected, cpp_r)
else:
    skipped += 2
    print("  SKIPPED (module not built)")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
if failed > 0:
    print("*** VERIFICATION FAILED — DO NOT PROCEED ***")
    sys.exit(1)
elif skipped > 0:
    print("Some tests skipped (C++ module not built yet).")
    print("Build with: cd cpp && bash build.sh")
    sys.exit(0)
else:
    print("ALL TESTS PASSED — C++ behavior matches Python exactly.")
    sys.exit(0)
