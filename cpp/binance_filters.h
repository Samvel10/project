#pragma once
// binance_filters.h — Price/quantity adjustment functions.
// 1:1 behavior match with Python functions in binance_futures.py.
// Pure functions, no global state, no dynamic allocation per call.

#include <cmath>
#include <string>
#include <cstdint>

namespace binance {

// Mirrors Python: _tick_decimals(tick) -> int
// Returns the number of decimal places implied by a tick/step size.
// Python behavior:
//   if tick <= 0 or tick >= 1: return 0
//   s = f"{tick:.15f}".rstrip('0')
//   if '.' in s: return len(s.split('.')[1])
//   return 0
int tick_decimals(double tick);

// Mirrors Python: adjust_price(symbol, price) -> float
// Rounds price DOWN to nearest tick_size multiple.
// If tick_size <= 0, returns price as-is.
// If price <= 0, returns 0.0.
// Python: math.floor(p / tick) * tick
double adjust_price(double price, double tick_size);

// Mirrors Python: adjust_quantity(symbol, requested_qty, price) -> float
// 1) Floor by step_size
// 2) Ensure >= min_qty
// 3) Ensure qty * price >= min_notional (ceil up if needed)
// Returns 0.0 if valid quantity cannot be constructed.
// hard_min_notional: 5.0 for USDT pairs, 0.0 otherwise (caller provides).
double adjust_quantity(double requested_qty, double price,
                       double step_size, double min_qty,
                       double min_notional, double hard_min_notional);

// Mirrors Python: adjust_close_quantity(symbol, requested_qty, round_up=False) -> float
// Like adjust_quantity but WITHOUT min_notional enforcement.
// Used for reduce-only exits.
// round_up=true: ceil to step_size. round_up=false: floor to step_size.
double adjust_close_quantity(double requested_qty, double step_size,
                             double min_qty, bool round_up);

// Mirrors Python: format with f"{value:.{decimals}f}"
// Formats a double to a string with exactly `decimals` decimal places.
// No dynamic allocation: writes to caller-provided buffer.
// Returns number of chars written (excluding null terminator).
int format_decimal(char* buf, size_t buf_size, double value, int decimals);

// Convenience: returns std::string.
std::string format_decimal_str(double value, int decimals);

} // namespace binance
