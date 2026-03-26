#include "binance_filters.h"
#include <cstdio>
#include <cstring>
#include <cmath>

namespace binance {

// Mirrors Python:
//   if tick <= 0 or tick >= 1: return 0
//   s = f"{tick:.15f}".rstrip('0')
//   if '.' in s: return len(s.split('.')[1])
//   return 0
int tick_decimals(double tick) {
    if (tick <= 0.0 || tick >= 1.0)
        return 0;

    // Format with 15 decimal places, same as Python f"{tick:.15f}"
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.15f", tick);

    // Find the decimal point
    char* dot = std::strchr(buf, '.');
    if (!dot)
        return 0;

    // Find last non-zero character after decimal point (rstrip('0'))
    size_t len = std::strlen(dot + 1);
    while (len > 0 && dot[len] == '0')
        --len;

    return static_cast<int>(len);
}

// Mirrors Python:
//   p = float(price)
//   if p <= 0: return 0.0
//   if tick_size <= 0: return p   (caller passes 0.0 if no filter)
//   p = math.floor(p / tick) * tick
double adjust_price(double price, double tick_size) {
    if (price <= 0.0)
        return 0.0;
    if (tick_size <= 0.0)
        return price;

    return std::floor(price / tick_size) * tick_size;
}

// Mirrors Python adjust_quantity():
//   1) Floor by step_size
//   2) Ensure >= min_qty
//   3) Ensure qty * price >= min_notional (ceil up)
//   Returns 0.0 if invalid.
//
// hard_min_notional: Python uses 5.0 for USDT pairs. Caller provides this.
// If min_notional < hard_min_notional, use hard_min_notional.
double adjust_quantity(double requested_qty, double price,
                       double step_size, double min_qty,
                       double min_notional, double hard_min_notional) {
    if (requested_qty <= 0.0 || price <= 0.0)
        return 0.0;

    double qty = requested_qty;

    // Apply hard_min_notional override (Python lines 3499-3500)
    double eff_min_notional = min_notional;
    if (hard_min_notional > 0.0 &&
        (eff_min_notional <= 0.0 || eff_min_notional < hard_min_notional)) {
        eff_min_notional = hard_min_notional;
    }

    // No filters case (Python lines 3475-3489)
    if (step_size <= 0.0 && min_qty <= 0.0 && eff_min_notional <= 0.0) {
        if (qty <= 0.0)
            return 0.0;
        // Python: math.ceil(qty * 1e8) / 1e8
        qty = std::ceil(qty * 1e8) / 1e8;
        // Python: float(f"{qty:.8f}")
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.8f", qty);
        double result;
        std::sscanf(buf, "%lf", &result);
        return result;
    }

    // 1) Apply stepSize (floor toward zero) — Python line 3503-3504
    if (step_size > 0.0) {
        qty = std::floor(qty / step_size) * step_size;
    }

    // 2) Ensure >= minQty — Python line 3507-3508
    if (min_qty > 0.0 && qty < min_qty) {
        qty = min_qty;
    }

    // 3) Ensure notional >= minNotional — Python lines 3511-3518
    if (eff_min_notional > 0.0) {
        double notional = qty * price;
        if (notional < eff_min_notional) {
            double required_qty = eff_min_notional / price;
            if (step_size > 0.0) {
                qty = std::ceil(required_qty / step_size) * step_size;
            } else {
                qty = required_qty;
            }
        }
    }

    if (qty <= 0.0)
        return 0.0;

    // Python: float(f"{qty:.8f}")
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.8f", qty);
    double result;
    std::sscanf(buf, "%lf", &result);
    return result;
}

// Mirrors Python adjust_close_quantity():
//   Floor/ceil by step_size, enforce min_qty, no min_notional.
double adjust_close_quantity(double requested_qty, double step_size,
                             double min_qty, bool round_up) {
    if (requested_qty <= 0.0)
        return 0.0;

    double qty = requested_qty;

    // No filters: return as-is (Python line 3643-3644: return max(0.0, qty))
    if (step_size <= 0.0 && min_qty <= 0.0) {
        return (qty > 0.0) ? qty : 0.0;
    }

    // Apply stepSize — Python lines 3650-3657
    if (step_size > 0.0) {
        if (round_up) {
            // Python: steps = int(qty / step_size)
            //         if steps * step_size < qty: steps += 1
            //         qty = float(steps) * step_size
            int steps = static_cast<int>(qty / step_size);
            if (static_cast<double>(steps) * step_size < qty)
                steps += 1;
            qty = static_cast<double>(steps) * step_size;
        } else {
            // Python: qty = math.floor(qty / step_size) * step_size
            qty = std::floor(qty / step_size) * step_size;
        }
    }

    // Ensure >= minQty — Python lines 3661-3662
    if (min_qty > 0.0 && qty < min_qty) {
        qty = min_qty;
    }

    if (qty <= 0.0)
        return 0.0;

    // Python: float(f"{qty:.8f}")
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.8f", qty);
    double result;
    std::sscanf(buf, "%lf", &result);
    return result;
}

int format_decimal(char* buf, size_t buf_size, double value, int decimals) {
    if (decimals < 0) decimals = 0;
    if (decimals > 15) decimals = 15;
    return std::snprintf(buf, buf_size, "%.*f", decimals, value);
}

std::string format_decimal_str(double value, int decimals) {
    char buf[64];
    format_decimal(buf, sizeof(buf), value, decimals);
    return std::string(buf);
}

} // namespace binance
