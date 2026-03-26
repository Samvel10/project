#include "binance_params.h"
#include <algorithm>
#include <cctype>

namespace binance {

// Helper: uppercase + strip a string (mirrors Python str.upper().strip())
static std::string upper_strip(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    // Strip leading whitespace
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
        ++start;
    // Strip trailing whitespace
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
        --end;
    // Uppercase
    for (size_t i = start; i < end; ++i)
        result += static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
    return result;
}

// Mirrors Python BinanceFuturesClient.place_order() body (lines 706-735)
ParamList build_place_order_params(
    const std::string& symbol,
    const std::string& side,
    const std::string& order_type,
    const std::optional<std::string>& quantity,
    bool reduce_only,
    const std::optional<std::string>& price,
    const std::optional<std::string>& stop_price,
    const std::optional<std::string>& close_position,
    const std::optional<std::string>& time_in_force,
    const std::optional<std::string>& position_side,
    const std::optional<std::string>& working_type)
{
    ParamList params;
    params.reserve(12);

    // Python: params = {"symbol": symbol, "side": side, "type": order_type}
    params.emplace_back("symbol", symbol);
    params.emplace_back("side", side);
    params.emplace_back("type", order_type);

    // Python: if quantity is not None: params["quantity"] = quantity
    if (quantity.has_value())
        params.emplace_back("quantity", quantity.value());

    // Python: if reduce_only: params["reduceOnly"] = "true"
    if (reduce_only)
        params.emplace_back("reduceOnly", "true");

    // Python: if price is not None: params["price"] = price
    if (price.has_value())
        params.emplace_back("price", price.value());

    // Python: if stop_price is not None: params["stopPrice"] = stop_price
    if (stop_price.has_value())
        params.emplace_back("stopPrice", stop_price.value());

    // Python: if close_position is not None: params["closePosition"] = str(close_position).lower()
    if (close_position.has_value())
        params.emplace_back("closePosition", close_position.value());

    // Python: if time_in_force is not None: params["timeInForce"] = time_in_force
    if (time_in_force.has_value())
        params.emplace_back("timeInForce", time_in_force.value());

    // Python: if working_type is not None: params["workingType"] = working_type
    if (working_type.has_value())
        params.emplace_back("workingType", working_type.value());

    // Python: if position_side is not None:
    //     ps = str(position_side).upper().strip()
    //     if ps: params["positionSide"] = ps
    if (position_side.has_value()) {
        std::string ps = upper_strip(position_side.value());
        if (!ps.empty())
            params.emplace_back("positionSide", ps);
    }

    return params;
}

// Mirrors Python BinanceFuturesClient.place_algo_order() body (lines 751-782)
ParamList build_place_algo_order_params(
    const std::string& symbol,
    const std::string& side,
    const std::string& algo_type,
    const std::string& order_type,
    const std::optional<std::string>& quantity,
    bool reduce_only,
    const std::optional<std::string>& trigger_price,
    bool close_position,
    const std::optional<std::string>& working_type,
    const std::optional<std::string>& time_in_force,
    const std::optional<std::string>& position_side)
{
    ParamList params;
    params.reserve(10);

    // Python: params = {"symbol": symbol, "side": side, "algoType": algo_type, "type": order_type}
    params.emplace_back("symbol", symbol);
    params.emplace_back("side", side);
    params.emplace_back("algoType", algo_type);
    params.emplace_back("type", order_type);

    // Python:
    //   use_close_position = bool(close_position)
    //   if use_close_position:
    //       params["closePosition"] = "true"
    //   else:
    //       if quantity is not None: params["quantity"] = quantity
    //       if reduce_only: params["reduceOnly"] = "true"
    if (close_position) {
        params.emplace_back("closePosition", "true");
    } else {
        if (quantity.has_value())
            params.emplace_back("quantity", quantity.value());
        if (reduce_only)
            params.emplace_back("reduceOnly", "true");
    }

    // Python: if trigger_price is not None: params["triggerPrice"] = trigger_price
    if (trigger_price.has_value())
        params.emplace_back("triggerPrice", trigger_price.value());

    // Python: if working_type is not None: params["workingType"] = working_type
    if (working_type.has_value())
        params.emplace_back("workingType", working_type.value());

    // Python: if time_in_force is not None: params["timeInForce"] = time_in_force
    if (time_in_force.has_value())
        params.emplace_back("timeInForce", time_in_force.value());

    // Python: if position_side is not None:
    //     ps = str(position_side).upper().strip()
    //     if ps: params["positionSide"] = ps
    if (position_side.has_value()) {
        std::string ps = upper_strip(position_side.value());
        if (!ps.empty())
            params.emplace_back("positionSide", ps);
    }

    return params;
}

// Mirrors Python BinanceFuturesClient.cancel_order() body (lines 662-667)
ParamList build_cancel_order_params(
    const std::string& symbol,
    const std::optional<int64_t>& order_id,
    const std::optional<std::string>& orig_client_order_id)
{
    ParamList params;
    params.reserve(3);

    // Python: params = {"symbol": str(symbol).upper()}
    std::string sym_upper = symbol;
    for (auto& c : sym_upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    params.emplace_back("symbol", sym_upper);

    // Python: if order_id is not None: params["orderId"] = int(order_id)
    if (order_id.has_value())
        params.emplace_back("orderId", std::to_string(order_id.value()));

    // Python: if orig_client_order_id is not None: params["origClientOrderId"] = str(...)
    if (orig_client_order_id.has_value())
        params.emplace_back("origClientOrderId", orig_client_order_id.value());

    return params;
}

// Mirrors Python BinanceFuturesClient.cancel_algo_order() body (line 786)
ParamList build_cancel_algo_order_params(int64_t algo_id) {
    // Python: params = {"algoId": int(algo_id)}
    ParamList params;
    params.emplace_back("algoId", std::to_string(algo_id));
    return params;
}

// Mirrors Python BinanceFuturesClient.open_orders() body (lines 651-654)
ParamList build_open_orders_params(const std::optional<std::string>& symbol) {
    ParamList params;
    // Python: params = {}; if symbol: params["symbol"] = str(symbol).upper()
    if (symbol.has_value()) {
        std::string sym_upper = symbol.value();
        for (auto& c : sym_upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        params.emplace_back("symbol", sym_upper);
    }
    return params;
}

// Mirrors Python BinanceFuturesClient.get_algo_open_orders() body (lines 791-794)
ParamList build_algo_open_orders_params(const std::optional<std::string>& symbol) {
    ParamList params;
    if (symbol.has_value()) {
        std::string sym_upper = symbol.value();
        for (auto& c : sym_upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        params.emplace_back("symbol", sym_upper);
    }
    return params;
}

} // namespace binance
