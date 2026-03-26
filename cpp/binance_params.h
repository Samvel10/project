#pragma once
// binance_params.h — Order parameter builders.
// 1:1 behavior match with BinanceFuturesClient method bodies in binance_futures.py.
// Each function builds a ParamList (ordered key-value pairs) that is then
// passed to sign_params() and sent via HttpClient.
//
// These are pure functions: no side effects, no global state.

#include "binance_sign.h"
#include <string>
#include <optional>

namespace binance {

// Mirrors Python: BinanceFuturesClient.place_order()
// Builds parameter list for POST /fapi/v1/order.
//
// Python behavior:
//   params = {"symbol": symbol, "side": side, "type": order_type}
//   if quantity is not None: params["quantity"] = quantity
//   if reduce_only: params["reduceOnly"] = "true"
//   if price is not None: params["price"] = price
//   if stop_price is not None: params["stopPrice"] = stop_price
//   if close_position is not None: params["closePosition"] = str(close_position).lower()
//   if time_in_force is not None: params["timeInForce"] = time_in_force
//   if working_type is not None: params["workingType"] = working_type
//   if position_side is not None:
//       ps = str(position_side).upper().strip()
//       if ps: params["positionSide"] = ps
ParamList build_place_order_params(
    const std::string& symbol,
    const std::string& side,
    const std::string& order_type,          // default "MARKET"
    const std::optional<std::string>& quantity,       // None = omit
    bool reduce_only,                       // default false
    const std::optional<std::string>& price,          // None = omit
    const std::optional<std::string>& stop_price,     // None = omit
    const std::optional<std::string>& close_position, // None = omit; "true"/"false"
    const std::optional<std::string>& time_in_force,  // None = omit
    const std::optional<std::string>& position_side,  // None = omit
    const std::optional<std::string>& working_type    // None = omit
);

// Mirrors Python: BinanceFuturesClient.place_algo_order()
// Builds parameter list for POST /fapi/v1/algoOrder.
//
// Python behavior:
//   params = {"symbol": symbol, "side": side, "algoType": algo_type, "type": order_type}
//   use_close_position = bool(close_position)
//   if use_close_position:
//       params["closePosition"] = "true"
//   else:
//       if quantity is not None: params["quantity"] = quantity
//       if reduce_only: params["reduceOnly"] = "true"
//   if trigger_price is not None: params["triggerPrice"] = trigger_price
//   if working_type is not None: params["workingType"] = working_type
//   if time_in_force is not None: params["timeInForce"] = time_in_force
//   if position_side is not None:
//       ps = str(position_side).upper().strip()
//       if ps: params["positionSide"] = ps
ParamList build_place_algo_order_params(
    const std::string& symbol,
    const std::string& side,
    const std::string& algo_type,           // default "CONDITIONAL"
    const std::string& order_type,          // default "STOP_MARKET"
    const std::optional<std::string>& quantity,
    bool reduce_only,
    const std::optional<std::string>& trigger_price,
    bool close_position,                    // bool, not optional string
    const std::optional<std::string>& working_type,
    const std::optional<std::string>& time_in_force,
    const std::optional<std::string>& position_side
);

// Mirrors Python: BinanceFuturesClient.cancel_order()
// Builds parameter list for DELETE /fapi/v1/order.
//
// Python behavior:
//   params = {"symbol": str(symbol).upper()}
//   if order_id is not None: params["orderId"] = int(order_id)
//   if orig_client_order_id is not None: params["origClientOrderId"] = str(...)
ParamList build_cancel_order_params(
    const std::string& symbol,
    const std::optional<int64_t>& order_id,
    const std::optional<std::string>& orig_client_order_id
);

// Mirrors Python: BinanceFuturesClient.cancel_algo_order()
// Builds parameter list for DELETE /fapi/v1/algoOrder.
//
// Python: params = {"algoId": int(algo_id)}
ParamList build_cancel_algo_order_params(int64_t algo_id);

// Mirrors Python: BinanceFuturesClient.open_orders()
// Builds parameter list for GET /fapi/v1/openOrders.
//
// Python: params = {}; if symbol: params["symbol"] = str(symbol).upper()
ParamList build_open_orders_params(const std::optional<std::string>& symbol);

// Mirrors Python: BinanceFuturesClient.get_algo_open_orders()
// Builds parameter list for GET /fapi/v1/openAlgoOrders.
ParamList build_algo_open_orders_params(const std::optional<std::string>& symbol);

} // namespace binance
