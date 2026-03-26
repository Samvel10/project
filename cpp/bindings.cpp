// bindings.cpp — pybind11 bindings for the C++ Binance execution modules.
// Exposes all functions to Python with the SAME input/output semantics
// as the original Python functions they replace.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "binance_filters.h"
#include "binance_sign.h"
#include "binance_params.h"
#include "binance_http.h"

namespace py = pybind11;

PYBIND11_MODULE(binance_fast, m) {
    m.doc() = "C++ accelerated Binance Futures execution primitives";

    // ─── binance_filters ─────────────────────────────────────────

    m.def("tick_decimals", &binance::tick_decimals,
          py::arg("tick"),
          "Return the number of decimal places implied by a tick/step size.\n"
          "Mirrors Python _tick_decimals().");

    m.def("adjust_price", &binance::adjust_price,
          py::arg("price"), py::arg("tick_size"),
          "Adjust price DOWN to nearest tick_size multiple.\n"
          "Mirrors Python adjust_price(). tick_size=0 means no filter.");

    m.def("adjust_quantity", &binance::adjust_quantity,
          py::arg("requested_qty"), py::arg("price"),
          py::arg("step_size"), py::arg("min_qty"),
          py::arg("min_notional"), py::arg("hard_min_notional"),
          "Adjust quantity: floor by step_size, enforce min_qty and min_notional.\n"
          "Mirrors Python adjust_quantity().");

    m.def("adjust_close_quantity", &binance::adjust_close_quantity,
          py::arg("requested_qty"), py::arg("step_size"),
          py::arg("min_qty"), py::arg("round_up") = false,
          "Adjust close quantity: floor/ceil by step_size, enforce min_qty.\n"
          "Mirrors Python adjust_close_quantity().");

    m.def("format_decimal", &binance::format_decimal_str,
          py::arg("value"), py::arg("decimals"),
          "Format a double to string with exactly N decimal places.\n"
          "Mirrors Python f\"{value:.{decimals}f}\".");

    // ─── binance_sign ────────────────────────────────────────────

    // ParamList is exposed as list of (str, str) tuples in Python.
    // pybind11 auto-converts std::vector<std::pair<std::string, std::string>>.

    m.def("url_encode",
          [](const std::vector<std::pair<std::string, std::string>>& params) {
              return binance::url_encode(params);
          },
          py::arg("params"),
          "URL-encode a list of (key, value) pairs.\n"
          "Mirrors Python urllib.parse.urlencode().");

    m.def("sign_params",
          [](const std::string& api_secret,
             const std::vector<std::pair<std::string, std::string>>& params) {
              return binance::sign_params(api_secret, params);
          },
          py::arg("api_secret"), py::arg("params"),
          "Sign parameters with HMAC-SHA256.\n"
          "Returns 'key=val&...&signature=hex'.\n"
          "Mirrors Python BinanceFuturesClient._sign().");

    m.def("timestamp_ms", &binance::timestamp_ms,
          "Current time in milliseconds (int64).\n"
          "Mirrors Python int(time.time() * 1000).");

    // ─── binance_params ──────────────────────────────────────────

    m.def("build_place_order_params", &binance::build_place_order_params,
          py::arg("symbol"), py::arg("side"), py::arg("order_type"),
          py::arg("quantity") = py::none(),
          py::arg("reduce_only") = false,
          py::arg("price") = py::none(),
          py::arg("stop_price") = py::none(),
          py::arg("close_position") = py::none(),
          py::arg("time_in_force") = py::none(),
          py::arg("position_side") = py::none(),
          py::arg("working_type") = py::none(),
          "Build parameter list for POST /fapi/v1/order.\n"
          "Mirrors BinanceFuturesClient.place_order() param construction.");

    m.def("build_place_algo_order_params", &binance::build_place_algo_order_params,
          py::arg("symbol"), py::arg("side"),
          py::arg("algo_type"), py::arg("order_type"),
          py::arg("quantity") = py::none(),
          py::arg("reduce_only") = false,
          py::arg("trigger_price") = py::none(),
          py::arg("close_position") = false,
          py::arg("working_type") = py::none(),
          py::arg("time_in_force") = py::none(),
          py::arg("position_side") = py::none(),
          "Build parameter list for POST /fapi/v1/algoOrder.\n"
          "Mirrors BinanceFuturesClient.place_algo_order() param construction.");

    m.def("build_cancel_order_params", &binance::build_cancel_order_params,
          py::arg("symbol"),
          py::arg("order_id") = py::none(),
          py::arg("orig_client_order_id") = py::none(),
          "Build parameter list for DELETE /fapi/v1/order.\n"
          "Mirrors BinanceFuturesClient.cancel_order() param construction.");

    m.def("build_cancel_algo_order_params", &binance::build_cancel_algo_order_params,
          py::arg("algo_id"),
          "Build parameter list for DELETE /fapi/v1/algoOrder.\n"
          "Mirrors BinanceFuturesClient.cancel_algo_order() param construction.");

    m.def("build_open_orders_params", &binance::build_open_orders_params,
          py::arg("symbol") = py::none(),
          "Build parameter list for GET /fapi/v1/openOrders.");

    m.def("build_algo_open_orders_params", &binance::build_algo_open_orders_params,
          py::arg("symbol") = py::none(),
          "Build parameter list for GET /fapi/v1/openAlgoOrders.");

    // ─── binance_http ────────────────────────────────────────────

    py::class_<binance::HttpClientConfig>(m, "HttpClientConfig")
        .def(py::init<>())
        .def_readwrite("base_url", &binance::HttpClientConfig::base_url)
        .def_readwrite("api_key", &binance::HttpClientConfig::api_key)
        .def_readwrite("api_secret", &binance::HttpClientConfig::api_secret)
        .def_readwrite("timeout_sec", &binance::HttpClientConfig::timeout_sec)
        .def_readwrite("recv_window", &binance::HttpClientConfig::recv_window)
        .def_readwrite("proxy", &binance::HttpClientConfig::proxy)
        .def_readwrite("tcp_keepalive", &binance::HttpClientConfig::tcp_keepalive)
        .def_readwrite("keepalive_idle_sec", &binance::HttpClientConfig::keepalive_idle_sec)
        .def_readwrite("keepalive_interval_sec", &binance::HttpClientConfig::keepalive_interval_sec);

    py::register_exception<binance::BinanceApiError>(m, "BinanceApiError");
    py::register_exception<binance::BinanceRequestError>(m, "BinanceRequestError");

    py::class_<binance::HttpClient>(m, "HttpClient")
        .def(py::init<const binance::HttpClientConfig&>(),
             py::arg("config"))
        .def("request",
             py::overload_cast<const std::string&, const std::string&,
                               const std::map<std::string, std::string>&, bool>(
                 &binance::HttpClient::request),
             py::arg("method"), py::arg("path"), py::arg("params"), py::arg("is_signed"),
             "Send HTTP request. Mirrors BinanceFuturesClient._request().\n"
             "Returns response body string on 200.\n"
             "Raises BinanceApiError on non-200.\n"
             "Raises BinanceRequestError on network error.")
        .def("request_no_params",
             py::overload_cast<const std::string&, const std::string&, bool>(
                 &binance::HttpClient::request),
             py::arg("method"), py::arg("path"), py::arg("is_signed"),
             "Send HTTP request with no params.")
        .def("is_connected", &binance::HttpClient::is_connected)
        .def("reconnect", &binance::HttpClient::reconnect);
}
