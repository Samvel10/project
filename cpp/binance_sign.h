#pragma once
// binance_sign.h — HMAC-SHA256 signing and URL encoding.
// 1:1 behavior match with BinanceFuturesClient._sign() in binance_futures.py.
// Thread-safe: no global state mutation.

#include <string>
#include <vector>
#include <utility>

namespace binance {

// A key-value pair for order parameters.
// Order matters: Binance requires alphabetical order for some endpoints,
// but urlencode preserves insertion order (Python dict preserves insertion
// order in 3.7+). We use a vector to preserve caller's order.
using ParamList = std::vector<std::pair<std::string, std::string>>;

// Mirrors Python: urlencode(params)
// Produces "key1=value1&key2=value2&..." with URL-encoding of special chars.
// Python behavior: urllib.parse.urlencode(dict)
// Note: Binance params are simple alphanumeric + dots, so minimal encoding needed.
std::string url_encode(const ParamList& params);

// Mirrors Python: BinanceFuturesClient._sign(params)
//   query = urlencode(params)
//   signature = hmac.new(api_secret, query.encode(), hashlib.sha256).hexdigest()
//   return f"{query}&signature={signature}"
//
// api_secret: raw bytes (Python: self.api_secret = api_secret.encode())
// params: key-value pairs in order
// Returns: "key=val&...&signature=<64-char hex>"
std::string sign_params(const unsigned char* api_secret, size_t secret_len,
                        const ParamList& params);

// Overload accepting std::string for api_secret.
std::string sign_params(const std::string& api_secret, const ParamList& params);

// Mirrors Python: BinanceFuturesClient._headers()
// Returns the API key header value. The header name is always "X-MBX-APIKEY".
// This is trivial but included for completeness.
// Caller uses: headers["X-MBX-APIKEY"] = api_key;
// No separate function needed — just use the api_key string directly.

// Utility: get current timestamp in milliseconds (matches Python's int(time.time() * 1000))
int64_t timestamp_ms();

} // namespace binance
