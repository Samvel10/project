#pragma once
// binance_http.h — Persistent HTTPS client for Binance REST API.
// 1:1 behavior match with BinanceFuturesClient._request() in binance_futures.py.
//
// Key behavioral contract from Python _request():
//   1. If signed: add timestamp (ms) + recvWindow (default 50000)
//   2. Call _sign(params) → query string with signature
//   3. Build URL: BASE_URL + path + "?" + signed_query
//   4. Send HTTP request with X-MBX-APIKEY header
//   5. On 200: return parsed JSON body
//   6. On 451/403: optionally retry with different proxy, blacklist symbol
//   7. On other errors: raise exception
//   8. On connection error: raise exception (or retry if proxy rotation)
//
// C++ differences from Python:
//   - Uses persistent TCP+TLS connection (libcurl with CURLOPT_TCP_KEEPALIVE)
//   - Reuses connection across requests (Connection: keep-alive)
//   - No per-request connection setup overhead
//   - Proxy support via CURLOPT_PROXY (same semantics)
//
// Thread safety: Each HttpClient instance is NOT thread-safe.
//   Use one instance per thread (same as Python — each account gets its own client).

#include <string>
#include <map>
#include <stdexcept>
#include <cstdint>

namespace binance {

// HTTP response from Binance.
struct HttpResponse {
    int         status_code;    // 200, 400, 403, 451, etc.
    std::string body;           // raw response body (JSON string)
    bool        ok;             // status_code == 200
};

// Matches the behavior of Python's Exception(f"Binance error {status}: {body}")
class BinanceApiError : public std::runtime_error {
public:
    int status_code;
    std::string body;

    BinanceApiError(int code, const std::string& body_text)
        : std::runtime_error("Binance error " + std::to_string(code) + ": " + body_text)
        , status_code(code)
        , body(body_text) {}
};

class BinanceRequestError : public std::runtime_error {
public:
    BinanceRequestError(const std::string& msg)
        : std::runtime_error("Binance request error: " + msg) {}
};

// Configuration for the HTTP client.
struct HttpClientConfig {
    std::string base_url = "https://fapi.binance.com";
    std::string api_key;
    std::string api_secret;     // raw secret string (will be used as bytes for HMAC)
    int         timeout_sec = 10;
    int         recv_window = 50000;    // matches Python default

    // Optional fixed proxy. Format: "http://host:port" or "http://user:pass@host:port"
    // Empty string = no proxy (direct connection).
    std::string proxy;

    // TCP keep-alive settings (prevent connection drop).
    bool tcp_keepalive = true;
    int  keepalive_idle_sec = 5;
    int  keepalive_interval_sec = 2;
};

class HttpClient {
public:
    explicit HttpClient(const HttpClientConfig& config);
    ~HttpClient();

    // Non-copyable (owns CURL handle).
    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // Move OK.
    HttpClient(HttpClient&& other) noexcept;
    HttpClient& operator=(HttpClient&& other) noexcept;

    // Mirrors Python: _request(method, path, params, signed=False)
    // For signed=true:
    //   1. Adds "timestamp" = current ms
    //   2. Adds "recvWindow" = config.recv_window (if not already in params)
    //   3. Signs params → query string with &signature=
    //   4. Sends request with X-MBX-APIKEY header
    // For signed=false:
    //   1. URL-encodes params as query string
    //   2. Sends request (no signature, no API key header needed but included anyway per Python)
    //
    // On success (200): returns response body as string.
    // On error: throws BinanceApiError (non-200) or BinanceRequestError (network).
    //
    // Python behavior preserved:
    //   - 451/403 with order endpoint: no retry (fixed proxy, attempts=1)
    //   - Other non-200: immediate throw
    //   - Connection error: immediate throw
    std::string request(const std::string& method, const std::string& path,
                        const std::map<std::string, std::string>& params,
                        bool is_signed);

    // Convenience: request with no params.
    std::string request(const std::string& method, const std::string& path,
                        bool is_signed);

    // Check if the persistent connection is alive.
    bool is_connected() const;

    // Force reconnect (e.g., after prolonged idle).
    void reconnect();

private:
    struct Impl;
    Impl* impl_;
};

} // namespace binance
