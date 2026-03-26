#include "binance_http.h"
#include "binance_sign.h"
#include <curl/curl.h>
#include <cstring>
#include <sstream>

namespace binance {

// libcurl write callback — appends data to std::string.
static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    size_t total = size * nmemb;
    buf->append(ptr, total);
    return total;
}

struct HttpClient::Impl {
    CURL*              curl = nullptr;
    struct curl_slist* default_headers = nullptr;
    HttpClientConfig   config;
    bool               initialized = false;

    Impl() = default;

    ~Impl() {
        if (default_headers) {
            curl_slist_free_all(default_headers);
            default_headers = nullptr;
        }
        if (curl) {
            curl_easy_cleanup(curl);
            curl = nullptr;
        }
    }

    void init(const HttpClientConfig& cfg) {
        config = cfg;

        curl = curl_easy_init();
        if (!curl)
            throw BinanceRequestError("Failed to initialize libcurl");

        // Persistent connection: keep-alive is default in libcurl.
        // Disable signal-based timeout handling (for thread safety).
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

        // TCP_NODELAY: disable Nagle's algorithm for lower latency.
        curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);

        // TCP keep-alive — mirrors Python behavior where connection stays open.
        if (config.tcp_keepalive) {
            curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE,
                             static_cast<long>(config.keepalive_idle_sec));
            curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL,
                             static_cast<long>(config.keepalive_interval_sec));
        }

        // Timeout — matches Python: timeout=10
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(config.timeout_sec));
        // Connection timeout (separate from transfer timeout).
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
        // Low-speed stall detection: abort if transfer rate < 1 byte/s for 25s.
        // CURLOPT_TIMEOUT alone does not catch stalled proxy POST connections;
        // this catches them so the _cpp_lock is released promptly (< 30s).
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 25L);

        // Enable connection reuse (default, but explicit).
        curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 0L);

        // Proxy — mirrors Python: proxies= fixed proxy dict.
        if (!config.proxy.empty()) {
            curl_easy_setopt(curl, CURLOPT_PROXY, config.proxy.c_str());
        }

        // Build default headers (reused across requests).
        // Python: {"X-MBX-APIKEY": self.api_key}
        std::string api_header = "X-MBX-APIKEY: " + config.api_key;
        default_headers = curl_slist_append(default_headers, api_header.c_str());

        // Write callback.
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);

        initialized = true;
    }

    // Core request method — mirrors Python _request() behavior.
    std::string do_request(const std::string& method, const std::string& path,
                           const std::map<std::string, std::string>& params,
                           bool is_signed) {
        if (!curl)
            throw BinanceRequestError("HttpClient not initialized");

        // Build ParamList from map (preserves iteration order of std::map = sorted).
        // Python dict preserves insertion order, but Binance doesn't require
        // alphabetical order — the signature covers whatever order is used.
        ParamList param_list;
        param_list.reserve(params.size() + 2);
        for (const auto& [k, v] : params) {
            param_list.emplace_back(k, v);
        }

        std::string query;

        if (is_signed) {
            // Python: params["timestamp"] = int(time.time() * 1000)
            param_list.emplace_back("timestamp", std::to_string(timestamp_ms()));

            // Python: if "recvWindow" not in params: params["recvWindow"] = 50000
            bool has_recv_window = false;
            for (const auto& [k, v] : param_list) {
                if (k == "recvWindow") { has_recv_window = true; break; }
            }
            if (!has_recv_window) {
                param_list.emplace_back("recvWindow", std::to_string(config.recv_window));
            }

            // Python: query = self._sign(params)
            query = sign_params(config.api_secret, param_list);
        } else {
            // Python: query = urlencode(params)
            query = url_encode(param_list);
        }

        // Python: url = f"{BASE_URL}{path}?{query}"
        std::string url = config.base_url + path + "?" + query;

        // Set URL
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Set method
        if (method == "GET") {
            curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        } else if (method == "POST") {
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            // POST with params in URL (no body) — Binance style.
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0L);
        } else if (method == "DELETE") {
            curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
        } else if (method == "PUT") {
            curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        }

        // Set headers
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, default_headers);

        // Response buffer
        std::string response_body;
        response_body.reserve(2048);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);

        // Execute
        CURLcode res = curl_easy_perform(curl);

        // Reset custom request method for next call
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, nullptr);

        if (res != CURLE_OK) {
            // Python: raise Exception(f"Binance request error: {e}")
            throw BinanceRequestError(curl_easy_strerror(res));
        }

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        // Python: if resp.status_code == 200: return resp.json()
        if (http_code == 200) {
            return response_body;
        }

        // Python: raise Exception(f"Binance error {resp.status_code}: {resp.text}")
        throw BinanceApiError(static_cast<int>(http_code), response_body);
    }
};

HttpClient::HttpClient(const HttpClientConfig& config)
    : impl_(new Impl())
{
    impl_->init(config);
}

HttpClient::~HttpClient() {
    delete impl_;
}

HttpClient::HttpClient(HttpClient&& other) noexcept
    : impl_(other.impl_)
{
    other.impl_ = nullptr;
}

HttpClient& HttpClient::operator=(HttpClient&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

std::string HttpClient::request(const std::string& method, const std::string& path,
                                const std::map<std::string, std::string>& params,
                                bool is_signed) {
    return impl_->do_request(method, path, params, is_signed);
}

std::string HttpClient::request(const std::string& method, const std::string& path,
                                bool is_signed) {
    std::map<std::string, std::string> empty;
    return impl_->do_request(method, path, empty, is_signed);
}

bool HttpClient::is_connected() const {
    return impl_ && impl_->curl && impl_->initialized;
}

void HttpClient::reconnect() {
    if (!impl_) return;
    HttpClientConfig cfg = impl_->config;
    delete impl_;
    impl_ = new Impl();
    impl_->init(cfg);
}

} // namespace binance
