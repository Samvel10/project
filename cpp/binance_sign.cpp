#include "binance_sign.h"
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <chrono>
#include <cstring>
#include <cstdio>

namespace binance {

// URL-encode a single string value.
// Binance params are mostly alphanumeric + "." + "-" + "_",
// but we must encode any other characters per RFC 3986.
static std::string url_encode_value(const std::string& value) {
    std::string result;
    result.reserve(value.size() * 1.2);
    for (unsigned char c : value) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
            result += static_cast<char>(c);
        } else {
            char hex[4];
            std::snprintf(hex, sizeof(hex), "%%%02X", c);
            result += hex;
        }
    }
    return result;
}

// Mirrors Python: urllib.parse.urlencode(params)
// Produces "key1=value1&key2=value2&..."
// Python's urlencode uses quote_plus (spaces → +), but Binance params
// never contain spaces, so the difference is irrelevant in practice.
// We use RFC 3986 encoding (spaces → %20) which is also accepted.
std::string url_encode(const ParamList& params) {
    std::string result;
    // Pre-allocate: typical param list is ~200-400 chars
    result.reserve(512);

    bool first = true;
    for (const auto& [key, val] : params) {
        if (!first) result += '&';
        first = false;
        result += url_encode_value(key);
        result += '=';
        result += url_encode_value(val);
    }
    return result;
}

// Mirrors Python:
//   query = urlencode(params)
//   signature = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
//   return f"{query}&signature={signature}"
std::string sign_params(const unsigned char* api_secret, size_t secret_len,
                        const ParamList& params) {
    std::string query = url_encode(params);

    // HMAC-SHA256
    unsigned char hmac_result[EVP_MAX_MD_SIZE];
    unsigned int hmac_len = 0;

    HMAC(EVP_sha256(),
         api_secret, static_cast<int>(secret_len),
         reinterpret_cast<const unsigned char*>(query.data()),
         query.size(),
         hmac_result, &hmac_len);

    // Convert to hex string (Python: .hexdigest() → lowercase hex)
    static const char hex_chars[] = "0123456789abcdef";
    char hex_sig[65]; // 32 bytes → 64 hex chars + null
    for (unsigned int i = 0; i < hmac_len; ++i) {
        hex_sig[i * 2]     = hex_chars[(hmac_result[i] >> 4) & 0x0F];
        hex_sig[i * 2 + 1] = hex_chars[hmac_result[i] & 0x0F];
    }
    hex_sig[hmac_len * 2] = '\0';

    query += "&signature=";
    query.append(hex_sig, hmac_len * 2);

    return query;
}

std::string sign_params(const std::string& api_secret, const ParamList& params) {
    return sign_params(
        reinterpret_cast<const unsigned char*>(api_secret.data()),
        api_secret.size(),
        params
    );
}

int64_t timestamp_ms() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    );
    return ms.count();
}

} // namespace binance
