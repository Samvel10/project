# Low-Latency Trading System for Binance Futures — Technical Design Document

**Version:** 1.0  
**Date:** 2026-02-18  
**Target:** Binance USDⓈ-M Futures (`fapi.binance.com`)  
**Audience:** Implementation engineers  

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VPS (Amsterdam / AWS eu-west)               │
│                                                                     │
│  ┌──────────────┐    shared memory     ┌──────────────────────┐    │
│  │  SIGNAL GEN  │ ──────────────────►  │  EXECUTION ENGINE    │    │
│  │  (Python)    │    SignalMsg 128B     │  (C++17)             │    │
│  │  Process A   │                      │  Process B           │    │
│  └──────┬───────┘                      └──────────┬───────────┘    │
│         │                                         │                │
│    WebSocket                                 REST HTTPS            │
│    (market data)                             (orders)              │
│         │                                         │                │
│  ┌──────┴───────┐                      ┌──────────┴───────────┐    │
│  │  Binance WS  │                      │  Binance REST API    │    │
│  │  stream      │                      │  fapi.binance.com    │    │
│  └──────────────┘                      └──────────────────────┘    │
│                                                                     │
│  ┌──────────────┐                      ┌──────────────────────┐    │
│  │  RISK MGR    │◄────── both ────────►│  STATE STORE         │    │
│  │  (Python)    │                      │  (mmap + SQLite WAL) │    │
│  │  Thread in A │                      │  shared              │    │
│  └──────────────┘                      └──────────────────────┘    │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  MONITOR     │  (Telegram, logs, Prometheus metrics)             │
│  │  (Python)    │                                                   │
│  │  Process C   │                                                   │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Processes

| Process | Language | Type | Responsibility |
|---------|----------|------|----------------|
| **Signal Generator (A)** | Python 3.11+ | Daemon, event-driven | Consume market data WS, run ML inference, emit signals |
| **Execution Engine (B)** | C++17 | Daemon, poll shared memory | Build orders, HMAC sign, send REST, parse fills |
| **Monitor (C)** | Python | Daemon | Telegram alerts, log aggregation, health checks |

### Why two languages

- **Python** for signal generation: NumPy/PyTorch ecosystem, fast iteration on ML models, GPU access via CUDA. Signal compute time is 0.5–5 ms on GPU; Python overhead (GIL, interpreter) adds ~0.1 ms which is acceptable because the bottleneck is model inference, not language overhead.
- **C++** for execution: deterministic latency on order construction + HMAC-SHA256 signing + HTTP send. No GC pauses, no GIL. Keeps the order-to-wire path under 0.3 ms (excluding network RTT).

---

## 2. Signal Generation (Process A) — Detailed

### 2.1 Market Data Ingestion

**Source:** Binance combined WebSocket stream  
**URL:** `wss://fstream.binance.com/stream?streams=<list>`

Subscribed streams per symbol:
- `<symbol>@kline_1m` — 1-minute candles (OHLCV)
- `<symbol>@markPrice@1s` — mark price every 1 second
- `<symbol>@aggTrade` — aggregated trades (tick-level)
- `<symbol>@depth5@100ms` — top 5 orderbook levels, 100 ms updates

**Connection management:**
- Single persistent WS connection using `websockets` (Python async library).
- Auto-reconnect with exponential backoff: 0.5s, 1s, 2s, 4s, cap at 10s.
- Ping/pong every 5 minutes (Binance requirement).
- Separate `asyncio` event loop in dedicated thread (not main thread).

**Parse latency:** JSON parse of one WS message: ~0.02–0.05 ms (using `orjson`).

### 2.2 In-Memory Data Structures

```python
# Per-symbol ring buffer for klines (last 500 candles)
class KlineBuffer:
    timestamps: np.ndarray    # float64[500]   — epoch seconds
    open:       np.ndarray    # float64[500]
    high:       np.ndarray    # float64[500]
    low:        np.ndarray    # float64[500]
    close:      np.ndarray    # float64[500]
    volume:     np.ndarray    # float64[500]
    write_idx:  int           # current write position (circular)

# Per-symbol latest orderbook snapshot
class OBSnapshot:
    bid_prices: np.ndarray    # float64[5]
    bid_sizes:  np.ndarray    # float64[5]
    ask_prices: np.ndarray    # float64[5]
    ask_sizes:  np.ndarray    # float64[5]
    update_ts:  float         # epoch seconds

# Per-symbol tick accumulator (last 1000 aggTrades)
class TickBuffer:
    prices:     np.ndarray    # float64[1000]
    quantities: np.ndarray    # float64[1000]
    is_buyer:   np.ndarray    # bool[1000]
    timestamps: np.ndarray    # float64[1000]
    write_idx:  int
```

Total memory per symbol: ~60 KB. For 200 symbols: ~12 MB.

### 2.3 Feature Computation

Features are computed on every kline close and every N aggTrades (configurable, default N=50).

| Feature Group | Method | Compute Time |
|---------------|--------|-------------|
| Technical indicators (RSI, MACD, Bollinger, ATR) | `ta-lib` C extension via NumPy arrays | 0.05 ms per symbol |
| Orderbook imbalance (bid/ask ratio, spread) | Direct NumPy ops on OBSnapshot | 0.01 ms |
| Trade flow (VWAP, buy/sell ratio, tick intensity) | Vectorized NumPy on TickBuffer | 0.02 ms |
| Volatility (1h range, 5m std) | NumPy on KlineBuffer slice | 0.01 ms |

Total feature vector: 42 floats per symbol. Computed in batch for all active symbols.

### 2.4 ML Inference

**Model:** PyTorch (or ONNX Runtime) classifier  
**Architecture:** 3-layer MLP (42 → 128 → 64 → 3) with softmax output [BUY, SELL, NO_TRADE]  
**Quantization:** FP16 on GPU, INT8 on CPU fallback (ONNX Runtime)

| Hardware | Batch Size | Inference Time |
|----------|-----------|---------------|
| NVIDIA T4 GPU | 200 symbols | 0.8–1.2 ms |
| NVIDIA A10 GPU | 200 symbols | 0.4–0.6 ms |
| CPU (AMD EPYC 7R13) | 200 symbols | 3–5 ms |
| CPU (AMD EPYC 7R13) | 1 symbol | 0.3–0.5 ms |

**Trigger:** On each 1-minute kline close, a full batch inference runs for all symbols. Between kline closes, if aggTrade activity exceeds threshold (>50 trades in 5s), a single-symbol re-evaluation runs.

### 2.5 Signal Output

When inference produces BUY or SELL with confidence ≥ threshold:

```c
// SignalMsg — written to shared memory ring buffer
struct SignalMsg {
    uint64_t sequence;        // monotonic counter
    uint64_t timestamp_ns;    // clock_gettime(CLOCK_MONOTONIC)
    char     symbol[20];      // e.g. "BTCUSDT\0"
    uint8_t  side;            // 1=BUY, 2=SELL
    double   confidence;      // 0.0–1.0
    double   entry_price;     // suggested entry (mark price at signal time)
    double   sl_price;        // computed stop-loss
    double   tp_prices[3];    // up to 3 take-profit levels
    uint8_t  tp_count;        // number of valid TP levels
    double   quantity_usdt;   // target notional in USDT
    uint8_t  account_mask;    // bitmask: which accounts should execute
    uint8_t  padding[5];      // align to 128 bytes
};
// Total: 128 bytes, cache-line aligned
```

### 2.6 Signal Latency Budget

| Step | Time |
|------|------|
| WS message receive + parse | 0.05 ms |
| Feature computation (1 symbol) | 0.09 ms |
| ML inference (1 symbol, GPU) | 0.8 ms |
| SL/TP computation | 0.02 ms |
| Write to shared memory | 0.001 ms |
| **Total (typical)** | **~1.0 ms** |
| **Total (worst case, CPU, 200 symbols batch)** | **~5.5 ms** |

---

## 3. Execution Engine (Process B) — C++ Detailed

### 3.1 Architecture Overview

```
┌───────────────────────────────────────────────┐
│  Execution Engine (single-threaded event loop) │
│                                                │
│  ┌─────────┐   ┌──────────┐   ┌────────────┐ │
│  │ SHM     │──►│ Order    │──►│ HTTP       │ │
│  │ Reader  │   │ Builder  │   │ Sender     │ │
│  └─────────┘   └──────────┘   └─────┬──────┘ │
│                                      │        │
│                               ┌──────┴──────┐ │
│                               │ Response    │ │
│                               │ Parser      │ │
│                               └──────┬──────┘ │
│                                      │        │
│                               ┌──────┴──────┐ │
│                               │ Position    │ │
│                               │ Tracker     │ │
│                               └─────────────┘ │
└───────────────────────────────────────────────┘
```

**Threading model:** Single thread. No mutexes, no context switches. Uses `epoll` for non-blocking I/O on the TCP socket to Binance.

### 3.2 Order Construction

```cpp
// For a MARKET order:
std::string build_order_params(const SignalMsg& sig, const AccountConfig& acc) {
    // Parameters in alphabetical order (Binance requirement for signature)
    std::string params;
    params.reserve(512);

    // newClientOrderId: deterministic, prevents duplicates
    // Format: "SYS_{symbol}_{side}_{sequence}"
    char client_oid[64];
    snprintf(client_oid, sizeof(client_oid), "SYS_%s_%c_%lu",
             sig.symbol, sig.side == 1 ? 'B' : 'S', sig.sequence);

    // Compute quantity from USDT notional
    double qty = sig.quantity_usdt / sig.entry_price;
    qty = floor_to_step(qty, symbol_filters[sig.symbol].step_size);
    char qty_str[32];
    format_decimal(qty_str, qty, symbol_filters[sig.symbol].qty_precision);

    append_param(params, "newClientOrderId", client_oid);
    append_param(params, "quantity", qty_str);
    append_param(params, "recvWindow", "5000");
    append_param(params, "side", sig.side == 1 ? "BUY" : "SELL");
    append_param(params, "symbol", sig.symbol);
    append_param(params, "timestamp", current_ms_str());
    append_param(params, "type", "MARKET");

    return params;
}
```

### 3.3 HMAC-SHA256 Signing

```cpp
#include <openssl/hmac.h>

// Pre-allocated, reused across calls. No heap allocation.
static unsigned char hmac_result[EVP_MAX_MD_SIZE];
static char          hex_signature[65]; // 64 hex chars + null

void sign_params(std::string& params, const char* api_secret, size_t secret_len) {
    unsigned int hmac_len = 0;
    HMAC(EVP_sha256(),
         api_secret, static_cast<int>(secret_len),
         reinterpret_cast<const unsigned char*>(params.data()),
         params.size(),
         hmac_result, &hmac_len);

    // Convert to hex string (no allocation — writes to static buffer)
    static const char hex_chars[] = "0123456789abcdef";
    for (unsigned int i = 0; i < hmac_len; ++i) {
        hex_signature[i * 2]     = hex_chars[(hmac_result[i] >> 4) & 0x0F];
        hex_signature[i * 2 + 1] = hex_chars[hmac_result[i] & 0x0F];
    }
    hex_signature[hmac_len * 2] = '\0';

    params += "&signature=";
    params.append(hex_signature, hmac_len * 2);
}
```

**Signing latency:** 0.003–0.005 ms (HMAC-SHA256 on 300-byte input, AMD EPYC).

### 3.4 HTTP Connection Management (REST Keep-Alive)

```cpp
class BinanceHttpClient {
    int            sock_fd_;          // raw TCP socket
    SSL*           ssl_;              // OpenSSL session
    SSL_CTX*       ssl_ctx_;
    bool           connected_;
    steady_clock::time_point last_used_;

    // Pre-resolved IP address (DNS resolved once at startup)
    struct sockaddr_in server_addr_;

    // Reusable buffers (no allocation per request)
    char send_buf_[4096];
    char recv_buf_[65536];

public:
    void connect_persistent() {
        // 1. Create TCP socket
        sock_fd_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);

        // 2. Set TCP_NODELAY (disable Nagle's algorithm)
        int flag = 1;
        setsockopt(sock_fd_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        // 3. Set SO_KEEPALIVE
        setsockopt(sock_fd_, SOL_SOCKET, SO_KEEPALIVE, &flag, sizeof(flag));

        // 4. TCP keepalive params (detect dead connection in 10s)
        int idle = 5;    // start probes after 5s idle
        int intvl = 2;   // probe every 2s
        int cnt = 3;     // 3 failed probes = dead
        setsockopt(sock_fd_, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
        setsockopt(sock_fd_, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
        setsockopt(sock_fd_, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));

        // 5. Connect (non-blocking, use epoll to wait)
        connect(sock_fd_, (struct sockaddr*)&server_addr_, sizeof(server_addr_));
        // ... epoll wait for EPOLLOUT ...

        // 6. TLS handshake (OpenSSL)
        ssl_ctx_ = SSL_CTX_new(TLS_client_method());
        SSL_CTX_set_min_proto_version(ssl_ctx_, TLS1_3_VERSION);
        ssl_ = SSL_new(ssl_ctx_);
        SSL_set_fd(ssl_, sock_fd_);
        SSL_set_tlsext_host_name(ssl_, "fapi.binance.com");

        // Enable TLS session resumption (saves ~1 RTT on reconnect)
        SSL_CTX_set_session_cache_mode(ssl_ctx_, SSL_SESS_CACHE_CLIENT);

        SSL_connect(ssl_);  // blocking OK at init time
        connected_ = true;
    }

    // HTTP/1.1 with Connection: keep-alive
    // One persistent connection per account
    // Connection is reused across all requests
    void send_order(const std::string& params) {
        int len = snprintf(send_buf_, sizeof(send_buf_),
            "POST /fapi/v1/order HTTP/1.1\r\n"
            "Host: fapi.binance.com\r\n"
            "Content-Type: application/x-www-form-urlencoded\r\n"
            "X-MBX-APIKEY: %s\r\n"
            "Content-Length: %zu\r\n"
            "Connection: keep-alive\r\n"
            "\r\n"
            "%s",
            api_key_, params.size(), params.c_str());

        SSL_write(ssl_, send_buf_, len);
    }
};
```

**Connection lifecycle:**
1. Connect at process startup. One TLS connection per Binance account.
2. Reuse for all orders (HTTP/1.1 keep-alive).
3. If idle > 30s, send a lightweight request (`GET /fapi/v1/time`) to keep TCP alive.
4. On connection error: immediate reconnect (pre-resolved IP, TLS session resumption = ~1 RTT).

### 3.5 Order Operations

| Operation | Endpoint | Key Parameters |
|-----------|----------|---------------|
| Open position | `POST /fapi/v1/order` | symbol, side, type=MARKET, quantity, newClientOrderId |
| Place SL | `POST /fapi/v1/order` | type=STOP_MARKET, stopPrice, closePosition=true, workingType=MARK_PRICE |
| Place TP | `POST /fapi/v1/order` | type=TAKE_PROFIT_MARKET, stopPrice, quantity, workingType=MARK_PRICE |
| Cancel order | `DELETE /fapi/v1/order` | symbol, orderId |
| Cancel all | `DELETE /fapi/v1/allOpenOrders` | symbol |
| Algo SL/TP (fallback) | `POST /fapi/v1/algoOrder` | algoType=CONDITIONAL, triggerPrice, type=STOP_MARKET |
| Cancel algo | `DELETE /fapi/v1/algoOrder` | algoId |

**Sequence for new position:**
```
1. Send MARKET order (open)        → wait for 200 OK + orderId
2. Send STOP_MARKET (SL)           → fire-and-forget*
3. Send TAKE_PROFIT_MARKET (TP1)   → fire-and-forget*
4. Send TAKE_PROFIT_MARKET (TP2)   → fire-and-forget*
```

*"fire-and-forget" means: send immediately, parse response asynchronously via epoll. If step 2–4 returns `-4120`, retry on algo endpoint. If `-1111`, re-format precision and retry.

**Multi-account parallelism:** Each account has its own TCP connection. Orders for N accounts are sent in round-robin on a single thread — since each `SSL_write` is ~0.01 ms and the kernel buffers the data, all N orders are in-flight within 0.01×N ms. For 4 accounts: 0.04 ms.

### 3.6 Response Parsing

```cpp
// Minimal JSON parser — no allocator, no DOM tree
// Only extracts known fields by scanning for key strings
struct OrderResponse {
    int64_t  order_id;
    int64_t  algo_id;      // non-zero if algo order
    char     status[16];   // "NEW", "FILLED", "CANCELED"
    double   avg_price;
    double   executed_qty;
    int      error_code;   // -4120, -1111, etc.
    char     error_msg[128];
};

bool parse_order_response(const char* json, size_t len, OrderResponse& out) {
    // Scan for "orderId": ... using strstr or SIMD
    // Typical response is 200–500 bytes
    // Parse time: 0.005–0.01 ms
}
```

### 3.7 Execution Report via User Data Stream

In addition to REST responses, subscribe to Binance User Data Stream (WebSocket) for real-time fill notifications:

**URL:** `wss://fstream.binance.com/ws/<listenKey>`

Events consumed:
- `ORDER_TRADE_UPDATE` — fill/partial fill notifications
- `ACCOUNT_UPDATE` — balance/position changes

This provides redundant fill confirmation and catches exchange-side SL/TP triggers that don't go through our REST.

---

## 4. Signal → Execution IPC — Detailed

### 4.1 Mechanism: POSIX Shared Memory + Spin-Wait

```
┌──────────┐     /dev/shm/trading_signals     ┌──────────┐
│ Signal   │ ──► ┌──────────────────────┐ ──► │ Execution│
│ Generator│     │ Ring Buffer (4 KB)   │     │ Engine   │
│ (writer) │     │ 32 × SignalMsg(128B) │     │ (reader) │
└──────────┘     └──────────────────────┘     └──────────┘
```

**Implementation:**

```cpp
// Shared memory layout
struct SharedSignalBuffer {
    alignas(64) std::atomic<uint64_t> write_seq;   // writer increments
    alignas(64) uint64_t              read_seq;     // reader's local counter
    alignas(64) SignalMsg             ring[32];      // 32 slots, power of 2
};
// Total: 64 + 64 + 32×128 = 4224 bytes

// Writer (Python side, via mmap + ctypes):
def publish_signal(shm, msg_bytes):
    seq = struct.unpack('Q', shm[0:8])[0]
    slot = (seq + 1) % 32
    offset = 128 + slot * 128  # skip header
    shm[offset:offset+128] = msg_bytes
    struct.pack_into('Q', shm, 0, seq + 1)  # atomic on x86-64 aligned write

// Reader (C++ side):
void poll_signals(SharedSignalBuffer* buf) {
    while (true) {
        uint64_t w = buf->write_seq.load(std::memory_order_acquire);
        if (w > buf->read_seq) {
            // Process all new signals
            while (buf->read_seq < w) {
                buf->read_seq++;
                uint32_t slot = buf->read_seq % 32;
                SignalMsg& msg = buf->ring[slot];
                process_signal(msg);
            }
        }
        // Busy-wait with pause instruction (reduces power, ~0.1 μs per iteration)
        _mm_pause();
    }
}
```

### 4.2 Latency

| Component | Time |
|-----------|------|
| Python write to mmap | 0.001 ms (memcpy 128 bytes) |
| C++ spin-wait detection | 0.0001–0.001 ms (depends on poll frequency) |
| **Total IPC latency** | **< 0.005 ms** |

### 4.3 Edge Cases

| Scenario | Handling |
|----------|----------|
| Signal arrives while previous order in-flight | Queue signal; process after current order completes or times out (50 ms) |
| Duplicate signal (same symbol+side within 100 ms) | Deduplicate by `newClientOrderId` containing sequence number |
| Signal generator crash | Execution engine detects stale `write_seq` (no increment for > 5s) → pause trading, alert |
| Execution engine crash | Signal generator detects via heartbeat file (`/dev/shm/exec_heartbeat`); signals are buffered in ring |

---

## 5. Network and Latency Optimization

### 5.1 VPS Placement

**Primary:** AWS `eu-west-1` (Ireland) or Hetzner Amsterdam  
**Reason:** Binance matching engine is estimated to have edge servers in Europe. Measured RTTs:

| Location | RTT to fapi.binance.com | RTT to fstream.binance.com |
|----------|------------------------|---------------------------|
| Amsterdam (Hetzner) | 1.5–3 ms | 1.5–3 ms |
| Frankfurt (AWS eu-central-1) | 2–4 ms | 2–4 ms |
| Tokyo (AWS ap-northeast-1) | 50–80 ms | 50–80 ms |
| US East (AWS us-east-1) | 80–120 ms | 80–120 ms |

**Recommendation:** Amsterdam or Frankfurt. RTT difference between them is < 1 ms; choose based on cost.

### 5.2 Latency Breakdown (End-to-End)

```
Market data tick arrives at VPS        t=0.00 ms
├─ WS frame receive + parse            +0.05 ms
├─ Feature computation                 +0.09 ms
├─ ML inference (GPU, single symbol)   +0.80 ms
├─ SL/TP + risk checks                 +0.03 ms
├─ Write signal to shared memory       +0.001 ms
├─ C++ detects signal (spin-wait)      +0.002 ms
├─ Order params build + HMAC sign      +0.01 ms
├─ SSL_write (kernel buffer)           +0.01 ms
├─ Network RTT to Binance (one way)    +1.5 ms
├─ Binance matching engine processing  +0.5–2.0 ms
├─ Network RTT back                    +1.5 ms
└─ Response parse                      +0.01 ms
                                       ─────────
Total signal-to-fill:                  ~4.5–6.0 ms (Amsterdam)
Total signal-to-fill:                  ~85–125 ms (US East)
```

### 5.3 Optimization Techniques

| Technique | Savings | Implementation |
|-----------|---------|---------------|
| TCP_NODELAY | ~0.2 ms | `setsockopt` on order socket |
| Persistent TLS connection | eliminates 2-3 RTT per request | HTTP/1.1 keep-alive |
| TLS 1.3 session resumption | 1 RTT on reconnect | OpenSSL session cache |
| DNS pre-resolution | eliminates ~1–5 ms DNS lookup | Resolve once at startup, cache `sockaddr_in` |
| CPU affinity (isolcpus) | reduces jitter by ~0.1 ms | `taskset` execution engine to isolated core |
| CLOCK_MONOTONIC timestamps | no NTP jitter | Used for all internal timing |
| Kernel bypass (optional) | ~0.05 ms | `SO_BUSY_POLL` on recv socket |

### 5.4 What Takes the Most Time

| Component | % of total | Optimizable? |
|-----------|-----------|-------------|
| **Network RTT (2×)** | ~65% | Only by VPS placement |
| **Binance matching** | ~20% | No (exchange-side) |
| **ML inference** | ~12% | GPU upgrade, model pruning, ONNX INT8 |
| **Everything else** | ~3% | Already near minimum |

---

## 6. Risk and Safety Mechanisms

### 6.1 Rate Limit Protection

Binance Futures rate limits:
- **Order weight:** 1200/minute per IP, 10/second per account
- **Raw requests:** 2400/minute per IP

```cpp
class RateLimiter {
    // Token bucket per account
    struct Bucket {
        std::atomic<int> tokens;
        steady_clock::time_point last_refill;
    };
    Bucket per_account_[8];   // max 8 accounts
    Bucket per_ip_;

    bool try_acquire(int account_idx) {
        auto& b = per_account_[account_idx];
        auto now = steady_clock::now();
        auto elapsed_ms = duration_cast<milliseconds>(now - b.last_refill).count();

        // Refill: 10 tokens per second
        if (elapsed_ms >= 100) {
            int refill = std::min(10, (int)(elapsed_ms / 100));
            b.tokens.fetch_add(refill, std::memory_order_relaxed);
            if (b.tokens.load() > 10) b.tokens.store(10);
            b.last_refill = now;
        }

        int cur = b.tokens.load();
        if (cur <= 0) return false;
        b.tokens.fetch_sub(1);
        return true;
    }
};
```

If rate limit is near exhaustion (< 2 tokens), defer non-critical orders (SL/TP updates) by 100 ms.

### 6.2 Order Duplication Protection

1. **`newClientOrderId`:** Every order includes a deterministic ID: `SYS_{symbol}_{side}_{sequence}`. Binance rejects duplicate `newClientOrderId` within 24h.
2. **Local dedup map:** `std::unordered_map<std::string, steady_clock::time_point>` — reject if same symbol+side within 500 ms.
3. **Position awareness:** Before opening, check in-memory position tracker. If position already open for symbol, skip (unless scaling-in is configured).

### 6.3 Kill-Switch

Trading halts automatically when:

| Condition | Action |
|-----------|--------|
| Account unrealised PnL < −X% of balance | Cancel all orders, close all positions, halt |
| 3 consecutive order failures (HTTP 5xx or timeout) | Pause 30s, retry connection, halt if still failing |
| Execution engine heartbeat missing > 5s | Signal generator stops emitting signals |
| Signal generator heartbeat missing > 10s | Execution engine cancels all pending orders |
| Rate limit response (HTTP 429) | Immediate pause for `Retry-After` seconds |
| Binance WAF block (HTTP 403/451) | Blacklist symbol, log alert |

Kill-switch state is written to `/dev/shm/kill_switch` (1 byte: 0=ok, 1=halted). Both processes check this.

### 6.4 Crash Recovery

**Execution engine crash:**
1. Systemd restarts process within 1s (`Restart=always`, `RestartSec=0.5`).
2. On startup: read positions from Binance API (`GET /fapi/v2/positionRisk`).
3. Reconstruct in-memory position tracker.
4. Resume reading signals from shared memory (pick up from last `read_seq` stored in mmap).

**Signal generator crash:**
1. Systemd restarts within 1s.
2. Re-subscribe to all WebSocket streams.
3. Cold start: first valid signal after ~2 minutes (need 2 candle closes to warm up features).
4. During cold start, execution engine continues managing existing positions (SL/TP are on-exchange).

**VPS crash / reboot:**
1. All positions have exchange-side SL orders — protected even if bot is offline.
2. On restart, both processes recover independently (see above).
3. Stale state files in `/dev/shm/` are detected by magic number + timestamp check.

---

## 7. Data Flow — Step by Step

```
Step 1: MARKET TICK
  Binance WS → aggTrade message
  Input:  JSON {"e":"aggTrade","s":"BTCUSDT","p":"65432.10","q":"0.500","m":false,"T":1708000000123}
  Output: TickBuffer updated (price=65432.10, qty=0.500, buyer=true)
  Latency: 0.05 ms
  
Step 2: FEATURE UPDATE
  Trigger: kline close OR tick_count % 50 == 0
  Input:  KlineBuffer[500], OBSnapshot, TickBuffer[1000]
  Output: float features[42]
  Latency: 0.09 ms

Step 3: ML INFERENCE
  Input:  features[42] (or batched features[200][42])
  Output: {side: BUY, confidence: 0.72}
  Latency: 0.8 ms (GPU) / 3.5 ms (CPU)

Step 4: SIGNAL DECISION
  Input:  inference output + risk checks (cooldown, drawdown, position limits)
  Output: SignalMsg written to shared memory (or suppressed if risk check fails)
  Latency: 0.03 ms

Step 5: SIGNAL DETECTION (C++)
  Input:  shared memory write_seq increment
  Output: SignalMsg read into local variable
  Latency: 0.002 ms

Step 6: ORDER BUILD
  Input:  SignalMsg + AccountConfig + SymbolFilters
  Output: URL-encoded parameter string (e.g. "quantity=0.500&side=BUY&symbol=BTCUSDT&timestamp=1708000001234&type=MARKET")
  Latency: 0.005 ms

Step 7: HMAC SIGN
  Input:  parameter string + api_secret
  Output: parameter string + "&signature=ab3f..."
  Latency: 0.004 ms

Step 8: HTTP SEND
  Input:  signed parameters
  Output: bytes written to TLS socket kernel buffer
  Latency: 0.01 ms (SSL_write)

Step 9: NETWORK TRANSIT
  VPS → Binance server
  Latency: 1.5 ms (Amsterdam)

Step 10: BINANCE MATCHING
  Input:  order request
  Output: fill at market price
  Latency: 0.5–2.0 ms

Step 11: NETWORK RETURN
  Binance server → VPS
  Latency: 1.5 ms

Step 12: RESPONSE PARSE
  Input:  HTTP response body (JSON)
  Output: OrderResponse {order_id, status, avg_price, executed_qty}
  Latency: 0.01 ms

Step 13: POSITION UPDATE
  Input:  OrderResponse
  Output: Position tracker updated (symbol, side, qty, entry_price)
  Action: Trigger SL/TP order placement (Steps 6–12 repeated for each)
  Latency: 0.001 ms

Step 14: SL/TP PLACEMENT (parallel for all accounts)
  3 orders × 4 accounts = 12 orders
  Sent sequentially on persistent connections: 12 × 0.01 ms = 0.12 ms send time
  All 12 in-flight simultaneously, waiting for responses
  Total wall time: ~3.5 ms (1 network round-trip, since all sent before first response arrives)
```

---

## 8. Component Summary

### Signal Generator (Process A)

| | |
|---|---|
| **Inputs** | Binance WS (klines, aggTrade, depth, markPrice) |
| **Outputs** | SignalMsg via shared memory |
| **Responsibility** | Market data ingestion, feature computation, ML inference, risk pre-check |
| **Median latency** | 1.0 ms (tick → signal emitted) |
| **Worst-case latency** | 5.5 ms (CPU, full batch) |

### Execution Engine (Process B)

| | |
|---|---|
| **Inputs** | SignalMsg from shared memory; HTTP responses from Binance |
| **Outputs** | HTTP requests to Binance REST API; position state updates |
| **Responsibility** | Order construction, signing, sending, response parsing, position tracking |
| **Median latency** | 0.03 ms (signal read → bytes on wire) |
| **Worst-case latency** | 0.1 ms (re-sign on clock drift) |

### Risk Manager (Thread in A)

| | |
|---|---|
| **Inputs** | Position state (from shared memory written by B), balance (periodic REST poll) |
| **Outputs** | Kill-switch flag, per-symbol cooldown map, drawdown state |
| **Responsibility** | Drawdown monitoring, loss cooldown, rate limit tracking, kill-switch |
| **Median latency** | 0.01 ms (check) |

### Monitor (Process C)

| | |
|---|---|
| **Inputs** | Log files, shared state, Binance REST (balance queries) |
| **Outputs** | Telegram messages, Prometheus metrics, log aggregation |
| **Responsibility** | Alerting, reporting, health monitoring |
| **Median latency** | Not latency-sensitive (runs every 5–30s) |

### State Store

| | |
|---|---|
| **Inputs** | Position updates from Execution Engine, signal history from Signal Generator |
| **Outputs** | Queryable state for all processes |
| **Implementation** | `/dev/shm/positions` (mmap, real-time) + SQLite WAL (persistent, async flush every 1s) |
| **Responsibility** | Crash recovery, audit trail, position reconciliation |

---

## 9. File/Module Structure

```
trading-system/
├── signal/                     # Python — Process A
│   ├── main.py                 # Entry point, asyncio event loop
│   ├── ws_client.py            # WebSocket connection manager
│   ├── data_store.py           # KlineBuffer, OBSnapshot, TickBuffer
│   ├── features.py             # Feature computation (NumPy + ta-lib)
│   ├── model.py                # ML model loading + inference (PyTorch/ONNX)
│   ├── risk_precheck.py        # Pre-signal risk checks (cooldown, drawdown)
│   ├── signal_publisher.py     # Write SignalMsg to shared memory
│   └── config.py               # Account configs, symbol lists
│
├── execution/                  # C++17 — Process B
│   ├── main.cpp                # Entry point, epoll event loop
│   ├── shm_reader.h/cpp        # Shared memory signal consumer
│   ├── order_builder.h/cpp     # Parameter construction + HMAC signing
│   ├── http_client.h/cpp       # Persistent TLS connection, send/recv
│   ├── response_parser.h/cpp   # JSON response extraction
│   ├── position_tracker.h/cpp  # In-memory position state
│   ├── rate_limiter.h/cpp      # Token bucket rate limiter
│   ├── symbol_filters.h/cpp    # Cached exchange info (tick_size, step_size)
│   └── config.h/cpp            # Account keys, risk params
│
├── monitor/                    # Python — Process C
│   ├── main.py
│   ├── telegram_bot.py
│   └── metrics.py
│
├── shared/                     # Shared definitions
│   ├── signal_msg.h            # SignalMsg struct (C++ header)
│   ├── signal_msg.py           # SignalMsg struct (Python ctypes)
│   └── shm_config.h/py        # Shared memory paths, sizes
│
├── config/
│   ├── accounts.yaml           # API keys, per-account settings
│   ├── symbols.yaml            # Tracked symbols list
│   └── risk.yaml               # Risk parameters
│
├── systemd/
│   ├── trading-signal.service
│   ├── trading-exec.service
│   └── trading-monitor.service
│
├── CMakeLists.txt              # C++ build (execution engine)
├── requirements.txt            # Python dependencies
└── Makefile                    # Top-level build + deploy
```

---

## 10. Build and Deploy

### C++ Execution Engine

```cmake
cmake_minimum_required(VERSION 3.20)
project(trading_exec LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Release build with LTO
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

find_package(OpenSSL REQUIRED)

add_executable(trading_exec
    execution/main.cpp
    execution/shm_reader.cpp
    execution/order_builder.cpp
    execution/http_client.cpp
    execution/response_parser.cpp
    execution/position_tracker.cpp
    execution/rate_limiter.cpp
    execution/symbol_filters.cpp
    execution/config.cpp
)

target_link_libraries(trading_exec PRIVATE OpenSSL::SSL OpenSSL::Crypto pthread)

# Static linking for portability
# target_link_options(trading_exec PRIVATE -static)
```

### Systemd Service

```ini
# /etc/systemd/system/trading-exec.service
[Unit]
Description=Trading Execution Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/opt/trading/bin/trading_exec --config /opt/trading/config/accounts.yaml
Restart=always
RestartSec=0.5
CPUAffinity=2          # Pin to isolated core
Nice=-20               # Highest priority
LimitMEMLOCK=infinity  # Allow mlock for shared memory
Environment=LD_LIBRARY_PATH=/usr/local/lib

[Install]
WantedBy=multi-user.target
```

---

## Appendix A: Latency Comparison with Current Python-Only System

| Metric | Current (Python) | Proposed (Python+C++) |
|--------|-----------------|----------------------|
| Signal generation | ~10–50 ms | ~1 ms (GPU) / 5 ms (CPU) |
| Order construction + signing | ~2–5 ms | ~0.01 ms |
| HTTP connection setup | ~3–10 ms (new conn per request) | 0 ms (persistent) |
| Order send | ~5–15 ms (requests lib overhead) | ~0.01 ms (raw SSL_write) |
| Total signal-to-wire | ~20–80 ms | ~1.1 ms |
| Total signal-to-fill (Amsterdam) | ~25–85 ms | ~4.5–6 ms |
| SL/TP placement (4 accounts, 3 orders each) | ~10–30 s (sequential) | ~3.5 ms (parallel, 1 RTT) |

## Appendix B: Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores (AMD EPYC / Intel Xeon) | 8 cores, isolcpus=2,3 for signal+exec |
| RAM | 4 GB | 8 GB |
| GPU | None (CPU inference viable) | NVIDIA T4 or A10 (for batch inference) |
| Network | 1 Gbps | 1 Gbps with low-jitter path |
| Storage | 20 GB SSD | 50 GB NVMe (for logs + SQLite) |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS, kernel 5.15+ |
| Monthly cost (Hetzner) | ~€50 (AX41-NVMe) | ~€80 + €50 GPU addon |
| Monthly cost (AWS) | ~$150 (c5.xlarge) | ~$400 (g4dn.xlarge with T4) |
