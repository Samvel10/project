# STEP A — Execution Path Analysis

**All functions listed below are in Python.**  
**File:** `/var/www/html/new_example_bot/execution/binance_futures.py`

---

## 1. Functions by Category

### a) ORDER OPEN (new position entry)

| # | Function | Line | Scope | Description |
|---|----------|------|-------|-------------|
| 1 | `BinanceFuturesClient.place_order()` | 692 | Instance method | Build params dict, call `_request("POST", "/fapi/v1/order", ...)` |
| 2 | `place_order()` (module-level) | 3724 | Public API | Multi-account orchestrator: iterates accounts, applies gating checks, calls `client.place_order()` per account. **NOT a candidate for C++ — contains business logic, gating, multi-account routing.** |

### b) ORDER CLOSE (position exit)

| # | Function | Line | Scope | Description |
|---|----------|------|-------|-------------|
| 3 | `close_position_market()` | 5088 | Public API | Adjusts qty, determines side, calls `client.place_order(reduce_only=True)` |
| 4 | `place_exchange_stop_order()` | 4688 | Public API | Places STOP_MARKET/TAKE_PROFIT_MARKET. Tries regular endpoint, falls back to algo on -4120. Calls `client.place_order()` or `client.place_algo_order()` |
| 5 | `BinanceFuturesClient.place_algo_order()` | 737 | Instance method | Build params dict for algo conditional order, call `_request("POST", "/fapi/v1/algoOrder", ...)` |

### c) CANCEL / REPLACE

| # | Function | Line | Scope | Description |
|---|----------|------|-------|-------------|
| 6 | `BinanceFuturesClient.cancel_order()` | 656 | Instance method | `_request("DELETE", "/fapi/v1/order", ...)` |
| 7 | `BinanceFuturesClient.cancel_algo_order()` | 784 | Instance method | `_request("DELETE", "/fapi/v1/algoOrder", ...)` |
| 8 | `cancel_order_by_id()` | 4805 | Public API | Dispatches to `client.cancel_order()` or `client.cancel_algo_order()` based on `is_algo` flag |
| 9 | `cancel_symbol_open_orders()` | 4826 | Public API | Queries `client.open_orders()` + `client.get_algo_open_orders()`, cancels each |
| 10 | `BinanceFuturesClient.open_orders()` | 650 | Instance method | `_request("GET", "/fapi/v1/openOrders", ...)` |
| 11 | `BinanceFuturesClient.get_algo_open_orders()` | 789 | Instance method | `_request("GET", "/fapi/v1/openAlgoOrders", ...)` |

### d) SIGNING

| # | Function | Line | Scope | Description |
|---|----------|------|-------|-------------|
| 12 | `BinanceFuturesClient._sign()` | 543 | Instance method | `urlencode(params)` → HMAC-SHA256 with `api_secret` → append `&signature=` |
| 13 | `BinanceFuturesClient._headers()` | 548 | Instance method | Returns `{"X-MBX-APIKEY": self.api_key}` |

### e) NETWORK SEND

| # | Function | Line | Scope | Description |
|---|----------|------|-------|-------------|
| 14 | `BinanceFuturesClient._request()` | 551 | Instance method | Adds timestamp+recvWindow, calls `_sign()`, builds URL, calls `requests.request()`, handles retries on 451/403, parses JSON response |

### f) PRICE/QUANTITY ADJUSTMENT (pre-order helpers)

| # | Function | Line | Scope | Description |
|---|----------|------|-------|-------------|
| 15 | `adjust_price()` | 3419 | Public | `math.floor(price / tick_size) * tick_size` |
| 16 | `adjust_quantity()` | 3451 | Public | Floor by step_size, enforce min_qty, enforce min_notional |
| 17 | `adjust_close_quantity()` | 3626 | Public | Floor/ceil by step_size, enforce min_qty (no min_notional) |
| 18 | `adjust_quantity_up()` | 3527 | Public | Ceil by step_size, enforce min_qty (no min_notional) |
| 19 | `_tick_decimals()` | 4678 | Private | Count decimal places from tick/step float |
| 20 | `_get_symbol_filters()` | 3290 | Private | Return cached filters dict; triggers `_ensure_exchange_info()` on first call |

---

## 2. Call Graph

```
                    ┌──────────────────────────────────────────┐
                    │  CALLERS (main.py exit manager, signals)  │
                    └────────┬──────────┬──────────┬───────────┘
                             │          │          │
                    ┌────────▼──┐  ┌────▼──────┐  ┌▼──────────────────┐
                    │ place_     │  │ close_    │  │ place_exchange_   │
                    │ order()   │  │ position_ │  │ stop_order()      │
                    │ [module]  │  │ market()  │  │                   │
                    │ #3724     │  │ #5088     │  │ #4688             │
                    └──┬────┬──┘  └─────┬─────┘  └──┬──────────┬────┘
                       │    │           │            │          │
              gating   │    │    ┌──────▼──────┐     │   ┌──────▼──────┐
              checks   │    │    │adjust_close │     │   │adjust_price │
                       │    │    │_quantity()  │     │   │() #3419     │
                       │    │    │#3626        │     │   └──────┬──────┘
                       │    │    └──────┬──────┘     │          │
                       │    │           │            │   ┌──────▼──────┐
                       │    │           │            │   │adjust_close │
                       │    │           │            │   │_quantity()  │
                       │    │           │            │   │#3626        │
                       │    │           │            │   └──────┬──────┘
                       │    │           │            │          │
                       ▼    ▼           ▼            ▼          ▼
                    ┌───────────────────────────────────────────────┐
                    │         BinanceFuturesClient (instance)       │
                    │                                               │
                    │  .place_order()          #692                 │
                    │  .place_algo_order()     #737                 │
                    │  .cancel_order()         #656                 │
                    │  .cancel_algo_order()    #784                 │
                    │  .open_orders()          #650                 │
                    │  .get_algo_open_orders() #789                 │
                    │         │                                     │
                    │         ▼                                     │
                    │  ._request(method, path, params, signed)     │
                    │  #551                                         │
                    │         │                                     │
                    │    ┌────┴────┐                                │
                    │    ▼         ▼                                │
                    │  ._sign()  ._headers()                       │
                    │  #543      #548                               │
                    │    │                                          │
                    │    ▼                                          │
                    │  urlencode → HMAC-SHA256 → signature          │
                    │                                               │
                    │  ._request() internals:                       │
                    │    1. Add timestamp + recvWindow               │
                    │    2. _sign(params) → query string            │
                    │    3. Build URL: BASE_URL + path + "?" + qs   │
                    │    4. Resolve proxies                          │
                    │    5. requests.request(method, url, ...)       │
                    │    6. Handle 200 → resp.json()                │
                    │    7. Handle 451/403 → retry or blacklist     │
                    │    8. Handle other → raise Exception          │
                    │                                               │
                    └───────────────────────────────────────────────┘

    cancel_order_by_id() #4805
        ├── client.cancel_order()      [is_algo=False]
        └── client.cancel_algo_order() [is_algo=True]

    cancel_symbol_open_orders() #4826
        ├── client.open_orders(symbol) → list
        │   └── for each: client.cancel_order(symbol, orderId)
        └── client.get_algo_open_orders(symbol) → list
            └── for each: client.cancel_algo_order(algoId)

    _get_symbol_filters(symbol) #3290
        └── _ensure_exchange_info() #3225  [one-time fetch]
            └── client.exchange_info()
                └── _request("GET", "/fapi/v1/exchangeInfo")
```

---

## 3. C++ Migration Candidates

The **latency-critical path** that benefits from C++ is the inner core:

| Layer | Functions | Current Latency | C++ Target | Migrate? |
|-------|-----------|----------------|------------|----------|
| **Signing** | `_sign()`, `_headers()` | ~0.5–2 ms (Python hmac + urlencode) | ~0.005 ms (OpenSSL HMAC) | **YES** |
| **Param build** | Parameter dict construction in `place_order()`, `place_algo_order()`, `cancel_order()`, etc. | ~0.1–0.3 ms | ~0.005 ms | **YES** |
| **URL encode** | `urlencode(params)` inside `_sign()` | ~0.1 ms | ~0.005 ms | **YES** |
| **Network send** | `requests.request()` inside `_request()` | ~3–15 ms (Python requests overhead + connection setup) | ~0.01 ms (persistent TLS + raw SSL_write) | **YES** |
| **Response parse** | `resp.json()` | ~0.1–0.5 ms | ~0.01 ms (minimal JSON) | **YES** |
| **Price/qty adjust** | `adjust_price()`, `adjust_quantity()`, `adjust_close_quantity()`, `_tick_decimals()` | ~0.01 ms each | ~0.001 ms each | **YES** (pure math, no side effects) |

**NOT migrating to C++** (business logic, must stay Python):
- `place_order()` module-level (#3724) — multi-account routing, gating checks, room sizing, fibo staking
- `place_exchange_stop_order()` (#4688) — algo fallback logic, logging, account state tracking
- `cancel_order_by_id()` (#4805) — dispatch wrapper
- `cancel_symbol_open_orders()` (#4826) — orchestration loop
- `close_position_market()` (#5088) — qty adjustment + side logic + logging
- All gating checks (`_is_delay_limit_ok`, `_is_symbol_loss_cooldown_ok`, etc.)

These wrappers will call into C++ for the hot inner operations but retain their existing Python logic unchanged.

---

## 4. Exact Functions to Port to C++

### Group 1: Core Client Operations (BinanceFuturesClient inner methods)

```
C++ Module: binance_fast_client

Functions:
  1. sign_request(api_secret, params_map) → "query_string&signature=hex"
  2. build_headers(api_key) → {"X-MBX-APIKEY": api_key}
  3. send_request(method, url, headers, timeout, proxies) → (status_code, body)
  4. parse_json_response(body) → dict-like result
```

### Group 2: Parameter Builders (1:1 with current Python methods)

```
C++ Module: binance_order_params

Functions:
  5. build_place_order_params(symbol, side, quantity, order_type, reduce_only,
                              price, stop_price, close_position, time_in_force,
                              position_side, working_type) → params_map
  6. build_place_algo_order_params(symbol, side, algo_type, order_type, quantity,
                                   reduce_only, trigger_price, close_position,
                                   working_type, time_in_force, position_side) → params_map
  7. build_cancel_order_params(symbol, order_id, orig_client_order_id) → params_map
  8. build_cancel_algo_order_params(algo_id) → params_map
```

### Group 3: Price/Quantity Math (pure functions, no side effects)

```
C++ Module: binance_filters

Functions:
  9.  adjust_price(price, tick_size) → float
  10. adjust_quantity(qty, price, step_size, min_qty, min_notional) → float
  11. adjust_close_quantity(qty, step_size, min_qty, round_up) → float
  12. tick_decimals(tick) → int
  13. format_decimal(value, decimals) → string
```

---

## 5. Data Flow Through These Functions

### Opening a MARKET order (single account path):

```
Python: place_order() [module-level, stays Python]
  │
  ├── gating checks (stay Python)
  ├── adjust_quantity() ──────────► C++ binance_filters.adjust_quantity()
  │
  └── client.place_order()
        │
        ├── build params dict ────► C++ binance_order_params.build_place_order_params()
        │
        └── _request("POST", ...)
              │
              ├── add timestamp ──► Python (time.time() * 1000)
              ├── _sign(params) ──► C++ binance_fast_client.sign_request()
              ├── build URL ──────► C++ (string concat)
              ├── HTTP send ──────► C++ binance_fast_client.send_request()
              └── parse JSON ─────► C++ binance_fast_client.parse_json_response()
```

### Placing a STOP_MARKET (SL/TP):

```
Python: place_exchange_stop_order() [stays Python — orchestration]
  │
  ├── adjust_price() ────────────► C++ binance_filters.adjust_price()
  ├── adjust_close_quantity() ───► C++ binance_filters.adjust_close_quantity()
  ├── _tick_decimals() ──────────► C++ binance_filters.tick_decimals()
  ├── format with precision ─────► C++ binance_filters.format_decimal()
  │
  ├── client.place_order() ──────► C++ (param build + sign + send)
  │   on -4120 error:
  └── client.place_algo_order() ─► C++ (param build + sign + send)
```

---

## 6. Summary of What Changes and What Doesn't

| Component | Current | After Migration | Behavior Change |
|-----------|---------|-----------------|-----------------|
| `_sign()` | Python hmac | C++ OpenSSL HMAC | NONE — same HMAC-SHA256 output |
| `_request()` | Python requests lib | C++ libcurl/raw socket | NONE — same HTTP semantics |
| `place_order()` instance method | Python param build | C++ param build + sign + send | NONE — same params, same endpoint |
| `place_algo_order()` instance method | Python param build | C++ param build + sign + send | NONE — same params, same endpoint |
| `cancel_order()` instance method | Python param build | C++ param build + sign + send | NONE — same params, same endpoint |
| `adjust_price()` | Python math.floor | C++ floor | NONE — same rounding |
| `adjust_quantity()` | Python math.floor/ceil | C++ floor/ceil | NONE — same rounding |
| `place_order()` module-level | Python orchestration | **STAYS PYTHON** | N/A |
| `place_exchange_stop_order()` | Python orchestration | **STAYS PYTHON** (calls C++ inner) | N/A |
| All gating checks | Python | **STAYS PYTHON** | N/A |
| All logging | Python | **STAYS PYTHON** | N/A |

**LOGIC IS NOT CHANGED. Only the execution speed of inner operations improves.**
