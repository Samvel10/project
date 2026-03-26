from __future__ import annotations
from flask import Blueprint, render_template, jsonify, g, request
from webapp.auth import login_required
from webapp import bot_data

financial_bp = Blueprint("financial", __name__)


@financial_bp.route("/dashboard/financial")
@login_required
def financial():
    accounts = bot_data.get_all_accounts_with_state()
    return render_template("dashboard/financial.html", user=g.current_user, accounts=accounts)


@financial_bp.route("/api/account/<int:idx>/balance")
@login_required
def api_balance(idx):
    accounts = bot_data.get_all_accounts_with_state()
    if idx >= len(accounts):
        return jsonify({"error": "Account not found"}), 404
    acc = accounts[idx]
    if not acc["api_key"] or not acc["api_secret"]:
        return jsonify({"error": "No API credentials"}), 400
    balances = bot_data.get_binance_account_balance(acc["api_key"], acc["api_secret"])
    usdt = next((b for b in balances if b.get("asset") == "USDT"), None)
    return jsonify({
        "name":            acc["name"],
        "trade_enabled":   acc["trade_enabled"],
        "wallet_balance":  float(usdt.get("balance", 0)) if usdt else 0,
        "available":       float(usdt.get("availableBalance", 0)) if usdt else 0,
        "unrealized_pnl":  float(usdt.get("crossUnPnl", 0)) if usdt else 0,
    })


@financial_bp.route("/api/account/<int:idx>/positions")
@login_required
def api_positions(idx):
    accounts = bot_data.get_all_accounts_with_state()
    if idx >= len(accounts):
        return jsonify([])
    acc = accounts[idx]
    if not acc["api_key"] or not acc["api_secret"]:
        return jsonify([])
    positions = bot_data.get_binance_positions(acc["api_key"], acc["api_secret"])
    result = []
    for p in positions:
        amt = float(p.get("positionAmt", 0))
        ep  = float(p.get("entryPrice", 0))
        mp  = float(p.get("markPrice", 0))
        pnl = float(p.get("unRealizedProfit", 0))
        pnl_pct = ((mp - ep) / ep * 100 * (1 if amt > 0 else -1)) if ep > 0 else 0
        result.append({
            "symbol":      p.get("symbol"),
            "side":        "LONG" if amt > 0 else "SHORT",
            "size":        abs(amt),
            "entry_price": ep,
            "mark_price":  mp,
            "pnl":         round(pnl, 4),
            "pnl_pct":     round(pnl_pct, 2),
            "leverage":    p.get("leverage"),
            "notional":    abs(float(p.get("notional", 0))),
        })
    result.sort(key=lambda x: abs(x["pnl"]), reverse=True)
    return jsonify(result)


@financial_bp.route("/api/account/<int:idx>/trades/<symbol>")
@login_required
def api_trades(idx, symbol):
    accounts = bot_data.get_all_accounts_with_state()
    if idx >= len(accounts):
        return jsonify([])
    acc = accounts[idx]
    if not acc["api_key"] or not acc["api_secret"]:
        return jsonify([])
    trades = bot_data.get_binance_trade_history(acc["api_key"], acc["api_secret"], symbol.upper())
    return jsonify(trades)


@financial_bp.route("/api/chart/<symbol>")
@login_required
def api_chart(symbol):
    """Return candlestick + trade marker data for financial chart."""
    interval = request.args.get("interval", "1m")
    limit    = min(int(request.args.get("limit", 300)), 1000)
    account_idx = int(request.args.get("account", 0))

    candles = bot_data.get_klines(symbol.upper(), interval, limit)
    return jsonify({"candles": candles})


@financial_bp.route("/api/market/symbols")
@login_required
def api_market_symbols():
    q = request.args.get("q", "").upper().strip()
    symbols = bot_data.get_binance_usdt_symbols()
    if q:
        symbols = [s for s in symbols if q in s]
    return jsonify(symbols[:80])


@financial_bp.route("/api/market/ticker/<symbol>")
@login_required
def api_ticker(symbol):
    t = bot_data.get_binance_ticker_24h(symbol.upper())
    m = bot_data.get_binance_mark_price(symbol.upper())
    return jsonify({
        "symbol":        t.get("symbol", symbol),
        "lastPrice":     t.get("lastPrice", "0"),
        "priceChange":   t.get("priceChange", "0"),
        "priceChangePct":t.get("priceChangePercent", "0"),
        "high":          t.get("highPrice", "0"),
        "low":           t.get("lowPrice", "0"),
        "volume":        t.get("volume", "0"),
        "quoteVolume":   t.get("quoteVolume", "0"),
        "markPrice":     m.get("markPrice", "0"),
        "indexPrice":    m.get("indexPrice", "0"),
        "fundingRate":   m.get("lastFundingRate", "0"),
    })


@financial_bp.route("/api/account/<int:idx>/orders")
@login_required
def api_open_orders(idx):
    accounts = bot_data.get_all_accounts_with_state()
    if idx >= len(accounts):
        return jsonify([])
    acc = accounts[idx]
    if not acc["api_key"]:
        return jsonify([])
    orders = bot_data.get_binance_open_orders(acc["api_key"], acc["api_secret"])
    result = []
    for o in orders:
        result.append({
            "symbol":        o.get("symbol"),
            "orderId":       o.get("orderId"),
            "type":          o.get("type"),
            "side":          o.get("side"),
            "price":         float(o.get("price", 0)),
            "origQty":       float(o.get("origQty", 0)),
            "executedQty":   float(o.get("executedQty", 0)),
            "status":        o.get("status"),
            "time":          o.get("time"),
            "reduceOnly":    o.get("reduceOnly", False),
            "closePosition": o.get("closePosition", False),
            "stopPrice":     float(o.get("stopPrice", 0)),
            "activatePrice": float(o.get("activatePrice", 0)),
            "priceRate":     o.get("priceRate"),
        })
    return jsonify(result)


@financial_bp.route("/api/account/<int:idx>/income")
@login_required
def api_income(idx):
    accounts = bot_data.get_all_accounts_with_state()
    if idx >= len(accounts):
        return jsonify([])
    acc = accounts[idx]
    if not acc["api_key"]:
        return jsonify([])
    income_type = request.args.get("type")
    symbol_filter = request.args.get("symbol", "").upper().strip()
    data = bot_data.get_binance_income_history(acc["api_key"], acc["api_secret"],
                                                income_type=income_type, limit=100)
    if symbol_filter:
        data = [r for r in data if r.get("symbol", "").upper() == symbol_filter]
    return jsonify(sorted(data, key=lambda x: x.get("time", 0), reverse=True))


@financial_bp.route("/api/account/<int:idx>/assets")
@login_required
def api_assets(idx):
    accounts = bot_data.get_all_accounts_with_state()
    if idx >= len(accounts):
        return jsonify([])
    acc = accounts[idx]
    if not acc["api_key"]:
        return jsonify([])
    assets = bot_data.get_binance_all_balances(acc["api_key"], acc["api_secret"])
    result = []
    for a in assets:
        wb  = float(a.get("walletBalance", 0))
        upnl = float(a.get("unrealizedProfit", 0))
        result.append({
            "asset":            a.get("asset"),
            "walletBalance":    round(wb, 6),
            "unrealizedProfit": round(upnl, 6),
            "marginBalance":    round(float(a.get("marginBalance", wb + upnl)), 6),
            "availableBalance": round(float(a.get("availableBalance", 0)), 6),
        })
    result.sort(key=lambda x: abs(x["walletBalance"]), reverse=True)
    return jsonify(result)


@financial_bp.route("/api/account/<int:idx>/listenkey", methods=["POST"])
@login_required
def api_create_listenkey(idx):
    accounts = bot_data.get_all_accounts_with_state()
    if idx >= len(accounts):
        return jsonify({"error": "Not found"}), 404
    acc = accounts[idx]
    if not acc["api_key"]:
        return jsonify({"error": "No API key"}), 400
    key = bot_data.create_listen_key(acc["api_key"])
    if not key:
        return jsonify({"error": "Failed to create listen key"}), 500
    return jsonify({"listenKey": key})


@financial_bp.route("/api/account/<int:idx>/listenkey/renew", methods=["POST"])
@login_required
def api_renew_listenkey(idx):
    accounts = bot_data.get_all_accounts_with_state()
    if idx >= len(accounts):
        return jsonify({"error": "Not found"}), 404
    acc = accounts[idx]
    listen_key = (request.get_json() or {}).get("listenKey", "")
    if not listen_key:
        return jsonify({"error": "No listenKey"}), 400
    ok = bot_data.renew_listen_key(acc["api_key"], listen_key)
    return jsonify({"ok": ok})
