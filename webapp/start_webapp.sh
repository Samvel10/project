#!/bin/bash
# Starts both the Flask webapp and Cloudflare tunnel.
# Runs forever — use with systemd or as a background process.

cd /var/www/html/new_example_bot
VENV=/var/www/html/new_example_bot/.venv/bin/python
LOG=/var/www/html/new_example_bot/logs

mkdir -p "$LOG"

echo "[WEBAPP] Starting Flask app..."
nohup $VENV /var/www/html/new_example_bot/run_webapp.py >> "$LOG/webapp.log" 2>&1 &
WEBAPP_PID=$!
echo "[WEBAPP] Flask PID=$WEBAPP_PID"

sleep 3

echo "[WEBAPP] Starting Cloudflare tunnel..."
nohup cloudflared tunnel --url http://localhost:8080 --no-autoupdate >> "$LOG/cloudflared.log" 2>&1 &
CF_PID=$!
echo "[WEBAPP] Cloudflare PID=$CF_PID"

sleep 10

URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG/cloudflared.log" | tail -1)
echo "[WEBAPP] Public URL: $URL"
