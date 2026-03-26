#!/bin/bash
# PostgreSQL backup script — runs daily via cron
# Keeps last 7 backups on server + always writes latest to data/db_backup_latest.sql

BACKUP_DIR="/var/www/html/new_example_bot/data/backups"
LATEST="/var/www/html/new_example_bot/data/db_backup_latest.sql"
DATE=$(date +%Y-%m-%d_%H-%M)
FILE="$BACKUP_DIR/tradebotdb_$DATE.sql"

mkdir -p "$BACKUP_DIR"

PGPASSWORD='TradeBotDB2026' pg_dump \
    -h 127.0.0.1 -U tradebot -d tradebotdb \
    --no-password --clean --if-exists \
    -f "$FILE"

if [ $? -eq 0 ]; then
    cp "$FILE" "$LATEST"
    echo "[BACKUP] $FILE created ($(du -sh "$FILE" | cut -f1))"
    # Keep only last 7 backups
    ls -t "$BACKUP_DIR"/tradebotdb_*.sql | tail -n +8 | xargs -r rm
else
    echo "[BACKUP] FAILED at $DATE"
fi
