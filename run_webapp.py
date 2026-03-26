"""Entry point for the trading dashboard web app."""
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from webapp.app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("WEBAPP_PORT", 8080))
    host = os.environ.get("WEBAPP_HOST", "0.0.0.0")
    debug = os.environ.get("WEBAPP_DEBUG", "false").lower() == "true"
    print(f"[WEBAPP] Starting on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
