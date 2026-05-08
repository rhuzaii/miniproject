"""
backend/app.py — Flask API server.

Run (from project root, with venv active):
  python backend/app.py

Endpoints:
  POST /trigger-command   — receive gesture command, forward to AWS API Gateway
  GET  /health            — health check

Port: 5001 (NOT 5000 — macOS Monterey+ uses 5000 for AirPlay Receiver)
"""

import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from backend.routes import commands_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

    CORS(app)

    app.register_blueprint(commands_bp)

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok", "service": "gesture-alexa-backend"}

    return app


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5001"))
    print(f"[Flask] Starting on http://localhost:{port}")
    print(f"[Flask] AWS Gateway: {os.getenv('AWS_API_GATEWAY_URL', '(not set)')}")
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)
