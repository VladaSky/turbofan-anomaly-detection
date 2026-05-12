"""
Production entrypoint — reads PORT from the environment (set by Render).
Usage:  python main.py
"""
import os
import uvicorn
from src.api_lite import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
