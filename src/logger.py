"""
logger.py
---------
Utility for saving chat logs in JSON format.
"""

import json
import os
from datetime import datetime


def log_interaction(mode, user_input, response):
    # ensure consistent timestamp
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")

    # always create logs in the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    file_path = os.path.join(logs_dir, f"session_{date}_{mode.lower()}_mode.json")

    entry = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "user_input": user_input,
        "model_response": response
    }

    # safely append new entries
    try:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r+", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([entry], f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to log interaction: {e}")
