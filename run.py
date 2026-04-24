"""Run the app on a free port (default 8080). Override with PORT=9000 python run.py."""

import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
