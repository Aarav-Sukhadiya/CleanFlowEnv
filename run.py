"""Entry point for running CleanFlowEnv locally."""
import os
import threading
import webbrowser

import uvicorn


def open_browser(port: int) -> None:
    """Open the dashboard in the default browser after a short delay."""
    import time
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{port}/docs")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting CleanFlowEnv on http://localhost:{port}")
    print(f"API docs:  http://localhost:{port}/docs")

    # Auto-open API docs in browser
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    uvicorn.run(
        "cleanflow_env.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
