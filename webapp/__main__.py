"""``python -m webapp [--host 127.0.0.1] [--port 8000]``: serve the browser simulator (local tool, no auth)."""
from __future__ import annotations

import argparse


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m webapp", description="Browser ADCS simulator server.")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; no authentication)")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    import uvicorn

    from webapp.server.app import create_app

    print(f"Serving the ADCS browser simulator at http://{a.host}:{a.port}/", flush=True)
    uvicorn.run(create_app(), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
