"""NightSeries development and container entry point."""
import os
from pathlib import Path

from waitress import serve

from nightseries.web import create_app


def main():
    local_data = Path(__file__).absolute().parent / ".nightseries"
    app = create_app(os.environ.get("NIGHTSERIES_DATA", str(local_data)))
    serve(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "18947")),
        threads=5,
        max_request_body_size=122 * 1024 * 1024,
        channel_timeout=120,
    )


if __name__ == "__main__":
    main()
