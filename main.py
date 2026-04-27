"""
main.py
=======
Entry point for the piano-playing robot system.

Run this file to launch the full application:
    python main.py

Make sure your Arduino is connected, DroidCam is running, and the
ArUco markers are placed correctly before starting.
"""

from config import AppConfig
from app import PianoBotApp


def main() -> None:
    """Create the app with default configuration and run the main loop."""
    app = PianoBotApp(AppConfig())
    app.run()


if __name__ == "__main__":
    main()
