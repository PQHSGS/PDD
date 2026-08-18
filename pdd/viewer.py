"""PDD Interactive Viewer launcher entrypoint.

Run via:
    python -m pdd.viewer [--port 8000]
"""
from pdd.viewer_server import main

if __name__ == "__main__":
    main()
