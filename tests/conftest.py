from __future__ import annotations

import sys
from pathlib import Path


# Ensure tests can import the local `app` package when pytest is executed
# from environments that do not automatically include project root on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
