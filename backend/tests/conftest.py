import os
import sys
import tempfile
from pathlib import Path

# Isolated data dir and a known key, set before the app modules import settings.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="cashguard-test-"))
os.environ.setdefault("API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
