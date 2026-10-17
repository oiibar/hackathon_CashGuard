import os
import sys
import tempfile
from pathlib import Path

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="cashguard-test-")
os.environ["API_KEY"] = "test-key"
os.environ["AUTO_RETRAIN"] = "false"
os.environ.pop("DATABASE_URL", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
