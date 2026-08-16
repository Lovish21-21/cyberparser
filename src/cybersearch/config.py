from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_ROOT = PROJECT_ROOT / "dataset" / "text"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_TOP_K = 5
