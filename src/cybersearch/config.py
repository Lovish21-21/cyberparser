from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_ROOT = PROJECT_ROOT.parent / "anno-ctr-lrec-coling-2024" / "AnnoCTR" / "text"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_TOP_K = 5
