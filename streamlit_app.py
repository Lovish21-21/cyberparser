from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# These imports must come AFTER the sys.path insert above, since `cybersearch`
# and `cyberparser` live under src/ and aren't importable until that path is added.
from cybersearch.engine import TfidfSearchEngine
from cyberparser.train import parse_text_to_dict, train_spacy_ner

DATASET_TEXT_ROOT = PROJECT_ROOT / "dataset" / "text"
DATASET_TRAIN = PROJECT_ROOT / "dataset" / "train_ext.json"
DATASET_DEV = PROJECT_ROOT / "dataset" / "dev.json"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "cti_spacy_ner"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@st.cache_resource
def load_search_engine(text_root: str | Path, artifact_dir: str | Path):
    text_root = Path(text_root)
    artifact_dir = Path(artifact_dir)

    if (artifact_dir / "vectorizer.pkl").exists() and (artifact_dir / "matrix.pkl").exists():
        return TfidfSearchEngine.from_artifacts(artifact_dir)

    if not text_root.exists():
        raise FileNotFoundError(f"Text corpus root not found: {text_root}")

    engine = TfidfSearchEngine.from_text_root(text_root)
    engine.save(artifact_dir)
    return engine


@st.cache_resource
def load_or_train_model(model_dir: str | Path, train_path: str | Path, dev_path: str | Path, force_retrain: bool = False):
    model_dir = Path(model_dir)
    train_path = Path(train_path)
    dev_path = Path(dev_path)

    if model_dir.exists() and not force_retrain:
        try:
            import spacy

            return spacy.load(str(model_dir))
        except OSError:
            pass

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset missing: {train_path}")

    model, _ = train_spacy_ner(
        train_path=str(train_path),
        dev_path=str(dev_path) if dev_path.exists() else None,
        output_dir=str(model_dir),
        n_iter=10,
        dropout=0.2,
        batch_size=8,
    )
    return model


st.set_page_config(page_title="CTI Search + Parser", page_icon="🛡️", layout="wide")

st.title("CTI Search + Entity Parser")
st.caption("Search the best-matching CTI record and parse the matching report text with the spaCy NER model.")

with st.sidebar:
    st.header("Settings")
    query = st.text_input(
        "Search query",
        value="APT29 phishing campaign",
        placeholder="Example: ransomware, APT29, credential theft",
    )
    top_k = st.slider("Number of related matches", min_value=1, max_value=10, value=5)
    text_root = st.text_input("Text corpus root", value=str(DATASET_TEXT_ROOT))
    train_path = st.text_input("Training JSONL", value=str(DATASET_TRAIN))
    dev_path = st.text_input("Dev JSONL", value=str(DATASET_DEV))
    model_dir = st.text_input("Model output directory", value=str(DEFAULT_MODEL_DIR))
    force_retrain = st.checkbox("Force retrain model", value=False, help="Retrain from train/dev JSONL even if a saved model already exists.")

if st.button("Search and parse best match", type="primary"):
    if not query.strip():
        st.warning("Please enter a query first.")
        st.stop()

    with st.spinner("Searching related CTI records and parsing the best match..."):
        try:
            engine = load_search_engine(text_root, DEFAULT_ARTIFACT_DIR)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()

        hits = engine.search(query, top_k=top_k)
        if not hits:
            st.warning("No matching CTI records were found for this query. Put your .txt reports under the text corpus root and try again.")
            st.stop()

        selected = hits[0].document
        model = load_or_train_model(model_dir, train_path, dev_path, force_retrain=force_retrain)
        entities = parse_text_to_dict(model, selected.text)

    st.subheader("Top related records")
    for hit in hits:
        doc = hit.document
        st.markdown(
            f"### Rank {hit.rank} — score {hit.score:.4f}\n"
            f"**ID:** {doc.doc_id}  |  **Title:** {doc.title[:120]}\n\n"
            f"{doc.text[:350]}"
        )
        st.markdown("---")

    st.subheader(f"Best match: {selected.doc_id}")
    st.write(f"**Title:** {selected.title}")
    st.write(f"**Source file:** {selected.filename}")
    st.code(selected.text[:2500], language="text")

    st.subheader("Parsed entities")
    if not entities:
        st.info("No entities were extracted from the selected CTI record.")
    else:
        st.json(entities, expanded=True)
else:
    st.info("Enter a query and click the button to search and parse the most relevant CTI record.")