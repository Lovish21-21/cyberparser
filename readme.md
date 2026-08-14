# CTI File Parser

A cyber threat intelligence (CTI) text parsing project that combines:

- a spaCy-based NER pipeline for entity extraction from CTI reports
- a lightweight TF-IDF retrieval layer for finding the most relevant document or report
- a Streamlit UI for querying the corpus and parsing the best match

This project is designed to help with CTI report workflows such as:

- searching for related CTI documents
- locating the best matching report to a user query
- extracting entities such as malware, groups, organizations, tools, tactics, dates, and locations

---

## Project structure

```text
CTI_file_parser/
├── dataset/
│   ├── dev.json
│   ├── train_ext.json
│   └── text/
│       ├── train/
│       └── dev/
├── models/
│   └── cti_spacy_ner/
├── src/
│   ├── cyberparser/
│   │   ├── main.py
│   │   ├── preprocessing.py
│   │   └── train.py
│   ├── cybersearch/
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── corpus.py
│   │   ├── engine.py
│   │   ├── index.py
│   │   ├── preprocess.py
│   │   └── __init__.py
│   └── __init__.py
├── requirements.txt
├── streamlit_app.py
├── readme.md
└── tests/
```

---

## Features

### 1. CTI entity extraction
The training pipeline in [src/cyberparser/train.py](src/cyberparser/train.py) builds a blank spaCy English NER model, adds labels extracted from the dataset, and trains a model using hyperparameter search across several candidate values.

It chooses the best-performing config using:

- entity-level F1 score
- token accuracy

### 2. Retrieval layer
The search module in [src/cybersearch](src/cybersearch) indexes text documents and ranks them using a TF-IDF cosine similarity search. This helps find the most relevant CTI text for a user query.

### 3. Streamlit app
The Streamlit app in [streamlit_app.py](streamlit_app.py) lets a user:

- enter a search query
- find the closest CTI documents in the text corpus
- select the best matching result
- run the NER parser on that file/text and inspect extracted entities

---

## Dataset format

The project uses two dataset styles:

### JSONL annotation corpus
The files in [dataset](dataset) such as `train_ext.json` and `dev.json` are used for training and evaluation.

Each record typically contains text and label information like:

- `text`
- `tokens`
- `all_tags`, `labels`, or `all_labels`

### Raw text corpus
For retrieval, the app can read plain text files placed under:

```text
dataset/text/
├── train/
│   └── *.txt
└── dev/
    └── *.txt
```

This is the recommended structure for searching real CTI reports and matching the best file to a query.

---

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries include:

- spaCy
- scikit-learn
- numpy
- streamlit

---

## Training the model

Run training from the project root:

```bash
python src/cyberparser/main.py --train dataset/train_ext.json --dev dataset/dev.json --output models/cti_spacy_ner
```

This will:

- load the JSONL training data
- build examples for spaCy
- run a hyperparameter sweep
- train the best model
- save it under `models/cti_spacy_ner`

---

## Running the app

Start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

Then in the app:

1. enter a CTI query like `APT29 phishing campaign`
2. set the text corpus root to `dataset/text`
3. click the button to search and parse the best match
4. view the extracted entities for the selected file

---

## Example workflow

```text
User query: "ransomware campaign using TrickBot"
        ↓
Search the text corpus for related reports
        ↓
Pick the best matching file
        ↓
Run CTI NER over that file
        ↓
Return entity groups such as:
  malware, group, org, tactic, date, loc, tool
```

---

## Notes

- This is a practical CTI parser and retrieval prototype.
- The retrieval layer is intentionally lightweight and easy to extend.
- The training pipeline uses a hyperparameter sweep and picks the best-performing model on dev data.
- Some noisy CTI text may still create annotation alignment warnings because of markdown, links, tables, and malformed offsets in the source data.

---

## Potential next steps

- improve entity alignment cleaning for noisy CTI text
- add chunked retrieval for long reports
- add BM25 or hybrid dense+sparse search
- build a better evaluation dashboard for F1 / precision / recall
- package the app for deployment

---

## License

This project is intended for research and internal CTI workflows. Update the license if you plan to share or deploy it more broadly.
