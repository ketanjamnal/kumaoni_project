# 🏔️ Kumaoni Hybrid NLP Pipeline

**Emotion Intent Detection + Sentiment Polarity Classification for the Kumaoni language**

A bilingual (Kumaoni + English) NLP system combining a hand-crafted rule-based lexicon with TF-IDF machine learning features.

---

## Live Demo

Deploy instantly on [Streamlit Community Cloud](https://streamlit.io/cloud).

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/kumaoni-nlp-pipeline.git
cd kumaoni-nlp-pipeline
```

### 2. Add your trained model files

Copy the `.pkl` files produced by the training notebook into the `models/` directory:

```
models/
  sentiment_model.pkl
  emotion_model.pkl
  word_tfidf.pkl
  char_tfidf.pkl
  rule_scaler.pkl
  sentiment_encoder.pkl
  emotion_encoder.pkl
```

> **Note:** Model files are not included in the repo (they can be large). Train them using the Jupyter notebook and copy them here, or download from the shared Drive folder.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run locally

```bash
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (include the `models/` folder with the `.pkl` files).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy** — done.

---

## Pipeline Architecture

```
Input: Kumaoni + English Text
          │
    Preprocessing
    (lowercase · punctuation · dedup)
          │
    ┌─────┴──────────────────────────┐
    │                                │
Rule-Based Engine            TF-IDF Features
(Lexicon · Negation ·       (Word 5k + Char 2k)
 Intensifiers · Flags)
    │                                │
    └──────── Feature Fusion ────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
  Sentiment Classifier   Emotion Classifier
  (LinearSVC calibrated)  (LinearSVC calibrated)
```

### Features

| Feature Group | Dimensions |
|---|---|
| Word TF-IDF (unigrams + bigrams) | 5,000 |
| Character TF-IDF (3–5 grams) | 2,000 |
| Rule-based (lexicon + flags) | 11 |
| **Total** | **7,011** |

---

## Results

| Task | Model | Accuracy | Macro F1 |
|---|---|---|---|
| Sentiment (3-class) | LinearSVC (calibrated) | ~88% | ~0.76 |
| Emotion (10-class) | LinearSVC (calibrated) | ~75% | ~0.66 |

**Training details:**
- 925 Kumaoni–English sentence pairs
- 80/20 stratified train-test split
- SMOTE oversampling for class imbalance
- 5-fold stratified cross-validation

---

## Emotion Classes

`Factual` · `Joy` · `Negative_Emotion` · `Inquiry` · `Instruction` · `Transactional` · `Greeting` · `Devotion` · `Conflict` · `Empathy` · `Other`

---

## Recommended Next Steps

1. **IndicBERT / MuRIL fine-tuning** — transformer embeddings for Indic languages
2. **Lexicon expansion** — crowd-source more Kumaoni sentiment words
3. **Multi-task learning** — shared BiLSTM trunk with dual output heads
4. **Cross-lingual transfer** — leverage Hindi / Garhwali data

---

## Citation

If you use this work, please cite:

```
@misc{kumaoni-nlp-2025,
  title  = {Kumaoni Hybrid NLP Pipeline: Emotion Intent and Sentiment Classification},
  year   = {2025},
  note   = {GitHub: https://github.com/<your-username>/kumaoni-nlp-pipeline}
}
```
