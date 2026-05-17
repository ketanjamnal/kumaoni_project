"""
Kumaoni Hybrid NLP Pipeline — Streamlit App
Emotion Intent Detection + Sentiment Polarity Classification
"""

import re
import numpy as np
import joblib
import os
import streamlit as st
from scipy.sparse import hstack, csr_matrix

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kumaoni NLP Pipeline",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Hide default Streamlit header */
    #MainMenu, header, footer { visibility: hidden; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0d1b2a 0%, #1e3a5f 50%, #0e7c86 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 { color: #29b6c5; font-size: 2.2rem; margin-bottom: 0.3rem; }
    .hero p  { color: #a8c0cc; font-size: 1rem; margin: 0; }

    /* Card */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    /* Metric big number */
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-sub { font-size: 0.85rem; color: #64748b; }

    /* Confidence bar wrapper */
    .bar-row { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
    .bar-label { width: 130px; font-size: 0.83rem; color: #374151; }
    .bar-track {
        flex: 1; height: 10px; background: #e5e7eb;
        border-radius: 9999px; overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 9999px; }
    .bar-pct { width: 48px; text-align: right; font-size: 0.83rem;
               font-weight: 600; color: #374151; }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.92rem;
        font-weight: 700;
        color: white;
    }
    .badge-positive  { background: #059669; }
    .badge-negative  { background: #dc2626; }
    .badge-neutral   { background: #2563eb; }
    .badge-joy       { background: #d97706; }
    .badge-factual   { background: #7c3aed; }
    .badge-inquiry   { background: #0284c7; }
    .badge-instruction { background: #be185d; }
    .badge-devotion  { background: #b45309; }
    .badge-negative_emotion { background: #9f1239; }
    .badge-conflict  { background: #1e3a5f; }
    .badge-transactional { background: #0f766e; }
    .badge-greeting  { background: #4338ca; }
    .badge-empathy   { background: #0e7c86; }
    .badge-other     { background: #6b7280; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0d1b2a; }
    section[data-testid="stSidebar"] * { color: #a8c0cc !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #29b6c5 !important; }

    /* Text input */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1.5px solid #e2e8f0 !important;
        font-size: 0.97rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #0e7c86 !important;
        box-shadow: 0 0 0 3px rgba(14,124,134,0.15) !important;
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(135deg, #0e7c86, #29b6c5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        width: 100%;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.88; }

    /* Divider */
    hr { border-color: #e2e8f0; margin: 1.2rem 0; }

    /* Example chips */
    .example-chip {
        display: inline-block;
        background: #e0f2f1;
        color: #0e7c86;
        padding: 0.28rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
SENT_NAMES = ["Negative", "Neutral", "Positive"]

SENT_COLORS = {
    "Positive": "#059669",
    "Neutral":  "#2563eb",
    "Negative": "#dc2626",
}

EMO_COLORS = {
    "Joy":              "#d97706",
    "Factual":          "#7c3aed",
    "Inquiry":          "#0284c7",
    "Instruction":      "#be185d",
    "Devotion":         "#b45309",
    "Negative_Emotion": "#9f1239",
    "Conflict":         "#1e3a5f",
    "Transactional":    "#0f766e",
    "Greeting":         "#4338ca",
    "Empathy":          "#0e7c86",
    "Other":            "#6b7280",
}

POSITIVE_WORDS = {
    'ramro': 2, 'khusi': 3, 'badhiya': 2, 'mast': 2, 'sukhi': 2,
    'bhal': 2, 'syooni': 2, 'chaan': 2, 'dail': 2, 'fail': 2,
    'swaad': 2, 'priya': 2, 'sundar': 2, 'khubsoorat': 2,
    'umeed': 1, 'aasha': 1, 'milaap': 1, 'pyaar': 3, 'mamta': 2,
    'prasann': 2, 'harsha': 2, 'anand': 3, 'sukoon': 2,
    'byo': 2, 'byol': 2, 'tyar': 2, 'pooj': 1, 'parv': 2,
    'kaam': 1, 'safal': 3, 'jeet': 3, 'tarakki': 2,
}
NEGATIVE_WORDS = {
    'dukhi': -3, 'bekar': -2, 'bura': -2, 'gussa': -3,
    'pareshan': -2, 'taklif': -2, 'dard': -3, 'rona': -2,
    'chot': -2, 'bimmar': -3, 'rog': -3, 'kharab': -2,
    'muskil': -2, 'dikkat': -2, 'samasya': -2, 'takleef': -2,
    'jhag': -2, 'gali': -3, 'maar': -3, 'kaat': -3, 'daaka': -3,
    'lutna': -3, 'chori': -2, 'nuksaan': -2,
    'dar': -2, 'bhay': -2, 'saap': -2, 'baagh': -2, 'bhalu': -2,
    'jooth': -2, 'gandh': -2, 'beizatti': -3, 'ninda': -2,
    'jhut': -2, 'ghalat': -1,
    'udaas': -2, 'suno': -1, 'akela': -2, 'ekalwaas': -2,
}
NEGATION_WORDS = {'na', 'ni', 'mat', 'nahi', 'bilkul ni', 'koi ni', 'kabhi ni'}
INTENSIFIERS = {
    'bahut': 1.5, 'bhot': 1.5, 'jyaada': 1.4, 'badi': 1.3,
    'bado': 1.3, 'thul': 1.3, 'ati': 1.4, 'bada': 1.3,
    'ghani': 1.3, 'atyant': 1.5,
}
EMOTION_KEYWORDS = {
    'joy_kw':     {'khusi', 'anand', 'sukhi', 'harsha', 'bhal', 'mast', 'byo', 'tyar'},
    'anger_kw':   {'gussa', 'jhag', 'gali', 'maar', 'kaat', 'risai', 'naaraj'},
    'fear_kw':    {'dar', 'bhay', 'dara', 'darak', 'saap', 'baagh', 'bhalu'},
    'sadness_kw': {'dukhi', 'rona', 'udaas', 'suno', 'ekalwaas', 'akela'},
    'disgust_kw': {'jooth', 'gandh', 'bekar', 'bekaar', 'ganda', 'kharab'},
    'surprise_kw':{'ajab', 'gajab', 'achanak', 'ekaaek', 'acharya'},
}
RULE_FEATURE_COLS = [
    'rule_score', 'negation_count', 'intensifier_count',
    'pos_word_count', 'neg_word_count',
    'joy_kw', 'anger_kw', 'fear_kw', 'sadness_kw', 'disgust_kw', 'surprise_kw',
]

EXAMPLES = [
    ("khusi chhu mast chhu", "i am happy and feeling great"),
    ("badi dukhi chhu", "i am very sad"),
    ("kaam kas chal rau", "how is the work going"),
    ("gussa mat ho", "do not be angry"),
    ("bahut sundar jagah chhu", "this is a very beautiful place"),
    ("dar lag rau", "i am feeling scared"),
]


# ─── Helper functions ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def rule_based_features(text: str) -> dict:
    words = text.split()
    score = 0.0
    pos_count = neg_count = negation_count = intensifier_count = 0
    intensifier_mult = 1.0

    for i, word in enumerate(words):
        context = words[max(0, i - 3):i]
        is_negated = any(w in NEGATION_WORDS for w in context)
        if word in INTENSIFIERS:
            intensifier_mult = INTENSIFIERS[word]
            intensifier_count += 1
        elif word in NEGATION_WORDS:
            negation_count += 1
        else:
            intensifier_mult = 1.0
        if word in POSITIVE_WORDS:
            val = POSITIVE_WORDS[word] * intensifier_mult
            score += -val if is_negated else val
            pos_count += 1
        elif word in NEGATIVE_WORDS:
            val = NEGATIVE_WORDS[word] * intensifier_mult
            score += -val if is_negated else val
            neg_count += 1

    features = {
        'rule_score': score,
        'negation_count': negation_count,
        'intensifier_count': intensifier_count,
        'pos_word_count': pos_count,
        'neg_word_count': neg_count,
    }
    word_set = set(words)
    for feat_name, kw_set in EMOTION_KEYWORDS.items():
        features[feat_name] = int(bool(word_set & kw_set))
    return features


@st.cache_resource(show_spinner=False)
def load_models():
    """Load all pkl artifacts. Returns None on failure."""
    model_dir = os.environ.get("MODEL_DIR", "models")
    required = [
        "sentiment_model.pkl", "emotion_model.pkl",
        "word_tfidf.pkl", "char_tfidf.pkl",
        "rule_scaler.pkl", "sentiment_encoder.pkl", "emotion_encoder.pkl",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        return None, missing
    try:
        return {
            "sent_clf":    joblib.load(os.path.join(model_dir, "sentiment_model.pkl")),
            "emo_clf":     joblib.load(os.path.join(model_dir, "emotion_model.pkl")),
            "word_vec":    joblib.load(os.path.join(model_dir, "word_tfidf.pkl")),
            "char_vec":    joblib.load(os.path.join(model_dir, "char_tfidf.pkl")),
            "scaler":      joblib.load(os.path.join(model_dir, "rule_scaler.pkl")),
            "sent_enc":    joblib.load(os.path.join(model_dir, "sentiment_encoder.pkl")),
            "emo_enc":     joblib.load(os.path.join(model_dir, "emotion_encoder.pkl")),
        }, []
    except Exception as e:
        return None, [str(e)]


def predict(models, kumaoni_text: str, english_text: str = "") -> dict:
    combined = clean_text(kumaoni_text) + " " + clean_text(english_text)
    rule_feats = rule_based_features(combined)
    rule_arr = models["scaler"].transform([[rule_feats[c] for c in RULE_FEATURE_COLS]])
    x_w = models["word_vec"].transform([combined])
    x_c = models["char_vec"].transform([combined])
    X_new = hstack([x_w, x_c, csr_matrix(rule_arr)])

    sent_pred  = models["sent_clf"].predict(X_new)[0]
    sent_proba = models["sent_clf"].predict_proba(X_new)[0]
    emo_pred   = models["emo_clf"].predict(X_new)[0]
    emo_proba  = models["emo_clf"].predict_proba(X_new)[0]

    sent_classes = list(models["sent_enc"].classes_)
    emo_classes  = list(models["emo_enc"].classes_)
    # Map encoded int back to display name
    sent_label = SENT_NAMES[sent_pred] if sent_pred < len(SENT_NAMES) else str(sent_pred)
    emo_label  = emo_classes[emo_pred]

    return {
        "sentiment":        sent_label,
        "sent_confidence":  float(sent_proba.max()),
        "sent_proba":       dict(zip(["Negative", "Neutral", "Positive"], sent_proba)),
        "emotion":          emo_label,
        "emo_confidence":   float(emo_proba.max()),
        "emo_proba":        dict(zip(emo_classes, emo_proba)),
        "rule_score":       rule_feats["rule_score"],
    }


def confidence_bar(label: str, pct: float, color: str):
    bar_html = f"""
    <div class="bar-row">
        <div class="bar-label">{label}</div>
        <div class="bar-track">
            <div class="bar-fill" style="width:{pct*100:.1f}%;background:{color};"></div>
        </div>
        <div class="bar-pct">{pct*100:.1f}%</div>
    </div>"""
    st.markdown(bar_html, unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏔️ Kumaoni NLP")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This app runs the **Kumaoni Hybrid NLP Pipeline** — "
        "a bilingual (Kumaoni + English) system for:\n\n"
        "- 🎭 **Sentiment** classification (3-class)\n"
        "- 💬 **Emotion intent** detection (10-class)"
    )
    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown(
        "**Features**\n"
        "- Word TF-IDF (5,000)\n"
        "- Char TF-IDF (2,000)\n"
        "- Rule-based lexicon (11)\n\n"
        "**Model:** LinearSVC (calibrated)\n\n"
        "**Balanced with:** SMOTE oversampling\n\n"
        "**Trained on:** 925 sentence pairs"
    )
    st.markdown("---")
    st.markdown("### Dataset Stats")
    st.markdown(
        "| Class | Count |\n"
        "|-------|-------|\n"
        "| Neutral | 735 |\n"
        "| Negative | 119 |\n"
        "| Positive | 71 |"
    )
    st.markdown("---")
    st.caption("Model artifacts loaded from `/models/`")


# ─── Main layout ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏔️ Kumaoni Hybrid NLP Pipeline</h1>
    <p>Emotion Intent Detection &amp; Sentiment Polarity Classification for the Kumaoni language</p>
</div>
""", unsafe_allow_html=True)

# Load models
models, missing = load_models()
models_loaded = models is not None

if not models_loaded:
    st.warning(
        "⚠️ **Model files not found.** Place the trained `.pkl` files in the `models/` directory.\n\n"
        f"Missing: `{', '.join(missing)}`\n\n"
        "Expected files: `sentiment_model.pkl`, `emotion_model.pkl`, `word_tfidf.pkl`, "
        "`char_tfidf.pkl`, `rule_scaler.pkl`, `sentiment_encoder.pkl`, `emotion_encoder.pkl`"
    )

# ── Input section ─────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Input Text</div>', unsafe_allow_html=True)

    # Session state for prefill
    if "kumaoni_input" not in st.session_state:
        st.session_state.kumaoni_input = ""
    if "english_input" not in st.session_state:
        st.session_state.english_input = ""

    kumaoni = st.text_area(
        "Kumaoni Text",
        value=st.session_state.kumaoni_input,
        placeholder="e.g. khusi chhu mast chhu",
        height=100,
        key="kumaoni_ta",
    )
    english = st.text_area(
        "English Translation (optional — helps the model)",
        value=st.session_state.english_input,
        placeholder="e.g. i am happy and feeling great",
        height=80,
        key="english_ta",
    )

    run = st.button("Analyse →", disabled=not models_loaded)

    st.markdown("---")
    st.markdown('<div class="card-title">Try an example</div>', unsafe_allow_html=True)
    for kum, eng in EXAMPLES:
        if st.button(f"📝 {kum}", key=f"ex_{kum}"):
            st.session_state.kumaoni_input = kum
            st.session_state.english_input = eng
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── Output section ────────────────────────────────────────────────────────────
with col_out:
    if run and kumaoni.strip():
        result = predict(models, kumaoni, english)

        # ── Sentiment card ──────────────────────────────────────────────────
        sent = result["sentiment"]
        sent_color = SENT_COLORS.get(sent, "#374151")
        badge_cls = f"badge-{sent.lower()}"

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Sentiment</div>
            <span class="badge {badge_cls}">{sent}</span>
            <span style="color:#64748b;font-size:0.88rem;margin-left:12px;">
                {result['sent_confidence']*100:.1f}% confidence
            </span>
            <br><br>
        """, unsafe_allow_html=True)

        for cls in ["Positive", "Neutral", "Negative"]:
            confidence_bar(cls, result["sent_proba"].get(cls, 0), SENT_COLORS[cls])

        rule_score = result["rule_score"]
        rule_color = "#059669" if rule_score > 0 else "#dc2626" if rule_score < 0 else "#64748b"
        st.markdown(f"""
            <br>
            <div style="font-size:0.82rem;color:#64748b;">
                Rule-based score: <span style="color:{rule_color};font-weight:700;">{rule_score:+.2f}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Emotion card ────────────────────────────────────────────────────
        emo = result["emotion"]
        emo_color = EMO_COLORS.get(emo, "#6b7280")
        badge_cls_e = f"badge-{emo.lower().replace(' ', '_')}"

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Emotion Intent</div>
            <span class="badge" style="background:{emo_color};">{emo}</span>
            <span style="color:#64748b;font-size:0.88rem;margin-left:12px;">
                {result['emo_confidence']*100:.1f}% confidence
            </span>
            <br><br>
        """, unsafe_allow_html=True)

        # Top 5 emotions
        top5 = sorted(result["emo_proba"].items(), key=lambda x: -x[1])[:5]
        for cls, prob in top5:
            c = EMO_COLORS.get(cls, "#6b7280")
            confidence_bar(cls, prob, c)

        st.markdown("</div>", unsafe_allow_html=True)

    elif run and not kumaoni.strip():
        st.warning("Please enter some Kumaoni text to analyse.")
    else:
        st.markdown("""
        <div class="card" style="border: 2px dashed #e2e8f0; background:#f8fafc; text-align:center; padding: 3rem 1rem;">
            <div style="font-size: 2.5rem;">🏔️</div>
            <div style="color:#94a3b8; margin-top:0.8rem; font-size:0.98rem;">
                Enter a Kumaoni sentence on the left<br>and click <strong>Analyse →</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Bottom info strip ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("88%", "Sentiment Accuracy", "#0e7c86"),
    ("75%", "Emotion Accuracy", "#d97706"),
    ("~0.76", "Sentiment Macro F1", "#7c3aed"),
    ("~0.66", "Emotion Macro F1", "#be185d"),
]
for col, (val, label, color) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="metric-val" style="color:{color};">{val}</div>
            <div class="metric-sub">{label}</div>
        </div>""", unsafe_allow_html=True)
