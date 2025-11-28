import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import sqlite3
from datetime import datetime
import re
import pandas as pd
import os

# ================================
# MODELS
# ================================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        return_all_scores=False
    )

sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()

# ================================
# DATABASE
# ================================
DB_PATH = "data/app.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT,
            sentiment TEXT,
            sentiment_confidence REAL,
            emotion TEXT,
            emoji TEXT,
            model_used TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_result(text, sentiment, confidence, emotion, emoji, model_used):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO analyses (input_text, sentiment, sentiment_confidence, emotion, emoji, model_used, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (text, sentiment, confidence, emotion, emoji, model_used, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ================================
# EMOJI MAP
# ================================
EMOJI_MAP = {
    "joy": "😊",
    "anger": "😡",
    "sadness": "😢",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😮",
    "neutral": "😐",
    "disappointment": "🙁",
    "annoyance": "😒",
    "approval": "🙂",
    "admiration": "🤝",
    "caring": "🤗",
    "confusion": "😕",
    "optimism": "😌",
    "realization": "💡",
}

# ================================
# STRICT SARCASM DETECTION
# ================================
def detect_sarcasm(text: str, vader_score: float, emotion_label: str, hf_sentiment: str) -> bool:
    if not text:
        return False

    text_low = text.lower()
    sarcasm_keywords = [
        "yeah right", "sure", "totally", "as if", "great, just what i needed",
        "just what i needed", "oh great", "amazing job", "nice going", "i love that",
        "love that for me", "thank you so much, not", "thanks a lot, not"
    ]
    if any(kw in text_low for kw in sarcasm_keywords):
        return True

    negative_emotions = {"disappointment", "anger", "annoyance", "sadness", "disgust", "fear"}
    positive_emotions = {"joy", "admiration", "approval", "optimism", "love", "caring"}

    # HF positive but VADER negative & emotion negative → sarcasm
    if hf_sentiment == "positive" and vader_score < -0.25 and emotion_label in negative_emotions:
        return True

    # HF negative but VADER positive & emotion positive → sarcasm
    if hf_sentiment == "negative" and vader_score > 0.25 and emotion_label in positive_emotions:
        return True

    return False

# ================================
# SENTIMENT FUSION
# ================================
def fuse_sentiment(text: str, hf_label: str, vader_sentiment: str, emotion_label: str, vader_score: float) -> str:
    hf_sent = "positive" if hf_label.lower() == "positive" else "negative"

    if detect_sarcasm(text, vader_score, emotion_label, hf_sent):
        return "sarcasm"

    negative_emotions = {"anger", "annoyance", "sadness", "fear", "disappointment", "disgust"}
    positive_emotions = {"joy", "love", "admiration", "approval", "optimism", "caring"}

    if emotion_label in negative_emotions:
        return "negative"
    if emotion_label in positive_emotions:
        return "positive"

    if vader_score > 0.6:
        return "positive"
    if vader_score < -0.6:
        return "negative"

    if -0.05 < vader_score < 0.05:
        return "neutral"

    return hf_sent

# ================================
# Clear text handler (safe)
# ================================
def clear_text():
    st.session_state["user_input"] = ""  # safely update session state

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="centered")
st.title("Social & Collaborative Systems — Sentiment & Emotion Analyzer")

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

user_input = st.text_area(
    "Enter your text here",
    value=st.session_state["user_input"],
    key="user_input",
    height=140
)

col1, col2, col3 = st.columns([1,1,1])
with col1:
    analyze_clicked = st.button("Analyze")
with col2:
    # now this works because clear_text is defined above
    st.button("Clear Text", on_click=clear_text)
with col3:
    load_history = st.button("Load Analysis History")



# Main analysis
if analyze_clicked:
    input_text = (st.session_state["user_input"] or "").strip()
    if not input_text:
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_text = re.sub(r'\s+', ' ', input_text).strip()

        # VADER
        vader = SentimentIntensityAnalyzer()
        vader_score = vader.polarity_scores(cleaned_text)["compound"]
        if vader_score >= 0.05:
            vader_sentiment = "positive"
        elif vader_score <= -0.05:
            vader_sentiment = "negative"
        else:
            vader_sentiment = "neutral"

        # HF
        hf_raw = sentiment_model(cleaned_text)[0]
        hf_label = hf_raw["label"].lower()
        hf_score = float(hf_raw["score"])

        # Emotion
        emotion_raw = emotion_model(cleaned_text)[0]
        emotion_label = emotion_raw["label"].lower()
        emoji = EMOJI_MAP.get(emotion_label, "🙂")

        # Fusion
        final_sentiment = fuse_sentiment(cleaned_text, hf_label, vader_sentiment, emotion_label, vader_score)

        # Save
        save_result(cleaned_text, final_sentiment, hf_score, emotion_label, emoji, "HF + VADER + Fusion")

        # Display
        st.subheader("Analysis Result")
        if final_sentiment == "sarcasm":
            st.markdown("**Final Sentiment:** 🟣 Sarcasm detected")
        else:
            st.markdown(f"**Final Sentiment:** {final_sentiment.capitalize()}")
        st.markdown(f"**Model Confidence (HF):** {hf_score*100:.1f}%")
        vader_desc = "Positive" if vader_score > 0.05 else ("Negative" if vader_score < -0.05 else "Neutral")
        st.markdown(f"**VADER Polarity Score:** {vader_score:.4f} — {vader_desc}")
        st.markdown(f"**Detected Emotion:** {emotion_label.capitalize()} {emoji}")
        st.info(f"HF label: {hf_label.capitalize()} ({hf_score:.2f}) • VADER: {vader_sentiment} ({vader_score:.3f}) • Emotion: {emotion_label}")

# History
if load_history:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT id, input_text, sentiment, sentiment_confidence, emotion, emoji, created_at
        FROM analyses ORDER BY id DESC
    """).fetchall()
    conn.close()
    if not rows:
        st.info("No history yet.")
    else:
        df = pd.DataFrame(rows, columns=["id", "text", "final_sentiment", "model_confidence", "emotion", "emoji", "created_at"])
        st.dataframe(df)
