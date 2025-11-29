import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import sqlite3
from datetime import datetime
import re
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')


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
    "disgust": "🤢"
}

# ================================
# STRICT SARCASM DETECTION
# (unchanged)
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
# (unchanged)
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
# Helper: sentiment color badge (HTML)
# ================================
def sentiment_badge_html(label: str) -> str:
    color_map = {
        "positive": "#1a9850",   # green
        "negative": "#d73027",   # red
        "neutral": "#999999",    # gray
        "sarcasm": "#6a51a3"     # purple
    }
    color = color_map.get(label, "#333333")
    return f'<div style="display:inline-block;padding:8px 14px;border-radius:12px;background:{color};color:#ffffff;font-weight:600">{label.capitalize()}</div>'

# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="centered")
st.title("Sentiment & Emotion Analyzer")
st.write("Enter text below and click **Analyze**. The dashboard shows model comparison and visualizations.")

# session state init
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
    st.button("Clear Text", on_click=clear_text)
with col3:
    load_history = st.button("Load Analysis History")

# ================================
# Main analysis (core logic unchanged)
# ================================
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

        # HF sentiment
        hf_raw = sentiment_model(cleaned_text)[0]
        hf_label = hf_raw["label"].lower()
        hf_score = float(hf_raw["score"])

        # Emotion (keep same; get emotion score if available)
        emotion_raw = emotion_model(cleaned_text)[0]
        emotion_label = emotion_raw["label"].lower()
        emotion_score = float(emotion_raw.get("score", 1.0))  # score often available
        emoji = EMOJI_MAP.get(emotion_label, "🙂")

        # Fusion
        final_sentiment = fuse_sentiment(cleaned_text, hf_label, vader_sentiment, emotion_label, vader_score)

        # Save
        save_result(cleaned_text, final_sentiment, hf_score, emotion_label, emoji, "HF + VADER + Fusion")

        # ---------------- Dashboard / Metrics ----------------
        st.subheader("Multi-Model Comparison Dashboard")

        # Top metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            # HF sentiment metric
            st.markdown("**HuggingFace (HF) Sentiment**")
            st.metric(label=f"HF: {hf_label.capitalize()}", value=f"{hf_score*100:.1f}%", delta=None)

        with m2:
            # VADER metric
            vader_desc = "Positive" if vader_score > 0.05 else ("Negative" if vader_score < -0.05 else "Neutral")
            st.markdown("**VADER Polarity**")
            st.metric(label=f"VADER: {vader_desc}", value=f"{vader_score:.3f}", delta=None)

        with m3:
            st.markdown("**Emotion Detector**")
            st.metric(label=f"{emotion_label.capitalize()}", value=f"{emoji}", delta=None)

        # Final sentiment badge
        st.markdown("**Final fused sentiment**")
        st.markdown(sentiment_badge_html(final_sentiment), unsafe_allow_html=True)

        # Conflict indicator (HF vs VADER)
        # strong disagreement: HF positive & VADER very negative, or vice-versa
        if (hf_label == "positive" and vader_score < -0.25) or (hf_label == "negative" and vader_score > 0.25):
            st.warning("Strong disagreement between HF and VADER detected — fusion logic applied.")
        else:
            st.success("HF and VADER are in agreement or only mild disagreement.")

        # ---------------- Plotly comparison chart ----------------
        # Normalize VADER to 0..1 for plotting: (vader_score + 1) / 2
        vader_norm = (vader_score + 1.0) / 2.0
        # HF and emotion already in 0..1 range (hf_score, emotion_score)
        df_plot = pd.DataFrame({
            "Model": ["HuggingFace (HF)", "VADER (norm)", "Emotion Model"],
            "Score": [hf_score, vader_norm, emotion_score]
        })

        fig = px.bar(
            df_plot,
            x="Model",
            y="Score",
            text=df_plot["Score"].apply(lambda x: f"{x:.2f}"),
            range_y=[0,1],
            title="Model Confidence Comparison (normalized to 0–1)"
        )
        fig.update_traces(marker_color=["#2b8cbe", "#f28e2b", "#7fc97f"])
        fig.update_layout(height=420, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- Text highlighting (simple) ----------------
        st.markdown("**Input (with detected highlights)**")
        # color-coded simple highlights for a few keywords (small heuristic)
        def highlight_text(text):
            low = text.lower()
            # simple lists
            positive_words = ["great", "amazing", "love", "happy", "excited", "fantastic", "improved"]
            negative_words = ["awful", "disappointed", "hate", "angry", "frustrating", "crash", "freezing", "error"]
            sarcasm_words = ["yeah right", "sure", "just what i needed", "great job"]

            # escape html
            import html
            out = html.escape(text)

            # highlight sarcasm phrases first
            for kw in sarcasm_words:
                if kw in low:
                    out = out.replace(kw, f'<span style="background:#6a51a3;color:#fff;padding:2px 4px;border-radius:4px">{kw}</span>')

            # positive and negative
            for kw in positive_words:
                if kw in low:
                    out = out.replace(kw, f'<span style="background:#d9f0d3;color:#06470b;padding:1px 3px;border-radius:3px">{kw}</span>')
            for kw in negative_words:
                if kw in low:
                    out = out.replace(kw, f'<span style="background:#fddcdc;color:#600000;padding:1px 3px;border-radius:3px">{kw}</span>')

            return out

        try:
            st.markdown(highlight_text(cleaned_text), unsafe_allow_html=True)
        except Exception:
            st.write(cleaned_text)


        # small raw info line
        st.info(f"HF label: {hf_label.capitalize()} ({hf_score:.2f})  •  VADER: {vader_sentiment} ({vader_score:.3f})  •  Emotion: {emotion_label}")


        # ================================
        # WORD CLOUD
        # ================================
        st.subheader("Word Cloud")

        try:
            wc = WordCloud(width=800, height=400, background_color="white").generate(cleaned_text)
            fig_wc, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
        except Exception as e:
            st.error(f"Word Cloud could not be generated: {e}")

                # ================================
        # NEW: SENTIMENT TIMELINE (Sentence Polarity)
        # ================================
        st.subheader("Sentiment Timeline (Per Sentence)")

        sentences = nltk.sent_tokenize(cleaned_text)
        vader = SentimentIntensityAnalyzer()

        timeline_scores = []
        for s in sentences:
            compound = vader.polarity_scores(s)["compound"]
            timeline_scores.append({"sentence": s, "compound": compound})

        df_timeline = pd.DataFrame(timeline_scores)

        if len(df_timeline) > 1:
            fig_timeline = px.line(
                df_timeline,
                x=df_timeline.index + 1,
                y="compound",
                markers=True,
                title="Sentiment Progression Across Sentences",
            )
            fig_timeline.update_layout(
                xaxis_title="Sentence Number",
                yaxis_title="VADER Compound Score (-1 to 1)",
                height=350,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Enter more than one sentence to see a timeline visualization.")


# ================================
# HISTORY
# ================================
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
