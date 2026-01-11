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
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)




# ================================
# MODELS
# ================================
@st.cache_resource
# function that will create and return the sentiment analysis model from HuggingFace
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
#  Hugging Face (DistilBERT) pretrained model which returns positive, negative or neuitral label + Vader 
# which retruns sentiment polarity score
        model="distilbert-base-uncased-finetuned-sst-2-english" 
        
    )

@st.cache_resource
# function to load the emotion detection model
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions", # RoBERTa trained on Google’s GoEmotions dataset
        return_all_scores=False # return only the top emotion
    )

sentiment_model = load_sentiment_model()
emotion_model = load_emotion_model()

# ================================
# EXTERNAL APIs CONFIG
# ================================

# NewsAPI
NEWS_API_KEY = "c90b27b1743b445eb23cc03e005abdaa"

# SerpAPI (YouTube)
YOUTUBE_API_KEY = "AIzaSyCp0yhPzgOK5VC0pdm1Obd5EnHZq-LCgS0"


# ================================
# DATABASE
# ================================
DB_PATH = "data/app.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# function to initialize db structure 
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

# function to save the results in the db
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
EMOJI_MAP = {   # dictionaryto map each emotion to an emoji
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
# ================================

# function to decide if a sentence is sarcastic or not
def detect_sarcasm(text: str, vader_score: float, emotion_label: str, hf_sentiment: str) -> bool:
    """
    Detect sarcasm using multi-signal fusion instead of single keywords.
    Returns True if sarcasm is likely, False otherwise.
    """

    if not text:
        return False

    sentences = nltk.sent_tokenize(text)

    conflict_count = 0
    negative_emotions = {"disappointment", "anger", "annoyance", "sadness", "disgust", "fear"}
    positive_emotions = {"joy", "love", "admiration", "approval", "optimism", "caring"}

    vader = SentimentIntensityAnalyzer()

    for s in sentences:
        vader_s = vader.polarity_scores(s)["compound"]
        vader_label = "positive" if vader_s > 0.05 else ("negative" if vader_s < -0.05 else "neutral")

        hf_raw = sentiment_model(s, truncation=True, max_length=512)[0]
        hf_label = hf_raw["label"].lower()

        emotion_raw = emotion_model(s, truncation=True, max_length=512)[0]
        emotion = emotion_raw["label"].lower()

        # Conflict patterns
        if hf_label == "positive" and vader_label == "negative" and emotion in negative_emotions:
            conflict_count += 1
        if hf_label == "negative" and vader_label == "positive" and emotion in positive_emotions:
            conflict_count += 1

    return conflict_count >= 2



# ================================
# SENTIMENT FUSION
# ================================

# function that combines Hugging Face sentiment, VADER sentiment score, Emotion detection, Sarcasm detection
# returns either "positive", "negative", "neutral" or "sarcasm"
def fuse_sentiment(text: str) -> str:
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return "neutral"

    vader = SentimentIntensityAnalyzer()
    pos_count = neg_count = neu_count = 0
    conflict_count = 0

    for s in sentences:
        # VADER
        vs = vader.polarity_scores(s)["compound"]
        vs_label = "positive" if vs > 0.05 else ("negative" if vs < -0.05 else "neutral")

        # HF
        hf_label = sentiment_model(s, truncation=True, max_length=512)[0]["label"].lower()

        # Emotion
        emotion_label = emotion_model(s, truncation=True, max_length=512)[0]["label"].lower()
        positive_emotions = {"joy","love","admiration","approval","optimism","caring"}
        negative_emotions = {"anger","annoyance","sadness","fear","disappointment","disgust"}

        if emotion_label in positive_emotions:
            em_sent = "positive"
        elif emotion_label in negative_emotions:
            em_sent = "negative"
        else:
            em_sent = "neutral"

        # Count conflicts (sarcasm or mixed)
        if (hf_label=="positive" and vs_label=="negative" and em_sent=="negative") or \
           (hf_label=="negative" and vs_label=="positive" and em_sent=="positive"):
            conflict_count += 1

        # Voting: emotion > HF > VADER
        if em_sent!="neutral":
            fused = em_sent
        elif hf_label!="neutral":
            fused = hf_label
        else:
            fused = vs_label

        if fused=="positive":
            pos_count += 1
        elif fused=="negative":
            neg_count += 1
        else:
            neu_count += 1

    # Aggregate
    if conflict_count >= 1:
        return "mixed"
    if pos_count > neg_count:
        return "positive"
    if neg_count > pos_count:
        return "negative"
    return "neutral"


# ================================
# Clear text handler
# ================================
def clear_text():
    st.session_state["user_input"] = ""

# ================================
# Helper: sentiment color badge (HTML)
# ================================

#function to map each sentiment to a color
def sentiment_badge_html(label: str) -> str:
    color_map = { 
        "positive": "#1a9850",  # green
        "negative": "#d73027",  # red
        "neutral": "#999999",   # grey
        "sarcasm": "#6a51a3"    # purple
    }
    color = color_map.get(label, "#333333")
    return f'<div style="display:inline-block;padding:8px 14px;border-radius:12px;background:{color};color:#ffffff;font-weight:600">{label.capitalize()}</div>'

def fetch_news_articles(query):
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "pageSize": 5, "apiKey": NEWS_API_KEY}
    response = requests.get(url, params=params).json()
    articles = response.get("articles", [])
    return [a["title"] + ". " + (a["description"] or "") for a in articles]

def fetch_youtube_comments(query, max_videos=3, max_comments=10):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"

    comments = []

    # 1️⃣ Search videos
    search_params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_videos,
        "key": YOUTUBE_API_KEY
    }

    search_resp = requests.get(search_url, params=search_params).json()
    video_ids = [
        item["id"]["videoId"]
        for item in search_resp.get("items", [])
        if "videoId" in item["id"]
    ]

    if not video_ids:
        return []

    # 2️⃣ Fetch comments
    for vid in video_ids:
        comment_params = {
            "part": "snippet",
            "videoId": vid,
            "maxResults": max_comments,
            "textFormat": "plainText",
            "key": YOUTUBE_API_KEY
        }

        comment_resp = requests.get(comments_url, params=comment_params).json()

        for item in comment_resp.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(text)

    return comments


def sentiment_to_numeric(label):
    mapping = {"positive": 1, "neutral": 0, "negative": -1, "sarcasm": -0.5}
    return mapping.get(label.lower(), 0)

def plot_sentiment_trend(texts):
    trend_data = []

    for i, text in enumerate(texts):
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        if not cleaned_text:
            continue

        # VADER
        vader_score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)["compound"]
        vader_sent = "positive" if vader_score > 0.05 else ("negative" if vader_score < -0.05 else "neutral")

        # HF
        hf_raw = sentiment_model(cleaned_text, truncation=True, max_length=512)[0]
        hf_label = hf_raw["label"].lower()

        # Emotion
        emotion_raw = emotion_model(cleaned_text, truncation=True, max_length=512)[0]
        emotion_label = emotion_raw["label"].lower()

        # Fusion
        fused = fuse_sentiment(cleaned_text)


        trend_data.append({
            "Post": i+1,
            "HF": sentiment_to_numeric(hf_label),
            "VADER": sentiment_to_numeric(vader_sent),
            "Fused": sentiment_to_numeric(fused)
        })

    df_trend = pd.DataFrame(trend_data)
    if df_trend.empty:
        st.info("No posts to analyze for trend.")
        return

    fig = px.line(df_trend, x="Post", y=["HF", "VADER", "Fused"], markers=True,
                  title="Sentiment Trend Across Multiple Posts")
    fig.update_layout(yaxis_title="Sentiment (numeric)",
                      yaxis=dict(tickvals=[-1, -0.5, 0, 1], ticktext=["Negative","Sarcasm","Neutral","Positive"]))
    st.plotly_chart(fig, use_container_width=True)

# ================================
# KEYWORD-LEVEL SENTIMENT
# ================================
from collections import Counter
import nltk
nltk.download('punkt', quiet=True)

def keyword_sentiment(text, top_n=10):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    vader = SentimentIntensityAnalyzer()

    words = nltk.word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha() and w.lower() not in stop_words]
    top_words = [w for w,_ in Counter(cleaned).most_common(top_n)]
    sentences = nltk.sent_tokenize(text)

    rows = []
    for kw in top_words:
        kw_scores = []
        for s in sentences:
            if kw.lower() in s.lower():
                kw_scores.append(vader.polarity_scores(s)["compound"])
        if not kw_scores:
            final_kw_sent = "neutral"
        else:
            avg = sum(kw_scores)/len(kw_scores)
            if avg > 0.05:
                final_kw_sent = "positive"
            elif avg < -0.05:
                final_kw_sent = "negative"
            else:
                final_kw_sent = "neutral"
        rows.append({"Keyword": kw, "Sentiment": final_kw_sent})
    return pd.DataFrame(rows)


def plot_keyword_sentiment(df_keywords):
    if df_keywords.empty:
        return
    color_map = {"positive":"#1a9850","negative":"#d73027","neutral":"#999999","mixed":"#f0ad4e"}
    
    fig = px.bar(
        df_keywords,
        x="Keyword",
        y=[1]*len(df_keywords),  # Dummy numeric value to make Plotly happy
        color="Sentiment",
        color_discrete_map=color_map,
        text="Sentiment",
        title="Keyword-Level Sentiment Map"
    )
    fig.update_layout(
        yaxis_title="Sentiment",
        yaxis=dict(showticklabels=False)  # Hide the dummy y-axis
    )
    st.plotly_chart(fig, use_container_width=True)

def extract_themes(text, top_n=5):
    df_keywords = keyword_sentiment(text, top_n=top_n)
    themes = {}
    for _, row in df_keywords.iterrows():
        themes[row["Keyword"]] = row["Sentiment"]  # <-- Fixed column name

    # Display
    st.subheader("🧩 Extracted Themes & Sentiments")
    for k, v in themes.items():
        st.markdown(f"**{k.capitalize()}** → {v.capitalize()}")



def cross_platform_comparison(keyword):
    """
    Performs cross-platform sentiment analysis for a keyword across:
    - Manual Input
    - News articles
    - YouTube comments
    Uses cached multi-model sentiment analysis and maps to numeric for visualization.
    """
    sources = ["Manual Input", "News", "YouTube"]
    results = []

    # ---------------- Manual Input ----------------
    results.append({
        "source": "Manual Input",
        "avg_sentiment": sentiment_to_numeric(analyze_text_cached(keyword))
    })

    # ---------------- News ----------------
    news_texts = fetch_news_articles(keyword)
    if news_texts:
        results.append({
            "source": "News",
            "avg_sentiment": sum([sentiment_to_numeric(analyze_text_cached(t)) for t in news_texts]) / len(news_texts)
        })
    else:
        results.append({"source": "News", "avg_sentiment": 0})

    # ---------------- YouTube ----------------
    yt_texts = fetch_youtube_comments(keyword)
    if yt_texts:
        results.append({
            "source": "YouTube",
            "avg_sentiment": sum([sentiment_to_numeric(analyze_text_cached(t)) for t in yt_texts]) / len(yt_texts)
        })
    else:
        results.append({"source": "YouTube", "avg_sentiment": 0})

    # ---------------- Display as Plotly bar ----------------
    df = pd.DataFrame(results)
    fig = px.bar(
        df,
        x="source",
        y="avg_sentiment",
        color="avg_sentiment",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        range_color=[-1, 1],
        title=f"🌐 Cross-Platform Average Sentiment for '{keyword}'"
    )
    fig.update_layout(
        yaxis_title="Average Sentiment (-1 Negative → 1 Positive)"
    )
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def analyze_text_cached(text):
    """
    Analyze text using HuggingFace sentiment, VADER, and emotion model.
    Returns the fused sentiment.
    """

    # 1️⃣ VADER sentiment
    vader = SentimentIntensityAnalyzer()
    vader_score = vader.polarity_scores(text)["compound"]
    vader_sent = "positive" if vader_score > 0.05 else ("negative" if vader_score < -0.05 else "neutral")

    # 2️⃣ HuggingFace sentiment
    hf_raw = sentiment_model(text, truncation=True, max_length=512)[0]
    hf_label = hf_raw["label"].lower()
    hf_score = float(hf_raw.get("score", 1.0))

    # 3️⃣ Emotion detection
    emotion_raw = emotion_model(text, truncation=True, max_length=512)[0]
    emotion_label = emotion_raw["label"].lower()
    emotion_score = float(emotion_raw.get("score", 1.0))

    # 4️⃣ Fuse sentiment correctly using actual VADER label
    fused = fuse_sentiment(text)


    return fused




# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="Sentiment & Emotion Analyzer", layout="centered")
st.title("Sentiment & Emotion Analyzer")
st.write("Enter text below and click **Analyze**. The dashboard shows model comparison and visualizations.")

data_source = st.selectbox(
    "Select data source",
    ["Manual Input", "News", "YouTube"]

)

# session state init
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if data_source == "Manual Input":
    user_input = st.text_area(
        "Enter your text here",
        value=st.session_state["user_input"],
        key="user_input",
        height=140
    )
else:
    user_input = st.text_input(
        "Enter keyword for data retrieval (News/YouTube)",
        value=st.session_state["user_input"],
        key="user_input"
    )



col1, col2, col3 = st.columns([1,1,1])
with col1:
    analyze_clicked = st.button("Analyze")
with col2:
    st.button("Clear Text", on_click=clear_text)
with col3:
    load_history = st.button("Load Analysis History")



# ================================
# Main analysis
# ================================
if analyze_clicked:
    if data_source == "Manual Input":
        input_text = (st.session_state["user_input"] or "").strip()

    elif data_source == "News":
        if not user_input.strip():
            st.warning("Please enter a keyword to fetch news articles.")
            st.stop()
        texts = fetch_news_articles(user_input.strip())
        input_text = " ".join(texts)

        # ---------------- Add Trend Chart ----------------
        plot_sentiment_trend(texts)

    elif data_source == "YouTube":
        if not user_input.strip():
            st.warning("Please enter a keyword to fetch YouTube comments.")
            st.stop()
        texts = fetch_youtube_comments(user_input.strip())

        if not texts:
            st.warning("Could not fetch YouTube comments for this keyword. Analyzing video titles instead.")
            input_text = user_input
        else:
            input_text = " ".join(texts)

            # ---------------- Add Trend Chart ----------------
            plot_sentiment_trend(texts)
        
    else:
        st.warning("Unknown data source selected.")
        st.stop()

    if not input_text:
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_text = re.sub(r'\s+', ' ', input_text).strip()

          # ---------------- Cross-Platform Comparison ----------------
        st.subheader("🌐 Cross-Platform Sentiment Comparison")
        cross_platform_comparison(user_input.strip())

        # VADER model for calculating sentiment score
        vader = SentimentIntensityAnalyzer()
        vader_score = vader.polarity_scores(cleaned_text)["compound"]
        if vader_score >= 0.05:
            vader_sentiment = "positive"
        elif vader_score <= -0.05:
            vader_sentiment = "negative"
        else:
            vader_sentiment = "neutral"

        # HF sentiment to return label and confidence score 
        hf_raw = sentiment_model(
    cleaned_text,
    truncation=True,
    max_length=512
)[0]

        hf_label = hf_raw["label"].lower()
        hf_score = float(hf_raw["score"])

        # Emotion model to detect emotion and asign emoji from emoji map
        # HuggingFace pipelines accept a 'truncation' argument for long sequences
        emotion_raw = emotion_model(cleaned_text, truncation=True, max_length=512)[0]
        emotion_label = emotion_raw["label"].lower()
        emotion_score = float(emotion_raw.get("score", 1.0))
        emoji = EMOJI_MAP.get(emotion_label, "🙂")

        # Fusion
        final_sentiment = fuse_sentiment(cleaned_text)


        # ---------------- Dashboard / Metrics ----------------
        st.subheader("Multi-Model Comparison Dashboard")
# dashboard to display HF sentiment label + confidence score, Vader polarity score + label, emotion + emoji
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("**HuggingFace (HF) Sentiment**")
            st.metric(label=f"HF: {hf_label.capitalize()}", value=f"{hf_score*100:.1f}%", delta=None)
        with m2:
            vader_desc = "Positive" if vader_score > 0.05 else ("Negative" if vader_score < -0.05 else "Neutral")
            st.markdown("**VADER Polarity**")
            st.metric(label=f"VADER: {vader_desc}", value=f"{vader_score:.3f}", delta=None)
        with m3:
            st.markdown("**Emotion Detector**")
            st.metric(label=f"{emotion_label.capitalize()}", value=f"{emoji}", delta=None)

        # Final sentiment badge
        st.markdown("**Final fused sentiment**")
        st.markdown(sentiment_badge_html(final_sentiment), unsafe_allow_html=True)

        st.markdown(f"**Data source:** {data_source}")

        # ---------------- Keyword-level sentiment ----------------
        st.subheader("🔑 Keyword-Level Sentiment")
        df_keywords = keyword_sentiment(cleaned_text, top_n=10)
        st.table(df_keywords)

        plot_keyword_sentiment(df_keywords)

        extract_themes(cleaned_text, top_n=8)

        # ---------------- User-friendly summary ----------------

# displays a text based on final sentiment 
        summary_text = ""
        if final_sentiment == "positive":
            summary_text += "Overall, the text feels **positive / happy**. "
        elif final_sentiment == "negative":
            summary_text += "Overall, the text feels **negative / unhappy**. "
        elif final_sentiment == "neutral":
            summary_text += "Overall, the text feels **neutral / balanced**. "
        elif final_sentiment == "sarcasm":
            summary_text += "Overall, the text appears **sarcastic**. "

        summary_text += f"The dominant emotion detected is **{emotion_label.capitalize()}** {emoji}. "

        if final_sentiment == "sarcasm":
            summary_text += "This may be because the text contains conflicting cues, such as positive words but negative tone or emotions, indicating sarcasm."

        st.subheader("📝 Text Summary")
        st.markdown(summary_text)

        # Save result
        save_result(cleaned_text, final_sentiment, hf_score, emotion_label, emoji, "HF + VADER + Fusion")

        # Conflict indicator
        if (hf_label == "positive" and vader_score < -0.25) or (hf_label == "negative" and vader_score > 0.25):
            st.warning("Strong disagreement between HF and VADER detected — fusion logic applied.")
        else:
            st.success("HF and VADER are in agreement or only mild disagreement.")

        # ---------------- Plotly comparison chart ----------------
        vader_norm = (vader_score + 1.0) / 2.0
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

        # ---------------- Text highlighting ----------------
        st.markdown("**Input (with detected highlights)**")
        def highlight_text(text):
            low = text.lower()
            positive_words = ["great", "amazing", "love",  "like", "happy", "excited", "fantastic", "improved"]
            negative_words = ["awful", "disappointed", "hate", "angry", "frustrating", "crash", "freezing", "error"]
            sarcasm_words = ["Yeah right", "sure", "just what i needed", "great job"]

            import html
            out = html.escape(text)

            for kw in sarcasm_words:
                if kw in low:
                    out = out.replace(kw, f'<span style="background:#6a51a3;color:#fff;padding:2px 4px;border-radius:4px">{kw}</span>')
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

        st.info(f"HF label: {hf_label.capitalize()} ({hf_score:.2f})  •  VADER: {vader_sentiment} ({vader_score:.3f})  •  Emotion: {emotion_label}")

        # ---------------- Word Cloud  for mostv frequent words in the text----------------
        st.subheader("Word Cloud")
        try:
            wc = WordCloud(width=800, height=400, background_color="white").generate(cleaned_text)
            fig_wc, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
        except Exception as e:
            st.error(f"Word Cloud could not be generated: {e}")

        # ---------------- Sentiment Timeline ----------------
        # splits text into sentences to show how sentiment changes across text
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
# loads previous analyses from db
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
        df = pd.DataFrame(rows, columns=["ID", "Text", "Final Sentiment", "Model Confidence", "Emotion", "Emoji", "Created At"])
        
        def color_sentiment(val):
            colors = {
                "positive": "#d4f4dd",   # light green
                "negative": "#f8d7da",   # light red
                "neutral": "#f0f0f0",    # light grey
                "sarcasm": "#e0d4f4"     # light purple
            }
            return f'background-color: {colors.get(val.lower(), "#ffffff")}'
        
        def confidence_gradient(val):
            from matplotlib import colors as mcolors
            import matplotlib
            cmap = matplotlib.cm.get_cmap('RdYlGn')
            norm_val = min(max(val, 0), 1)
            rgba = cmap(norm_val)
            hex_color = mcolors.to_hex(rgba)
            return f'background-color: {hex_color}; color:#000'

        styled_df = df.style \
            .applymap(color_sentiment, subset=['Final Sentiment']) \
            .applymap(confidence_gradient, subset=['Model Confidence']) \
            .set_properties(subset=['Emoji'], **{'font-size': '18px'}) \
            .set_properties(subset=['Emotion'], **{'font-weight': '600'}) \
            .set_properties(subset=['Text'], **{'max-width': '350px', 'text-overflow': 'ellipsis', 'white-space': 'nowrap', 'overflow': 'hidden'})

        st.subheader("📊 Analysis History")
        st.dataframe(styled_df, height=400)
