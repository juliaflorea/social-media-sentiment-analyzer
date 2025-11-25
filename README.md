 Social Media Sentiment Analyzer

A lightweight Python web application for real-time sentiment and emotion analysis of user-provided text.
The app identifies:

Sentiment → Positive, Negative, or Neutral

Sentiment Score → from −1.0 (very negative) to +1.0 (very positive)

Emotion → Love, Joy, Anger, Sadness, Fear, Disgust (basic set)

Built using Python, Streamlit, VADER/TextBlob, and an optional emotion lexicon.

 Features

-Real-time sentiment analysis

-Numerical sentiment score

-Emotion detection using a simple lexicon or ML model

-Clean, interactive Streamlit UI

-Emoji-based visualization

-Optional local storage via SQLite or JSON

-Runs locally — no external services required

 Technologies Used
Component	Technology
Programming Language	Python 3.10
UI Framework	Streamlit
Sentiment Analysis	VADER / TextBlob
Emotion Analysis	Custom lexicon or HuggingFace model
Optional Storage	SQLite / JSON
Development	Visual Studio Code

Architecture Overview

User Input
    ↓
Sentiment Analyzer (label + score)
    ↓
Emotion Detector (lexicon/model)
    ↓
Visualization (UI + emojis + colors)
    ↓
(Optional) Local Storage (SQLite/JSON)

Installation
1. Clone the repository
git clone https://github.com/juliaflorea/social-media-sentiment-analyzer.git
cd social-media-sentiment-analyzer
2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
3. Install dependencies
pip install streamlit textblob vaderSentiment numpy
Run the Application
streamlit run app.py
The app will open automatically at:
http://localhost:8501

Project Structure
sentiment-analyzer/
│
├── app.py
├── emotion_lexicon.json
├── data/
│   └── app.db (optional SQLite database)
├── venv/
├── LICENSE
└── README.md

Examples

Input:

I absolutely love this new design!


Output:

Sentiment: Positive

Score: 0.82

Emotion: Love 💖

Input:

I hate waiting so long...


Output:

Sentiment: Negative

Score: −0.76

Emotion: Anger 😠