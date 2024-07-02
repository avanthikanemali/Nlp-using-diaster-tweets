import time
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import tweepy
from tweepy import TweepyException, TooManyRequests
from datetime import datetime, timedelta, timezone

# Load the pre-trained TF-IDF vectorizer and Random Forest model
tfidf_vectorizer = joblib.load(r'C:\Users\ADMIN\Desktop\UI component\tfidf_vectorizer.pkl')
rf_best = joblib.load(r'C:\Users\ADMIN\Desktop\UI component\RandomForest_model.pkl')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Twitter API credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAAO9puQEAAAAAibBDDQEa%2BOYP%2BI%2BkQNdGzbfgOQQ%3DSryMUwRRPymuUcj0pA44K7TkHdRKXPE8LQrBeXkrQsIKgUHWDd"

# Authenticate with the Twitter API using tweepy.Client
client = tweepy.Client(bearer_token=bearer_token)

# Load the English stopwords
stopwords = set(stopwords.words('english'))

# Function to clean the text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')
    text = "".join([i for i in text if i not in string.punctuation])
    words = word_tokenize(text)
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub(r"\s[\s]+", " ", text).strip()
    return text

# Function to predict if the tweet is a disaster or not
def predict_new_tweet(new_tweet):
    cleaned_tweet = clean_text(new_tweet)
    tweet_tfidf = tfidf_vectorizer.transform([cleaned_tweet])
    tweet_score = rf_best.predict_proba(tweet_tfidf)[:, 1][0]
    threshold = 0.4
    prediction = 1 if tweet_score >= threshold else 0
    return prediction, tweet_score

# Function to fetch the tweet text from the URL with exponential backoff strategy
def fetch_tweet_text(url):
    try:
        tweet_id = url.split('/')[-1].split('?')[0] if 'status' in url else url.split('/')[-1]
        max_retries = 5
        retry_count = 0
        backoff_time = 60

        while retry_count < max_retries:
            try:
                tweet = client.get_tweet(tweet_id, tweet_fields=["text"])
                if tweet and tweet.data:
                    return tweet.data['text']
                else:
                    raise ValueError("Tweet data is not available.")
            except TooManyRequests:
                retry_count += 1
                st.warning(f"Rate limit reached. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2
            except TweepyException as e:
                st.error(f"Tweepy Exception: {e}")
                break
            except Exception as e:
                st.error(f"Error fetching tweet: {e}")
                break
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    return None

# Function to fetch tweets within the specified time range and with additional parameters
def fetch_disaster_tweets(time_range):
    end_time = datetime.now(timezone.utc)
    start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0) if time_range == "Today" else end_time - timedelta(hours=5)
    end_time_adjusted = end_time - timedelta(seconds=10)
    query = "disaster -is:retweet"
    max_results = 100

    try:
        tweets = client.search_recent_tweets(query=query, start_time=start_time.isoformat(), end_time=end_time_adjusted.isoformat(), tweet_fields=["created_at", "text", "author_id", "geo"], max_results=max_results)
        return tweets.data
    except TweepyException as e:
        st.error(f"Error fetching tweets: {e}")
        return None

def set_background_and_styles():
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url('https://th.bing.com/th/id/OIP.rz7oNAp4ZZdzvy3s3uWsagAAAA?rs=1&pid=ImgDetMain');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: black; /* Changed background color to black */
    }
    .title-text {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-bottom: 2px solid white; /* Underline instead of border */
        margin-bottom: 1rem;
        color: white; /* Changed text color to white */
        font-family: 'Arial', sans-serif; /* Changed font family */
        text-shadow: 3px 3px 5px rgba(0,0,0,0.5); /* 3D Text effect */
    }
    .description {
        display: none; /* Hide the description text */
    }
    .stRadio > div, .stSelectbox > div {
        color: white !important;
        font-size: 1.5em;
        background: black;
        padding: 10px;
        border-radius: 10px;
        border: 2px solid white; /* Changed border color to white */
    }
    .stButton > button {
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 1.5em;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 5px;
        background-color: #1E90FF; /* Changed background color to blue */
        color: white; /* Changed text color to white */
        font-weight: bold;
    }
    .stButton > button:hover {
        border: 2px solid white; /* Changed border color to white */
    }
    .output-text-disaster, .output-text-not-disaster {
        font-size: 2em;
        font-weight: bold;
        background: linear-gradient(90deg, #43C6AC 0%, #191654 100%);
        text-align: center;
        border: 2px solid white; /* Changed border color to white */
        padding: 10px;
        border-radius: 10px;
        margin-top: 2rem;
        margin-bottom: 2rem;
        color: red;
    }
    .output-text-not-disaster {
        color: white; /* Changed text color to white */
    }
    .option-text, .input-text, .time-range-text {
        font-size: 1.5em;
        color: white; /* Changed text color to white */
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    set_background_and_styles()
    st.markdown("<h1 class='title-text'><span style='font-family: Arial, sans-serif;'>Twitter Disaster Analysis</span></h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<p class='input-text'>Enter a Tweet URL or text:</p>", unsafe_allow_html=True)
        user_input = st.text_input("")

        if st.button("Fetch & Classify", key="classify_button"):
            if user_input:
                if 'http' in user_input:
                    tweet_text = fetch_tweet_text(user_input)
                    if tweet_text:
                        prediction_and_score = predict_new_tweet(tweet_text)
                    else:
                        st.warning("Invalid tweet URL. Please enter a valid URL.")
                else:
                    tweet_text = user_input
                    prediction_and_score = predict_new_tweet(tweet_text)

                prediction, score = prediction_and_score
                st.markdown(f"<p style='font-size: 2em; color: white;'>{tweet_text}</p>", unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown(f"<p class='output-text-disaster'><span>Prediction: Disaster</span></p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='output-text-not-disaster'><span>Prediction: Not a Disaster</span></p>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a tweet URL or text.")

    with col2:
        st.markdown("<p class='time-range-text'>Time Range:</p>", unsafe_allow_html=True)
        time_range = st.selectbox("", ["Today", "Last 5 Hours"])

        if st.button("Fetch Tweets", key="fetch_tweets_button"):
            tweets = fetch_disaster_tweets(time_range)
            if tweets:
                for tweet in tweets:
                    st.markdown(f"<p style='font-size: 1.5em; color: white;'>{tweet['text']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 1em; color: gray;'>Created At: {tweet['created_at']} | Author ID: {tweet['author_id']}</p>", unsafe_allow_html=True)
            else:
                st.warning("No tweets found for the selected time range.")

if __name__ == "__main__":
    main()
