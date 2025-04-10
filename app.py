# Run it using: streamlit run app.py
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import hashlib
import json
import emoji
from datetime import datetime



USER_DB_PATH = "TwitterData/users.json"

# ---------- Helper Functions ----------
def load_users():
    if not os.path.exists(USER_DB_PATH):
        return {"admin": hashlib.sha256("admin123".encode()).hexdigest()}  # Default admin
    with open(USER_DB_PATH, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB_PATH, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------- Login Form (Main Area) ----------
def login():
    st.title("🔐 Login Page")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            users = load_users()
            if username in users and users[username] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.login_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
                st.markdown(f"##### 🔐 Not Registered, whatsapp >> +254700921906 for login credentials")

# ---------- Admin User Registration (Sidebar) ----------
def add_user_ui():
    st.sidebar.subheader("👤 Add New User (Admin Only)")
    with st.sidebar.form("add_user_form", clear_on_submit=True):
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        add_user_submit = st.form_submit_button("Add User")

        if add_user_submit:
            users = load_users()
            if new_user in users:
                st.sidebar.warning("User already exists.")
            else:
                users[new_user] = hash_password(new_pass)
                save_users(users)
                st.sidebar.success(f"User '{new_user}' added!")

# ---------- Login Gate ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# Show login page if not logged in
if not st.session_state.logged_in:
    login()
    st.stop()

# ---------- Show Sidebar Admin UI ----------
if st.session_state.username == "admin":
    add_user_ui()

# ---------- Welcome Message ----------
st.info(f"👋 Welcome, **{st.session_state.username}**! Logged in at: {st.session_state.login_time}")

# 👇 You can place the rest of your dashboard code here 👇
st.markdown(f"### 📊 **{st.session_state.username}**, >> Your Main Dashboard Goes Here")







# Load labeled tweet data
try:
    df = pd.read_csv("TwitterData/labeled_model.csv")
except FileNotFoundError:
    st.error("Error: 'TwitterData/labeled_model.csv' file not found. Please check the file path.")
    st.stop()


#side bar logout
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()



# Sidebar: Search bar
query = st.sidebar.text_input("🔎 Search tweets by keyword")

# Dashboard Title and Description
st.title("Tweet Category Dashboard")
st.markdown("""
Analyze tweets classified into **Social**, **Political**, **Economy**, or **Technology** categories.
This dashboard provides insights into sentiment distribution and allows filtering tweets dynamically.
""")

# Show Raw Data Option
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Tweet Data")
    st.dataframe(df)

# Distribution of Tweet Categories
st.subheader("Category Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Labels", palette="Set2", ax=ax)
ax.set_title("Distribution of Tweet Labels")
ax.set_xlabel("Categories")
ax.set_ylabel("Count")
st.pyplot(fig)

# Layout for Category Filter and Count Plot
st.subheader("Analyze Category and Sentiments")
col1, col2 = st.columns(2)

# Category Filter in the First Column
with col1:
    selected_category = st.selectbox("Select a category:", df["Labels"].unique())
    filtered = df[df["Labels"] == selected_category]
    st.write(f"Showing **{len(filtered)} tweets** labeled as **{selected_category}**:")
    st.dataframe(filtered[["text", "sentiment", "Labels"]].reset_index(drop=True))

# Count Plot for the Filtered Category in the Second Column
with col2:
    st.write(f"Sentiment Distribution for **{selected_category}**")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered, x="sentiment", palette="Set2", ax=ax)
    ax.set_title(f"Sentiment Distribution for '{selected_category}'")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Search Results
if query:
    st.sidebar.subheader("Search Results")
    results = df[df["text"].str.contains(query, case=False, na=False)]
    st.sidebar.write(f"Found **{len(results)} tweets** containing '{query}':")
    st.sidebar.dataframe(results[["text", "sentiment", "Labels"]])













# ---------- Rule-based Labeling Functions (same as your training logic) ---------

# ---------- Helper Functions for Sarcasm and Emoji Analysis ----------

def detect_sarcasm(text):
    """
    A simple heuristic for sarcasm detection based on known sarcastic phrases and emojis.
    """
    text_lower = text.lower()
    # A list of phrases/emojis that might hint at sarcasm.
    sarcasm_indicators = [
        "yeah right", "as if", "oh great", "sure, because", "i'm so thrilled",
        "not really", "lol, right", "whatever", "smh", "🙄", "😒"
    ]
    for indicator in sarcasm_indicators:
        if indicator in text_lower:
            return True
    return False

def analyze_emojis(text):
    """
    Analyze emojis in the text by converting them to their textual representation.
    Returns "positive", "negative", or None if no clear signal.
    """
    demojized = emoji.demojize(text)
    # Define groups of keywords representing positive and negative emojis
    positive_emojis = ["face_with_tears_of_joy", "smiling_face_with_heart_eyes", "grinning_face"]
    negative_emojis = ["pensive_face", "confounded_face", "disappointed_face", "frowning_face"]
    
    pos_count = sum(demojized.count(name) for name in positive_emojis)
    neg_count = sum(demojized.count(name) for name in negative_emojis)
    
    if pos_count > neg_count and pos_count > 0:
        return "positive"
    elif neg_count > pos_count and neg_count > 0:
        return "negative"
    else:
        return None

# ---------- Rule-based Labeling Functions (same as your training logic) ----------
def fallback_rotator():
    labels = ["Social", "Political", "Economy", "Technology"]
    while True:
        for label in labels:
            yield label

fallback_gen = fallback_rotator()

def label_data(text):
    text_lower = str(text).lower()
    keywords = {
        "Political": [
            "government", "vote", "voting", "policy", "war", "army", "military",
            "president", "prime minister", "minister", "campaign", "election", "laws",
            "rights", "parliament", "senator", "bill", "democrat", "republican",
            "congress", "united nations", "political", "protest", "constitution",
            "politics", "legislation", "administration", "lobby", "diplomacy", "referendum",
            "ideology", "politicians", "cabinet", "political party", "governance", "veto",
            "public office"
        ],
        "Economy": [
            "money", "work", "job", "jobs", "salary", "bills", "tips", "tax", "taxes",
            "pay", "payment", "broke", "unemployed", "layoff", "budget", "shopping",
            "expensive", "cheap", "credit", "income", "market", "sell", "sale",
            "promotion", "business", "rent", "finance", "investment", "stocks", "trade",
            "recession", "inflation", "economic", "commerce", "entrepreneur", "merger",
            "acquisition", "fiscal", "monetary", "GDP", "economics", "central bank", "debt",
            "financial", "market trends"
        ],
        "Social": [
            "family", "friends", "marriage", "relationship", "mother", "mom", "mum",
            "father", "dad", "children", "kids", "school", "college", "graduation",
            "hospital", "community", "love", "hate", "emotion", "cry", "tears", "miss you",
            "birthday", "party", "church", "religion", "culture", "funeral", "wedding",
            "social media", "twitter", "facebook", "myspace", "bullying", "sad", "mental health",
            "alone", "lonely", "hangout", "society", "lifestyle", "dating", "entertainment",
            "public opinion", "social network", "local event", "social justice", "neighbor",
            "family values", "interpersonal"
        ],
        "Technology": [
            "computer", "internet", "app", "application", "software", "hardware", "ipod",
            "phone", "iphone", "android", "twitter", "facebook", "myspace", "google",
            "windows", "win7", "vista", "website", "email", "text", "sms", "camera",
            "download", "upload", "update", "beta", "code", "script", "programming",
            "bug", "reboot", "wifi", "network", "game", "gaming", "console", "ps3", "wii",
            "tech", "device", "charger", "screen", "battery", "patch",
            "innovation", "iot", "blockchain", "crypto", "artificial intelligence", "machine learning",
            "robotics", "gadgets", "cybersecurity", "virtual reality", "augmented reality",
            "cloud computing", "big data", "analytics", "5g", "drone"
        ]
    }

    # Initialize a score dictionary.
    scores = {category: 0 for category in keywords}
    
    # Count occurrences of each category's keywords in the text.
    for category, word_list in keywords.items():
        for word in word_list:
            scores[category] += text_lower.count(word)
    
    best_category = max(scores, key=scores.get)
    
    if scores[best_category] > 0:
        return best_category
    else:
        return next(fallback_gen)

# ---------- Classification Section ----------
st.subheader("Classify Your Own Tweet from Trained Data")

# Train models once and cache (for sentiment prediction only)
@st.cache_resource
def train_models():
    clean_df = df.dropna(subset=['text', 'sentiment', 'Labels'])
    sentiment_model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    sentiment_model.fit(clean_df['text'], clean_df['sentiment'])
    return sentiment_model

sentiment_model = train_models()

# --- Create/Load user data CSV ---
user_data_path = "TwitterData/users_data.csv"
os.makedirs("TwitterData", exist_ok=True)

if not os.path.exists(user_data_path):
    pd.DataFrame(columns=["text", "sentiment", "Label"]).to_csv(user_data_path, index=False)
    
user_df = pd.read_csv(user_data_path)

# --- User Input Form ---
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_area("Enter a tweet to classify:")
    submitted = st.form_submit_button("Submit")

if submitted and user_input.strip():
    # Use the sentiment model to predict sentiment
    predicted_sentiment = sentiment_model.predict([user_input])[0]
    # Use rule-based function for label based on keywords
    predicted_label = label_data(user_input)
    
    # Adjust sentiment if sarcasm is detected
    if detect_sarcasm(user_input) and predicted_sentiment.lower() == "neutral":
        predicted_sentiment = "negative"
    
    # Adjust sentiment based on emoji analysis, if available
    emoji_sentiment = analyze_emojis(user_input)
    if emoji_sentiment is not None:
        predicted_sentiment = emoji_sentiment

    # Create new entry as DataFrame
    new_entry = pd.DataFrame([{
        "text": user_input,
        "sentiment": predicted_sentiment,
        "Label": predicted_label
    }])
    user_df = pd.concat([user_df, new_entry], ignore_index=True)
    user_df.to_csv(user_data_path, index=False)
    st.success("Tweet classified and saved!")

# --- Display Results (Most Recent First) ---
if not user_df.empty:
    st.markdown("### 🧾 Classification Result (All Submitted Tweets)")
    st.dataframe(user_df.iloc[::-1].reset_index(drop=True))
    st.markdown("### 📊 Tweet Distributions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sentiment Count**")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=user_df, x="sentiment", palette="Set2", ax=ax1)
        ax1.set_title("Sentiment Distribution")
        st.pyplot(fig1)
    with col2:
        st.markdown("**Label Count**")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=user_df, x="Label", palette="Set3", ax=ax2)
        ax2.set_title("Label (Category) Distribution")
        st.pyplot(fig2)
