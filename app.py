# Run it using: streamlit run app.py
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load labeled tweet data
try:
    df = pd.read_csv("TwitterData/labeled_model.csv")
except FileNotFoundError:
    st.error("Error: 'TwitterData/labeled_model.csv' file not found. Please check the file path.")
    st.stop()

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








# --- Classify User Input Based on Trained Data ---

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.subheader("Classify Your Own Tweet from Trained Data")

# Create and train models for Sentiment and Labels
@st.cache_resource
def train_models():
    # Drop rows with missing values in required columns
    clean_df = df.dropna(subset=['text', 'sentiment', 'Labels'])

    # Vectorizer and classifier for sentiment
    sentiment_model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    sentiment_model.fit(clean_df['text'], clean_df['sentiment'])

    # Vectorizer and classifier for labels
    label_model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    label_model.fit(clean_df['text'], clean_df['Labels'])

    return sentiment_model, label_model


sentiment_model, label_model = train_models()

# User input
user_input = st.text_area("Enter a tweet to classify:")

# If user entered text, predict sentiment and label
if user_input.strip():
    predicted_sentiment = sentiment_model.predict([user_input])[0]
    predicted_label = label_model.predict([user_input])[0]

    # Display results in a single-row table
    st.markdown("### 🧾 Classification Result")
    result_df = pd.DataFrame([{
        "text": user_input,
        "sentiment": predicted_sentiment,
        "Label": predicted_label
    }])
    st.dataframe(result_df)
