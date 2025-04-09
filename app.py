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








from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

st.subheader("Classify Your Own Tweet from Trained Data")

# Train models once and cache
@st.cache_resource
def train_models():
    clean_df = df.dropna(subset=['text', 'sentiment', 'Labels'])

    sentiment_model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    sentiment_model.fit(clean_df['text'], clean_df['sentiment'])

    label_model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression())
    ])
    label_model.fit(clean_df['text'], clean_df['Labels'])

    return sentiment_model, label_model

sentiment_model, label_model = train_models()

# --- Create/Load user data CSV ---
user_data_path = "TwitterData/users_data.csv"
os.makedirs("TwitterData", exist_ok=True)

# Initialize if file doesn't exist
if not os.path.exists(user_data_path):
    pd.DataFrame(columns=["text", "sentiment", "Label"]).to_csv(user_data_path, index=False)

# Load existing data
user_df = pd.read_csv(user_data_path)

# --- User Input Form ---
with st.form("user_input_form", clear_on_submit=True):
    user_input = st.text_area("Enter a tweet to classify:")
    submitted = st.form_submit_button("Submit")

if submitted and user_input.strip():
    predicted_sentiment = sentiment_model.predict([user_input])[0]
    predicted_label = label_model.predict([user_input])[0]

    # Add new entry and save
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

    # Side-by-side charts
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

