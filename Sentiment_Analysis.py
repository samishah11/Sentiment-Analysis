import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download VADER lexicon (only needed once)
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit UI
st.title("📝 Sentiment Analysis App 🚀")
st.write("Analyze the sentiment of any text with")
st.subheader("Made by 'Shahbaz Mehmood'")
# User input
user_input = st.text_area("Enter text (multiple sentences supported):", 
                          "I love AI! It's amazing. But sometimes, it can be frustrating.")

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Split text into sentences
        sentences = user_input.split(".")
        sentiment_results = []
        all_text = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                scores = sia.polarity_scores(sentence)
                sentiment_results.append((sentence, scores))
                all_text += sentence + " "

        # Overall sentiment score (average)
        avg_score = sum([res[1]["compound"] for res in sentiment_results]) / len(sentiment_results)

        # Determine sentiment label
        if avg_score >= 0.05:
            sentiment_label = "😊 Positive"
            emoji = "😍"
            st.success(f"**Overall Sentiment:** {sentiment_label} {emoji}")
        elif avg_score <= -0.05:
            sentiment_label = "😡 Negative"
            emoji = "😡"
            st.error(f"**Overall Sentiment:** {sentiment_label} {emoji}")
        else:
            sentiment_label = "😐 Neutral"
            emoji = "😐"
            st.warning(f"**Overall Sentiment:** {sentiment_label} {emoji}")

        # Display sentiment scores
        st.subheader("🔢 Sentiment Scores per Sentence")
        for sentence, scores in sentiment_results:
            st.write(f"**📌 Sentence:** {sentence}")
            st.write(f"🔹 Positive: {scores['pos']}, 🔹 Neutral: {scores['neu']}, 🔹 Negative: {scores['neg']}, 🔹 Compound: {scores['compound']}")
            st.write("---")

        # Word Cloud
        st.subheader("🌥 Word Cloud of Text")
        wordcloud = WordCloud(width=600, height=300, background_color="white").generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Sentiment Distribution Chart
        st.subheader("📊 Sentiment Score Distribution")
        scores_df = {"Sentiment": ["Positive", "Neutral", "Negative"],
                     "Score": [sum([res[1]["pos"] for res in sentiment_results]),
                               sum([res[1]["neu"] for res in sentiment_results]),
                               sum([res[1]["neg"] for res in sentiment_results])]}
        
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=scores_df["Sentiment"], y=scores_df["Score"], palette=["green", "gray", "red"])
        ax.set_ylabel("Score")
        ax.set_title("Sentiment Score Breakdown")
        st.pyplot(fig)

    else:
        st.warning("⚠ Please enter some text to analyze!")

