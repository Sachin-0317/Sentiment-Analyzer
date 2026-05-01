import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("Sentiment Analyzer")
text = st.text_area("Enter text here", height=150)

if st.button("Analyze"):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    
    if compound >= 0.05:
        label = "POSITIVE 😊"
    elif compound <= -0.05:
        label = "NEGATIVE 😞"
    else:
        label = "NEUTRAL 😐"
    
    st.write(f"**Sentiment:** {label}")
    st.write(f"**Score:** {round(compound, 2)}")