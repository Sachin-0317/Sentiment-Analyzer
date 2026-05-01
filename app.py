import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")
st.title("🎭 Sentiment Analyzer")
st.write("Analyze the sentiment of one or multiple sentences.")

text = st.text_area("Enter text (one sentence per line)", height=200)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text!")
    else:
        analyzer = SentimentIntensityAnalyzer()
        sentences = [s.strip() for s in text.strip().split("\n") if s.strip()]
        results = []

        for sentence in sentences:
            score = analyzer.polarity_scores(sentence)
            compound = score['compound']
            if compound >= 0.05:
                label = "POSITIVE 😊"
                color = "green"
            elif compound <= -0.05:
                label = "NEGATIVE 😞"
                color = "red"
            else:
                label = "NEUTRAL 😐"
                color = "gray"
            results.append({"Sentence": sentence, "Sentiment": label, "Score": round(compound, 2), "Color": color})

        for r in results:
            st.markdown(f":{r['Color']}[**{r['Sentiment']}** — *{r['Sentence']}*] (Score: {r['Score']})")

        st.divider()
        st.subheader("📊 Overall Breakdown")
        scores = analyzer.polarity_scores(" ".join(sentences))
        chart_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Score": [scores['pos'], scores['neg'], scores['neu']]
        })
        st.bar_chart(chart_data.set_index("Sentiment"))