import streamlit as st
import nltk
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")
st.title("🎭 Sentiment Analyzer")

analyzer = SentimentIntensityAnalyzer()

# Input options
option = st.radio("Input method", ["Type Text", "Upload .txt File"])

if option == "Upload .txt File":
    file = st.file_uploader("Upload a .txt file", type=["txt"])
    text = file.read().decode("utf-8") if file else ""
else:
    text = st.text_area("Enter text (one sentence per line)", height=200)

if st.button("Analyze") and text.strip():
    sentences = [s.strip() for s in text.strip().split("\n") if s.strip()]
    results = []

    for sentence in sentences:
        score = analyzer.polarity_scores(sentence)
        compound = score['compound']
        if compound >= 0.05:
            label, color = "POSITIVE 😊", "green"
        elif compound <= -0.05:
            label, color = "NEGATIVE 😞", "red"
        else:
            label, color = "NEUTRAL 😐", "gray"
        results.append({"Sentence": sentence, "Sentiment": label, "Score": round(compound, 2), "Color": color})

    for r in results:
        st.markdown(f":{r['Color']}[**{r['Sentiment']}** — *{r['Sentence']}*] (Score: {r['Score']})")

    st.divider()

    # Bar chart
    st.subheader("📊 Overall Breakdown")
    scores = analyzer.polarity_scores(" ".join(sentences))
    chart_data = pd.DataFrame({
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Score": [scores['pos'], scores['neg'], scores['neu']]
    })
    st.bar_chart(chart_data.set_index("Sentiment"))

    # Word cloud
    st.subheader("☁️ Word Cloud")
    wc = WordCloud(width=800, height=300, background_color="white").generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Download CSV
    st.divider()
    df = pd.DataFrame([{"Sentence": r["Sentence"], "Sentiment": r["Sentiment"], "Score": r["Score"]} for r in results])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv, "results.csv", "text/csv")