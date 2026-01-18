import streamlit as st
import joblib

# Load files
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("emotion_model.pkl")
le    = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Emotion Detection", page_icon="😊")

st.title("🧠 Emotion Detection App")
st.subheader("Created by Aixi ✨")
st.write("Detect emotions from text")

text = st.text_area("Enter your sentence")

if st.button("Predict Emotion"):
    if text.strip():
        vec = tfidf.transform([text])
        pred = model.predict(vec)
        emotion = le.inverse_transform(pred)[0]

        emoji = {
            "joy": "😄",
            "sadness": "😢",
            "anger": "😡",
            "fear": "😨",
            "love": "❤️",
            "surprise": "😲"
        }

        st.success(f"Emotion: {emoji[emotion]} **{emotion.upper()}**")
    else:
        st.warning("Please enter some text")

st.markdown("---")
st.markdown(
    "<center> Developed by <b>Aixi</b></center>",
    unsafe_allow_html=True
)
