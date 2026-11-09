"""
Streamlit app for Next Word Prediction (PyTorch)
Run:  streamlit run app_torch.py
"""
import pickle

import streamlit as st
import torch

from model import NextWordLSTM, predict_top_k

st.set_page_config(page_title="Next Word Prediction", page_icon="📜")


@st.cache_resource
def load_assets():
    with open("vocab.pickle", "rb") as f:
        v = pickle.load(f)
    model = NextWordLSTM(len(v["word2idx"]) + 1)
    model.load_state_dict(torch.load("next_word_lstm.pt", map_location="cpu"))
    model.eval()
    return model, v["word2idx"], v["idx2word"], v["max_len"]


model, word2idx, idx2word, max_len = load_assets()

st.title("📜 Next Word Prediction using LSTM")
st.caption("Trained on Shakespeare's Hamlet")

text = st.text_input("Enter a sequence of words:", "To be or not to")

if st.button("Predict Next Word"):
    results = predict_top_k(model, word2idx, idx2word, max_len, text, k=3)
    if results:
        st.success(f"Predicted next word: **{results[0][0]}**")
        st.write("Top 3 predictions:")
        for word, p in results:
            st.write(f"- {word}  ({p:.1%})")
    else:
        st.warning("None of those words are in the Hamlet vocabulary. Try different words.")
