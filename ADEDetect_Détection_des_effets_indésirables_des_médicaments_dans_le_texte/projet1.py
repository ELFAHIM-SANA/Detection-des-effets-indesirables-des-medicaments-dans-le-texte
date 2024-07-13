import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger les objets nécessaires
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)
model = load_model('lstm_model.h5')

# Ajouter du CSS pour styliser l'arrière-plan et le bouton
st.markdown(
    """
    <style>
    .main {
        background-color: #ADD8E6;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
    }
    .stButton>button:hover {
        background-color: #1C86EE;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('ADEDetect : Détection des effets indésirables dans les médicaments ✅❌')
user_input = st.text_area("Entrez votre texte ici:")

if st.button("Classer"):
    if user_input:
        sequence = tokenizer.texts_to_sequences([user_input])
        sequence_padded = pad_sequences(sequence, maxlen=100)
        prediction = model.predict(sequence_padded)
        predicted_class = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
        st.write(f"Classe prédite: {predicted_class[0]}")
    else:
        st.write("Veuillez entrer un tweet.")