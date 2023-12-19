import streamlit as st
from transformers import AutoTokenizer, CamembertForSequenceClassification
import pandas as pd
import docx2txt
import pdfplumber
import torch
import io
import speech_recognition as sr


image_url = 'https://media.istockphoto.com/id/928583526/fr/vectoriel/clipart-dr%C3%B4le-dun-homme-fran%C3%A7ais-avec-baguette-et-vin.jpg?s=612x612&w=0&k=20&c=XqWZ8jOt6EM5EgfLdZ0CPBZ8iAucA1gqvimtBrtMslo='
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="{image_url}" alt="Image centrée" width="300">
    </div>
    """,
    unsafe_allow_html=True
)
@st.cache_resource
def load_model():
    model_path = r'./camembert_full_model'
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction_idx = torch.argmax(outputs.logits, dim=1).item()
    return label_mapping[prediction_idx]

label_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}


# Interface Streamlit
st.title('Prédicteur du niveau de votre français')

option = st.selectbox('Choisissez votre mode de saisie de texte', ('Saisie manuelle', 'Téléchargement de fichier', 'Téléchargement de fichier audio'))

def transcribe_audio(file):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file)
    with audio_file as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="fr-FR")
    except sr.UnknownValueError:
        text = "La reconnaissance vocale n'a pas réussi à comprendre l'audio"
    except sr.RequestError:
        text = "Impossible d'obtenir des résultats depuis le service de reconnaissance vocale"
    return text
def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return docx2txt.process(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
            return ' '.join(df.astype(str).values.flatten())
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                pages = [page.extract_text() for page in pdf.pages if page.extract_text() is not None]
            return "\n".join(pages)
    return ""

if option == 'Saisie manuelle':
    user_input = st.text_area("Entrez le texte que vous souhaitez analyser")
    text_to_analyze = user_input
elif option == 'Téléchargement de fichier':
    uploaded_file = st.file_uploader("Téléchargez votre fichier (txt, docx, xlsx, pdf)", type=['txt', 'docx', 'xlsx', 'pdf'])
    text_to_analyze = process_uploaded_file(uploaded_file)
else:
    uploaded_audio = st.file_uploader("Téléchargez votre fichier audio", type=['wav'])
    if uploaded_audio is not None:
        text_to_analyze = transcribe_audio(uploaded_audio)
        st.write(text_to_analyze)

# Mapping des valeurs numériques aux labels
labels = {1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2', 5: 'C1', 6: 'C2'}
value = st.slider("Essayer d'estimer votre niveau de français avant de prédire le score", 1, 6, 3)
st.write(f"Votre sélection : {labels[value]}")

# Bouton pour prédiction
if st.button('Prédire'):
    if text_to_analyze:
        model = load_model()
        tokenizer = AutoTokenizer.from_pretrained('camembert-base')
        prediction = predict(text_to_analyze, model, tokenizer)
        st.write(f"Prédiction : {prediction}")
    else:
        st.write("Veuillez entrer du texte ou télécharger un fichier.")