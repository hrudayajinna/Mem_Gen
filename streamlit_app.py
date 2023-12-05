import streamlit as st
import os
from PIL import Image
from generator import MemeGenerator
from deephumor.models import (
    CaptioningLSTM, 
    CaptioningLSTMWithLabels, 
    CaptioningTransformerBase,
    CaptioningTransformer
)

FILE_TO_CLASS = {
    'LSTMDecoderWords.best.pth': CaptioningLSTM,
    'TransformerDecoderBaseWords.best.pth': CaptioningTransformerBase,
    'TransformerDecoderChars.best.pth': CaptioningTransformer
}
MEMES_PREDICTIONS = 'memes_predictions'

uploaded_file = st.file_uploader("Choose an image ", type="jpg")

caption = st.text_input("Caption")
temperature_slider = st.select_slider("Temperature", options=[0.1, 0.5, 1.0, 1.5, 2.0])
beam_size_input = st.text_input("Beam size", value=10)
top_k_input = st.text_input("Top k", value=10)
model_name = st.selectbox("Model", options=list(FILE_TO_CLASS.keys()))

if uploaded_file is not None:
    image = Image.open(uploaded_file)


def on_click():
    print('clicked')
    st.session_state.clicked = True


submit_btn = st.button("Generate Meme", on_click=on_click)
if submit_btn and uploaded_file is not None:
    st.write("Generating meme...")
    st.write("Loading model ...")
    model = FILE_TO_CLASS[model_name].from_pretrained(f"./models/{model_name}")
    model.eval()

    generator = MemeGenerator(model=model, mode='word')
    pred = generator.generate(img_path=uploaded_file, caption=caption, T=1.3, beam_size=10, top_k=100)
    st.image(pred)
    pred.save(os.path.join(MEMES_PREDICTIONS, uploaded_file.name))