import streamlit as st
import torch
from PIL import Image
from gtts import gTTS
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, MarianMTModel
import os

# Load the captioning model
caption_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
caption_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

# Load the translation model
src = "en"  # source language
trg = "ar"  # target language
translation_model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
translation_model = MarianMTModel.from_pretrained(translation_model_name)
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)

# Streamlit app
st.title("Arabic Image Captioning")

uploaded_file = st.file_uploader("Choose an image...", type="jpg",)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Transform the image
            pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values

            # Generate caption
            generated_ids = caption_model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            st.write("Caption in English:", generated_caption)

            # Translate caption to Arabic
            batch = translation_tokenizer([generated_caption], return_tensors="pt")
            generated_ids = translation_model.generate(**batch)
            arabic_generated_caption = translation_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            st.write("Caption in Arabic:", arabic_generated_caption)

            # Generate audio for the Arabic caption
            arabic_tts = gTTS(text=arabic_generated_caption, lang='ar')
            audio_path = "arabic_caption_audio.mp3"
            arabic_tts.save(audio_path)

            # Display audio
            audio_file = open(audio_path, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

            # Clean up
            os.remove(audio_path)

