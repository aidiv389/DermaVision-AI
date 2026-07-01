import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------- PAGE SETTINGS ---------------- #

st.set_page_config(
    page_title="DermaVision AI",
    page_icon="🩺",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "saved_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Acne", "Eczema", "Psoriasis", "Vitiligo"]

# ---------------- QUESTION BANK ---------------- #

INFO = {

    "Acne": {

        "symptoms": [
            "Blackheads or whiteheads",
            "Red or pus-filled pimples",
            "Oily skin",
            "Tender or painful bumps",
            "Breakouts on the face, chest or back"
        ],

        "causes": [
            "Hormonal changes",
            "Excess oil (sebum) production",
            "Clogged hair follicles",
            "Bacterial growth",
            "Stress and genetics"
        ],

        "tips": [
            "Wash your face gently twice a day",
            "Avoid squeezing or picking pimples",
            "Use non-comedogenic skincare products",
            "Keep your skin clean and moisturized",
            "Consult a dermatologist if acne becomes severe"
        ]
    },

    "Eczema": {

        "symptoms": [
            "Dry, rough or cracked skin",
            "Intense itching",
            "Red or inflamed patches",
            "Skin thickening due to scratching",
            "Recurring flare-ups"
        ],

        "causes": [
            "Genetic predisposition",
            "Allergies",
            "Dry weather",
            "Harsh soaps or chemicals",
            "Stress"
        ],

        "tips": [
            "Moisturize your skin regularly",
            "Avoid harsh soaps and fragrances",
            "Take short lukewarm showers",
            "Avoid scratching affected areas",
            "Wear soft cotton clothing"
        ]
    },

    "Psoriasis": {

        "symptoms": [
            "Thick red patches",
            "Silvery scales",
            "Itching or burning sensation",
            "Dry, cracked skin",
            "Possible nail changes"
        ],

        "causes": [
            "Overactive immune response",
            "Family history",
            "Stress",
            "Skin injury",
            "Certain infections"
        ],

        "tips": [
            "Keep skin well moisturized",
            "Avoid scratching lesions",
            "Use gentle skincare products",
            "Manage stress effectively",
            "Follow medical advice from a dermatologist"
        ]
    },

    "Vitiligo": {

        "symptoms": [
            "White patches due to pigment loss",
            "Symmetrical skin discoloration",
            "Premature whitening of hair",
            "Usually painless",
            "Gradual spread of patches"
        ],

        "causes": [
            "Autoimmune condition",
            "Family history",
            "Genetic factors",
            "Stress",
            "Unknown environmental triggers"
        ],

        "tips": [
            "Protect affected skin from sunlight",
            "Use broad-spectrum sunscreen",
            "Avoid skin injuries when possible",
            "Consider cosmetic camouflage if desired",
            "Consult a dermatologist for treatment options"
        ]
    }

}

# ---------------- UI ---------------- #

st.title("🩺 DermaVision AI")
st.subheader("AI-Powered Skin Disease Analysis & Symptom Assessment")

st.divider()

st.markdown("### 📤 Upload Skin Image")

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- MAIN APP ---------------- #

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    if st.button("🔍 Analyze Image", use_container_width=True):

        with st.spinner("Analyzing image..."):

            img = image.resize((224, 224))

            img_array = np.array(img)

            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction)
        prediction_name = class_names[predicted_index]
        confidence = float(np.max(prediction) * 100)

        # ---------------- RESULTS ---------------- #

        st.success("Analysis Complete!")

        st.divider()

        st.subheader("🩺 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction", prediction_name)

        with col2:
            st.metric("AI Confidence", f"{confidence:.2f}%")

        st.divider()

        # ---------------- SYMPTOMS ---------------- #

        st.subheader("🩺 Typical Symptoms")

        for symptom in INFO[prediction_name]["symptoms"]:
            st.write(f"✔ {symptom}")

        st.divider()

        # ---------------- CAUSES ---------------- #

        st.subheader("🧬 Possible Causes / Triggers")

        for cause in INFO[prediction_name]["causes"]:
            st.write(f"• {cause}")

        st.divider()

        # ---------------- TIPS ---------------- #

        st.subheader("💡 General Skin Care Tips")

        for tip in INFO[prediction_name]["tips"]:
            st.write(f"• {tip}")

        st.divider()

        # ---------------- DISCLAIMER ---------------- #

        st.warning(
            """
### ⚠ Medical Disclaimer

DermaVision AI provides an AI-assisted prediction based on the uploaded image.

This application is intended for **educational purposes only** and **does not replace professional medical diagnosis or treatment**.

If you have persistent, worsening, or concerning skin symptoms, please consult a qualified dermatologist.
"""
        )
        st.warning("Project Team Members:-")
        st.write("Divya Naruka")
        st.write("Anushka Sahu")
        st.write("Ishika Singh")
