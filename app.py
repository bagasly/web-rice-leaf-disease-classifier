import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model("model/model.h5", compile=False)

model = load_model_cached()
labels = ['Bacterial Blight', 'Blast', 'Brown Spot', 'Tungro']

# Preprocess image function
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function for prevention and treatment
def get_treatment_info(prediction):
    if prediction == 'Bacterial Blight':
        return (
            "#### Pencegahan:\n- Gunakan benih bebas penyakit.\n- Terapkan rotasi tanaman.\n- Gunakan varietas padi yang tahan terhadap penyakit.\n\n"
            "#### Pengobatan:\n- Penyemprotan fungisida berbasis tembaga.\n- Pemangkasan daun yang terinfeksi."
        )
    elif prediction == 'Blast':
        return (
            "#### Pencegahan:\n- Gunakan benih yang tahan terhadap blast.\n- Lakukan rotasi tanaman.\n- Hindari kelembaban berlebih.\n\n"
            "#### Pengobatan:\n- Gunakan fungisida berbasis triazol atau strobilurin.\n- Pemangkasan tanaman yang terinfeksi."
        )
    elif prediction == 'Brown Spot':
        return (
            "#### Pencegahan:\n- Gunakan benih sehat dan bebas penyakit.\n- Jaga keseimbangan pH tanah.\n- Hindari kelembaban tinggi.\n\n"
            "#### Pengobatan:\n- Penyemprotan fungisida seperti mancozeb atau copper oxychloride.\n- Pemangkasan daun yang sakit."
        )
    elif prediction == 'Tungro':
        return (
            "#### Pencegahan:\n- Gunakan varietas padi tahan tungro.\n- Jaga kebersihan lingkungan pertanian.\n- Kendalikan vektor serangga.\n\n"
            "#### Pengobatan:\n- Pengendalian vektor wereng dengan insektisida.\n- Pembuangan tanaman yang terinfeksi."
        )
    else:
        return "Tidak ada informasi untuk penyakit ini."

# Streamlit app interface
st.title('Rice Leaf Disease Classifier')

# File uploader
uploaded_file = st.file_uploader("Upload an image of a rice leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    target_height = 400
    target_width = int(image.width * (target_height / image.height))
    st.image(image, caption="Uploaded Image", width=target_width)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    preds = model.predict(processed_image)[0]
    pred_index = np.argmax(preds)
    prediction = labels[pred_index]
    confidence = round(preds[pred_index] * 100, 2)

    # Display prediction result
    st.write(f"Prediksi: {prediction}")
    st.write(f"Tingkat Keyakinan: {confidence}%")

    # Display prevention and treatment info
    treatment_info = get_treatment_info(prediction)
    st.subheader("Pencegahan dan Pengobatan")
    st.markdown(treatment_info)
