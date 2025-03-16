import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import json

st.header('Garment Fabric Type Classifier')
try:
    model = load_model('garmet_model1.keras')
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()  

data_cat = [
    'Acrylic', 'Artificial_fur', 'Artificial_leather', 'Chenille', 'Corduroy', 'Cotton', 
    'Crepe', 'Denim', 'Felt', 'Fleece', 'Leather', 'Linen', 'Lut', 'Nylon', 'Polyester', 
    'Satin', 'Silk', 'Suede', 'Terrycloth', 'Velvet', 'Viscose', 'Wool'
]

# Load fabric details from a local JSON file
def load_fabric_details():
    with open('fabric_details.json') as f:
        return json.load(f)

fabric_details = load_fabric_details()

img_height = 180
img_width = 180
image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
if image is not None:
    image_load = tf.keras.preprocessing.image.load_img(image, target_size=(img_height, img_width))
    img_arr = tf.keras.preprocessing.image.img_to_array(image_load)  
    img_bat = tf.expand_dims(img_arr, 0) 
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    score_np = score.numpy().flatten()
    st.image(image, width=500)
    predicted_class = data_cat[np.argmax(score_np)]
    predicted_prob = np.max(score_np) * 100
    st.write(f'Fabric in the image is: **{predicted_class}**')
    st.write(f'Prediction confidence: **{predicted_prob:.2f}%**')
    #st.write('Class Probabilities:')
    #for i, class_name in enumerate(data_cat):
    #   st.write(f'{class_name}: {score_np[i] * 100:.2f}%')
    
    # Fetch fabric details from the local JSON file
    if predicted_class in fabric_details:
        fabric_info = fabric_details[predicted_class]
        
        st.subheader(f'Details about {predicted_class}')
        
        # Benefits
        st.write('### Benefits:')
        st.write(fabric_info.get('benefits', 'No information available.'))
        
        # Ideal Season and Climate
        st.write('### Ideal Season and Climate:')
        st.write(fabric_info.get('ideal_season', 'No information available.'))
        
        # Functional Recommendations
        st.write('### Functional Recommendations:')
        st .write(fabric_info.get('recommendations', 'No information available.'))