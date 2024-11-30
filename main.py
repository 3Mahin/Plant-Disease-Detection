import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu

# Custom CSS for background image
def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Call the function with your image URL or local path
add_background_image("https://example.com/path-to-your-image.jpg")  # Replace with your image URL

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Navbar
selected = option_menu(
    menu_title=None,
    options=["Detect", "About"],
    icons=["search", "info-circle-fill"],
    default_index=0,
    orientation="horizontal",
)

# Main Page
if selected == "Detect":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Predict"):
        st.write("Your image")
        st.image(test_image, width=4, use_column_width=True)
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        class_name = [
            'Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
            'Blueberry healthy', 'Cherry (including sour) Powdery mildew',
            'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot',
            'Corn (maize) Common rust', 'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy',
            'Grape Black rot', 'Grape Esca (Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)',
            'Grape healthy', 'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot',
            'Peach healthy', 'Pepper,bell Bacterial spot', 'Pepper,bell healthy',
            'Potato Early blight', 'Potato Late blight', 'Potato healthy',
            'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew',
            'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot',
            'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
            'Tomato Septoria leaf spot', 'Tomato Spider mites Two spotted spider mite',
            'Tomato Target Spot', 'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus',
            'Tomato healthy'
        ]
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

# About Project
elif selected == "About":
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
        This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
        A new directory containing 33 test images is created later for prediction purpose.
        #### Content
        1. train (70295 images)
        2. test (33 images)
        3. validation (17572 images)
    """)
