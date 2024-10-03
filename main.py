import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Disease Recognition","About"])

#Main Page
if(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    # if(st.button("Show Image")):
    #     st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Your image")
        st.image(test_image,width=4,use_column_width=True)
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
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
                      'Tomato healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

#About Project
elif(app_mode=="About"):
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

# #Prediction Page
# elif(app_mode=="Disease Recognition"):
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:")
#     if(st.button("Show Image")):
#         st.image(test_image,width=4,use_column_width=True)
#     #Predict button
#     if(st.button("Predict")):
#         st.snow()
#         st.write("Our Prediction")
#         result_index = model_prediction(test_image)
#         #Reading Labels
#         class_name = ['Apple Apple_scab', 'Apple Black_rot', 'Apple Cedar_apple_rust', 'Apple healthy',
#                     'Blueberry healthy', 'Cherry_(including_sour) Powdery_mildew', 
#                     'Cherry_(including_sour) healthy', 'Corn_(maize) Cercospora_leaf_spot Gray_leaf_spot', 
#                     'Corn_(maize) Common_rust_', 'Corn_(maize) Northern_Leaf_Blight', 'Corn_(maize) healthy', 
#                     'Grape Black_rot', 'Grape Esca_(Black_Measles)', 'Grape Leaf_blight_(Isariopsis_Leaf_Spot)', 
#                     'Grape healthy', 'Orange Haunglongbing_(Citrus_greening)', 'Peach Bacterial_spot',
#                     'Peach healthy', 'Pepper,_bell Bacterial_spot', 'Pepper,_bell healthy', 
#                     'Potato Early_blight', 'Potato Late_blight', 'Potato healthy', 
#                     'Raspberry healthy', 'Soybean healthy', 'Squash Powdery_mildew', 
#                     'Strawberry Leaf_scorch', 'Strawberry healthy', 'Tomato Bacterial_spot', 
#                     'Tomato Early_blight', 'Tomato Late_blight', 'Tomato Leaf_Mold', 
#                     'Tomato Septoria_leaf_spot', 'Tomato Spider_mites Two-spotted_spider_mite', 
#                     'Tomato Target_Spot', 'Tomato Tomato_Yellow_Leaf_Curl_Virus', 'Tomato Tomato_mosaic_virus',
#                       'Tomato healthy']
#         st.success("Model is Predicting it's a {}".format(class_name[result_index]))
