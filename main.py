import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model

model = load_model('./Model/BCD.h5',compile=False)


lab = {0: 'Beetle',
 1: 'Butterfly',
 2: 'Cat',
 3: 'Cow',
 4: 'Dog',
 5: 'Elephant',
 6: 'Gorilla',
 7: 'Hippo',
 8: 'Lizard',
 9: 'Monkey',
 10: 'Mouse',
 11: 'Panda',
 12: 'Spider',
 13: 'Tiger',
 14: 'Zebra'}


def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    img1 = Image.open('./logo/logo.jpeg')
    img1 = img1.resize((350,350))
    st.image(img1,use_column_width=False)
    st.title("Animal Species Prediction")
    st.markdown('''<h4 style='text-align: left; color: #ffa500;'> " A lot of animal species can be included in this data set, which is why it gets revised regularly. "</h4>''',
                unsafe_allow_html=True)
    st.markdown('''<h4 style='text-align: left; color: #adff2f;'> " Eng\ Ahmed Khairullah "</h4>''',
                unsafe_allow_html=True)
    count=0


    img_file = st.file_uploader("Choose an Image of Animal",accept_multiple_files=False, type=["jpg", "png", "jpeg"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './Uploaded_Images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

             
        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Animal is: "+result)
run()