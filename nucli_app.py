# pip install streamlit --user
# streamlit run nucli.py

import streamlit as st
import test_model
import os
from PIL import Image
from skimage.io import imread, imshow
from skimage.transform import resize
st.set_page_config(layout="wide")
st.title("Nuclie Segementation App")


TEST_PATH = "data/stage1_test"
TEST_IMAGES_DIR = os.listdir(TEST_PATH)
test_n = len(os.listdir(TEST_PATH))

list=[]
list1=[]
for i in range(test_n):
    img_path = TEST_PATH + "/" + TEST_IMAGES_DIR[i] +"/images"
    img_name = os.listdir(img_path)[0]
    img=img_path+"/"+img_name
    list.append(img_name)
    list1.append(img)

# for i in range(test_n):
#     pic=Image.open(list1[i])
#     pic.save(list[i])
    

img = st.sidebar.selectbox(
    "Select Image", 
    ([list[i] for i in range(len(list))])
)
    
input_image = img

image = Image.open(input_image)
st.image(image, width=400)

detect_mask = st.button("Detect Mask")

if detect_mask:
    mask = test_model.do_pred(input_image)
    st.write("Predicted Mask")
    st.image(mask, width=500)