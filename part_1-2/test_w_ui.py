import cv2 as cv
import numpy as np
import os
from random import randint
from model_drop import load_model
from PIL import Image
import torch
import pandas as pd 
import streamlit as st
from process2 import show, find_limits, get_bounds, crop, center, square, get_blocks

st.title("Image to LaTeX Equation")
st.markdown("Welcome to our Computer Vision Project that   \
    seeks to help you in converting your screenshots \
        to LaTeX equation. Simply upload a PNG or JPG file and watch \
            the magic happen :)")
btn = st.button("Celebrate!")

if btn:
    st.balloons()




#print(meta)


def get_latex_eqn(im_dir, model):
    open_cv_image = np.array(im_dir)
    img  = cv.cvtColor(open_cv_image, cv.COLOR_BGR2GRAY)
    #img = cv.imread(open_cv_image, cv.IMREAD_GRAYSCALE)
    _, thresh = cv.threshold(img, 210, 255, cv.THRESH_BINARY)
    bounds = get_bounds(thresh)
    thresh = crop(thresh, bounds, pad=30)
    img = crop(img, bounds, pad=30)
    seg = thresh.min(axis=0) < 127
    blocks = get_blocks(seg)
    chars = [square(center(img[:,l:r], pad=30)) for l, r in blocks]
    eqn = ''
    for char in chars:
        img = Image.fromarray(char)
        #print(model.classify(img))
        latex = model.classify(img)[0][0]
        eqn += latex + ' '
    return eqn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("custom_drop.pt", device)


file_up = st.file_uploader("Upload an image", type =['png', 'jpg'])
if not file_up:
    st.empty().info("Please upload a file of type: " + ", ".join(['png', 'jpg']))

if file_up:
    image = Image.open(file_up)
    st.image(image, caption = "Uploaded Image.", use_column_width = True)
    prediction = get_latex_eqn(image, model)
    st.write('The image you have uploaded is interepreted as: ' )
    st.latex(prediction)
    st.write("The LaTeX representation of the equation is: ")
    st.markdown("## " + prediction)

