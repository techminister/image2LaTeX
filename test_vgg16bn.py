import cv2 as cv
import numpy as np
import os
from random import randint
from vgg_16bn import load_model, classify
from PIL import Image
import pandas as pd 
from process2 import show, find_limits, get_bounds, crop, center, square, get_blocks
meta = pd.read_csv('test/screenshots/meta.csv',names=['test_num', 'latex_eqn'], header=None)


#print(meta)


def get_latex_eqn(im_dir, model):
    img = cv.imread(im_dir, cv.IMREAD_GRAYSCALE)
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
        #latex = model.classify(img)[0][0]
        latex = classify(model, img)[0][0]
        eqn += latex + ' '
    return eqn

model = load_model("vgg_16bn.pt", "cuda")
image_dir = r"test/screenshots/"

total_count = 0
correct_count = 0
wrong_samples = []
for index, row in meta.iterrows():
    eqn_actual = row['latex_eqn'].strip().split()  # split the equation in a list
    im_dir = image_dir + row['test_num'] + '.PNG'
    eqn_pred = get_latex_eqn(im_dir, model).strip().split()
    try:
        for i in range(len(eqn_actual)):
            if eqn_actual[i] == eqn_pred[i]:
                correct_count +=1
            elif index not in wrong_samples:
                wrong_samples.append(index)
    except:
        if index not in wrong_samples:
            wrong_samples.append(index)
    total_count += len(eqn_actual)
print('accuracy on test set: ', correct_count/total_count)
print('test samples that are wrongly corrected: ', wrong_samples)

print('checking the wrongly predicted samples....')
for test_num in wrong_samples:
    im_dir = f"{image_dir}test{test_num}.PNG"
    print(im_dir)
    pred = get_latex_eqn(im_dir, model)
    actual = meta['latex_eqn'][test_num]
    print(f'actual latex equation for test{test_num}: {actual}')
    print(f'predicted latex equation for test{test_num}: {pred}')
    print('-'*20)
