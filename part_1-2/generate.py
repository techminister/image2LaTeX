import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2, io
from pathlib import Path
from symbols import symbols

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"""\usepackage{amsmath}
\usepackage{amssymb}""")


def find_limits(arr):
	# Given a boolean array, find the left and rightmost index which are True.
	return arr.argmax(), len(arr) - np.flip(arr).argmax() - 1

def get_bounds(img):
	# Get the bounding box of non-white area in the image.
	binarize = img != 255
	vert = np.sum(binarize, axis=0) != 0
	hori = np.sum(binarize, axis=1) != 0
	vl, vr = find_limits(vert)
	hl, hr = find_limits(hori)
	return (vl, hl), (vr, hr)

def center(img):
	# Center the image around non-white area.
	(vl, hl), (vr, hr) = get_bounds(img)
	roi = img[hl:hr+1, vl:vr+1]
	vpad = (img.shape[0] - roi.shape[0])
	hpad = (img.shape[1] - roi.shape[1])
	return cv2.copyMakeBorder(roi, vpad//2, (vpad+1)//2, hpad//2, (hpad+1)//2, cv2.BORDER_CONSTANT, value=255)

def fig_to_img(fig, dpi):
	# Convert fig to cv2 image
	buff = io.BytesIO()
	fig.savefig(buff, format="png", dpi=dpi, pad_inches=0.0)
	plt.close()
	buff.seek(0)
	arr = np.frombuffer(buff.read(), dtype=np.uint8)
	return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

def generate_latex(formula, file_id, fontsize=12, size=120):
	fig = plt.figure(figsize=(0.3, 0.3))
	fig.text(0.025, 0.35, "\\begin{align*}" + formula + "\\end{align*}", fontsize=fontsize)
	image = fig_to_img(fig, size*10/3)
	image = center(image)
	cv2.imwrite(f"data/{file_id:04d}/0001.png", image)


if __name__ == "__main__":
	for ind in range(len(symbols)):
		Path(f"data/{ind + 1}").mkdir(parents=True, exist_ok=True)
	for ind, symbol in enumerate(tqdm(symbols)):
		generate_latex(symbol, ind)
	pd.DataFrame(enumerate(symbols), columns=["file_id", "formula"]).to_csv("meta.csv", index=False)