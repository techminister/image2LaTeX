import cv2 as cv
import numpy as np
import os
from random import randint
from model import load_model
from PIL import Image


def show(img, wait=True):
	cv.imshow("im", img)
	if wait: cv.waitKey(0)

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

def crop(img, bounds, pad=None):
	# Crop the image onto the given bounds.
	(vl, hl), (vr, hr) = bounds
	roi = img[hl:hr+1, vl:vr+1]
	if pad != None:
		vpad = hpad = pad
	else:
		vpad = (img.shape[0] - roi.shape[0])
		hpad = (img.shape[1] - roi.shape[1])
	return cv.copyMakeBorder(roi, vpad//2, (vpad+1)//2, hpad//2, (hpad+1)//2, cv.BORDER_CONSTANT, value=255)

def center(img, pad=None):
	return crop(img, get_bounds(img), pad=pad)

def square(img):
	# Make an image square by padding the short edge.
	height, width = img.shape[0], img.shape[1]
	if height == width: return img
	side = max(height, width)
	vpad = side - height
	hpad = side - width
	return cv.copyMakeBorder(img, vpad//2, (vpad+1)//2, hpad//2, (hpad+1)//2, cv.BORDER_CONSTANT, value=255)

def get_blocks(arr):
	# Given a boolean array, return an array of the left/right indices for contiguous True blocks
	indices = []
	index = 0
	while True in arr:
		left = arr.argmax()
		right = left + arr[left:].argmin()
		arr = arr[right:]
		indices.append((index + left, index + right))
		index += right
	return indices

# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

#image = image_resize(image, height = 800)


if __name__ == "__main__":


	image_dir = r"D:\CVProject2\LaTeXTranscription\test\screenshots"
	dirs = list(os.scandir(image_dir))
	name = dirs[randint(0, len(dirs) - 1)].name
	img = cv.imread(f"{image_dir}/test19.png", cv.IMREAD_GRAYSCALE)
	#show(img)
	#img = image_resize(img, height = 30)
	#img = imutils.resize(img, height=30)
	#show(img)
	# img = cv.imread(f"{image_dir}/{name}", cv.IMREAD_GRAYSCALE)

	# Denoise the image
	# blur = cv.GaussianBlur(img, (3, 3), 0)--
	# blur = cv.medianBlur(img, 3)
	# blur = img
	# Now threshold and center around ROI
	_, thresh = cv.threshold(img, 210, 255, cv.THRESH_BINARY)
	#show(thresh)
	bounds = get_bounds(thresh)
	thresh = crop(thresh, bounds, pad=30)
	img = crop(img, bounds, pad=30)
	seg = thresh.min(axis=0) < 127

	# Each block of True is a character by itself;
	# find the regions contained by these blocks
	blocks = get_blocks(seg)

	model = load_model("scc.model", "cuda")
	#show(img)
	chars = [square(center(img[:,l:r], pad=30)) for l, r in blocks]
	eqn = ''
	for char in chars:
		img = Image.fromarray(char)
		#print(model.classify(img))
		latex = model.classify(img)[0][0]
		eqn += latex + ' '
		#show(char)
		#show(char)
	print(eqn)
	# Fundamentally, what we need is not to detecet the edge; it is to detect the blank space.