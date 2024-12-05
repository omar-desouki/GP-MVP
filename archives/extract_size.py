import cv2
import numpy as np

# Open an image file
img = cv2.imread("comic.png")

# Get the size of the image
height, width, channels = img.shape

print(f"Width: {width}, Height: {height}")
