'''
script for line extraction for scanned document based on image processing
'''
import numpy as np
import cv2

# load image
image = cv2.imread("test.png")

# output images
output_line = image.copy()

# gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# clean the image using otsu method with the inversed binarized image
ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# morphology closing
kernel = np.ones((1,15), np.uint8)
closed = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel2,iterations=2)

# dilation
kernel = np.ones((3,1), np.uint8)
line_img = cv2.dilate(temp_img,kernel,iterations=3)

# find contours
(contours, _) = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)

	# filter out image
	if h*w >1000:
		# adding bouding box		
		cv2.rectangle(output_line,(x-1,y-5),(x+w,y+h),(0,255,0),5)
	
# write output
cv2.imwrite("output_line.jpg", output_line)

