from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2

#OPEN THE IMAGE 
image = cv2.imread('/Users/sahityasridhar/Downloads/Satellite Images of different areas in delhi/Category A/anandlok.jpg')

#APPLY GAUSSIAN BLUR
blur = cv2.GaussianBlur(img,(5,5),0)

#CHOOSE A KERNEL AND PERFORM EROSION AND DILATION
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(blur,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 1)
image_kmeans = dilation

#SAVE IMAGE SHAPE AND RESIZE IMAGE
a = image_kmeans.shape[0]
b = image_kmeans.shape[1]
image_kmeans = cv2.cvtColor(image_kmeans,cv2.COLOR_BGR2RGB)
image_kmeans = image_kmeans.reshape((image_kmeans.shape[0] * image_kmeans.shape[1], 3))

#TRAIN KMEANS TO SEPARATE GREEN FROM OTHER COLOURS
clt = KMeans(n_clusters = 2)
clt.fit(image_kmeans)

#PREDICT ON IMAGE
prediction = clt.predict(image_kmeans)

#RESIZE LABELLED IMAGE AND DISPLAY
prediction = prediction.reshape(a,b)
plt.imshow(prediction)
