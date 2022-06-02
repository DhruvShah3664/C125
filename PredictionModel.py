import pandas as pd 
import numpy as np
import PIL.ImageOps
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

X,y = fetch_openml('mnist_784', version = 1, return_X_y = True)

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, random_state = 9, test_size = 2500, train_size = 7500)

XtrainScaled = Xtrain/255.0
XtestScaled = Xtest/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(XtrainScaled, ytrain)

def getPrediction(image):
    impil = Image.open(image)
    image_bw = impil.convert('L')
    imgbwResized = image_bw.resize((28, 28), Image.ANTIALIAS)
    pixelfilter = 20
    min_pixel = np.percentile(imgbwResized, pixelfilter)
    imgbwResized_inverted_scaled = np.clip(imgbwResized-min_pixel, 0,255)
    max_pixel = np.max(imgbwResized)
    imgbwResized_inverted_scaled = np.asarray(imgbwResized_inverted_scaled)/max_pixel
    testSample = np.array(imgbwResized_inverted_scaled).reshape(1, 784)
    test_pred = clf.predict(testSample)
    return test_pred[0]

