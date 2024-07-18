import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
from sklearn.decomposition import PCA
import joblib

path = os.listdir('./archive/Training')

classes = {'notumor':0, 'pituitary':1}
X = []
Y = []
for cls in classes:
    pth = './archive/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])
X = np.array(X)
Y = np.array(Y)
np.unique(Y)
pd.Series(Y).value_counts()
X.shape
plt.imshow(X[0], cmap='gray')
X_updated = X.reshape(len(X), -1)
X_updated.shape
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)
xtrain.shape, xtest.shape
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
print(xtrain.shape, xtest.shape)

pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest

warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)
sv = SVC()
sv.fit(pca_train, ytrain)
print("Training Score:", lg.score(pca_train, ytrain))
print("Testing Score:", lg.score(pca_test, ytest))
print("Training Score:", sv.score(pca_train, ytrain))
print("Testing Score:", sv.score(pca_test, ytest))
pred = sv.predict(pca_test)
np.where(ytest!=pred)
pred[36]
ytest[36]
dec = {0:'No Tumor', 1:'Positive Tumor'}
plt.figure(figsize=(12,8))
p = os.listdir('./archive/Testing')
c=1
for i in os.listdir('./archive/Testing/notumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('./archive/Testing/notumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1
plt.figure(figsize=(12,8))
p = os.listdir('./archive/Testing/')
# c=1
# for i in os.listdir('./archive/Testing/pituitary')[:16]:
#     plt.subplot(4,4,c)
    
#     img = cv2.imread('./archive/Testing/pituitary'+i,0)
#     img1 = cv2.resize(img, (200,200))
#     img1 = img1.reshape(1,-1)/255
#     p = sv.predict(img1)
#     plt.title(dec[p[0]])
#     plt.imshow(img, cmap='gray')
#     plt.axis('off')
#     c+=1

# Save the trained model
joblib.dump(sv, 'trained_model.pkl')