#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import random
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPool2D, Conv2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Activation, SimpleRNN
from tensorflow.keras.layers import BatchNormalization, Reshape, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from PIL import Image
from PIL import UnidentifiedImageError

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import itertools
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


# In[3]:


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[4]:


#upright = os.listdir(r'/content/drive/MyDrive/posture/Upright Sitting')
#Crossed_legs = os.listdir(r'/content/drive/MyDrive/posture/Crossing legs')
#slant_bending = os.listdir(r'/content/drive/MyDrive/posture/Slant Bending')
#Lordosis = os.listdir(r'/content/drive/MyDrive/posture/Lordosis')
#slouching = os.listdir(r'/content/drive/MyDrive/posture/Slouching')

#Sitting_Posture = os.listdir('C:/Users/Dell/Desktop/Sitting_Posture')
bending = os.listdir('C:/Users/Dell/Desktop/Human_posture/bending')
lying = os.listdir('C:/Users/Dell/Desktop/Human_posture/lying')
sitting = os.listdir('C:/Users/Dell/Desktop/Human_posture/sitting')
standing = os.listdir('C:/Users/Dell/Desktop/Human_posture/standing')


# In[5]:


img_w, img_h = 150, 150 #setting the image width and height for easy processing
X = []
y = []

for i in bending:
    try:
        img = Image.open("C:/Users/Dell/Desktop/Human_posture/bending/" + i).convert('RGB')
        img = img.resize((img_w,img_h))
        X.append(np.asarray(img))
        y.append(0)
    except UnidentifiedImageError: # it passes an image that wasn't correctly identified
        pass
    
for i in lying:
    try:
        img = Image.open("C:/Users/Dell/Desktop/Human_posture/lying/" + i).convert('RGB')
        img = img.resize((img_w,img_h))
        X.append(np.asarray(img))
        y.append(1)
    except UnidentifiedImageError:
        pass
    
for i in sitting:
    try:
        img = Image.open("C:/Users/Dell/Desktop/Human_posture/sitting/" + i).convert('RGB')
        img = img.resize((img_w,img_h))
        X.append(np.asarray(img))
        y.append(2)
    except UnidentifiedImageError:
        pass
    
for i in standing:
    try:
        img = Image.open("C:/Users/Dell/Desktop/Human_posture/standing/" + i).convert('RGB')
        img = img.resize((img_w,img_h))
        X.append(np.asarray(img))
        y.append(3)
    except UnidentifiedImageError:
        pass
    
    
    

X = np.asarray(X)
y = np.asarray(y)
print(X.shape, y.shape)

# I resized each image to our manually defined width and height(img_w, img_h)
# I also changed the images to an array


# In[6]:


x = X.astype('float32')
x /= 255

#scaler = MinMaxScaler()
#x = scaler.fit_transform(x)

num_classes = 4

labels = keras.utils.to_categorical(y, num_classes)
print(labels[0])


# In[7]:


# splitting our dataset into train and test
x_train1, x_test, y_train1, y_test = train_test_split(x, labels, test_size = 0.15, random_state=5)
x_train,x_val,y_train,y_val=train_test_split(x_train1,y_train1,test_size=0.15,random_state=5)


print('Number of train: {}'.format(len(x_train)))
print('Number of validation: {}'.format(len(x_val)))
print('Number of test: {}'.format(len(x_test)))


# In[8]:


datagen = ImageDataGenerator(
    rotation_range=20.,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=30.,
    zoom_range=0.2,
    horizontal_flip=0.2,
    rescale=None)

datagen.fit(x_train)


# In[9]:


Des121 = tf.keras.applications.DenseNet121(input_shape=(150,150,3),include_top=False,weights="imagenet")


# In[10]:


Des121.trainable = True
#let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(Des121.layers))


# In[11]:


for layer in Des121.layers:
    layer.trainable=False


# In[12]:


# My neural network consists of 3 layers densely connected
# dropout helps prevent over fitting of the model when testing it
model_densenet=Sequential()
model_densenet.add(Des121)
model_densenet.add(Flatten())
model_densenet.add(Dense(128, activation='relu'))
model_densenet.add(Dropout(0.5))
model_densenet.add(Dense(128, activation='relu'))
model_densenet.add(Dropout(0.5))
model_densenet.add(Dense(4, activation='softmax'))
model_densenet.summary()


# In[13]:


# here i am defining my performance metrics to check the performance of the model
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

Metrics = [
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
        f1_score,
]


# In[14]:


lrd = ReduceLROnPlateau(monitor = 'recall',patience = 10,verbose = 1,factor = 0.70, min_lr = 1e-5)
# the above line of code reduces the learning rate when there is no improvement to the metric
# i added this line of code because i trained the model without it and the performance was stagnant
# the performance was stuck on 0.96nnn as the accuracy so i added the line of code to reduce
# the learing rate after two epochs
mcp = ModelCheckpoint('model.h5')
es = EarlyStopping(verbose=1, patience=15)

model_densenet.compile(optimizer='adam', loss = tf.keras.losses.CategoricalCrossentropy(),metrics=Metrics)


# In[15]:


# this is where i fit and trained my model
get_ipython().run_line_magic('time', '')
history=model_densenet.fit(x_train,y_train,validation_data=(x_val,y_val),epochs = 30,verbose = 1, callbacks=[lrd,mcp,es])


# In[16]:


Test_data = model_densenet.evaluate(x_test, y_test)


# In[18]:


plt.style.use("ggplot")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['accuracy']
val_loss = history.history['val_accuracy']
epochs_range = range(len(history.history['val_accuracy']))
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'go--', c = "r", label='Training Loss')
plt.plot(history.history['val_loss'], 'go--', c = "b", label='Validation Loss')
plt.legend(loc='upper right')
plt.title('DenseNet121 Model For Train and Valid Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'go--', c = "r", label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'go--', c = "b", label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('DenseNet121 Model For Train and Valid Accuracy')
plt.show()


# In[19]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('DenseNet121 Model Training and validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.plot(epochs, accuracy, 'r', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'g', label='Validation Accuracy')
plt.title('DenseNet121 Model Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[22]:


# here i want to define and plot my confusion matrix
# and also my classification report
def plot_confusion_matrix(cm, classes, normalize=True, title='DenseNet121 Model Confusion matrix', cmap=plt.cm.Purples):
    
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual Posture')
    plt.xlabel('Predicted Posture')
    
    
y_pred = (model_densenet.predict(x_test) > 0.5).astype("int32")

y_test_c = np.argmax(y_test, axis=1)
target_names = ["bending", "lying", "sitting", "standing"]

Y_pred = np.argmax(model_densenet.predict(x_test),axis=1)
print('DenseNet121 Model Confusion Matrix')
cm = confusion_matrix(y_test_c, Y_pred)
plot_confusion_matrix(cm, target_names, normalize=False, title='DenseNet121 Model Confusion Matrix')

print('Classification Report')
print(classification_report(y_test_c, Y_pred, target_names=target_names))


# In[23]:


#importing all the necessary libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


#Multiclass ROC
#predicting the data
y_pred_cnb = model_densenet.predict(x_test)
y_prob_pred_cnb = model_densenet.predict(x_test)

#roc auc score
roc_auc_score(y_test, y_prob_pred_cnb, multi_class='ovo', average='weighted')
# roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_class = 4

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test_c, y_prob_pred_cnb[:,i], pos_label=i)
    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='--',color='red', label='bending')
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='lying')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='sitting')
plt.plot(fpr[3], tpr[3], linestyle='--',color='orange', label='standing')
plt.title('ROC curve for DenseNet121')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC',dpi=300);  


# In[24]:


auc = metrics.roc_auc_score(y_test, y_prob_pred_cnb)

print(auc)


# In[26]:


from sklearn.metrics import precision_recall_curve, roc_curve
precision = dict()
recall = dict()
target_names = ["bending", "lying", "sitting", "standing"]
for i in range(n_class):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], 
                                                        y_prob_pred_cnb[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.title("Precision vs Recall curve for DenseNet121")
plt.show()


# In[30]:


from sklearn.metrics import precision_recall_curve, roc_curve
fpr = dict()
tpr = dict()
for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], 
                                  y_prob_pred_cnb[:, i])
    plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))
plt.xlabel("False positive rate")
plt.ylabel("True positive  rate")
plt.legend(loc="best")
plt.title("ROC curve for DenseNet121")
plt.show()


# In[ ]:




