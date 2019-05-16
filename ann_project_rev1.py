# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:15:26 2019

@author: baykut
"""
# %%

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical 
from sklearn import svm
import timeit

print(tf.__version__)

# %%

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# %%

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %%

train_images.shape

# %%

train_labels

# %%

test_images.shape

# %%

test_labels
 
 # %%
 
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# %%

#train_images = np.where(train_images > 0 ,1 , 0)
#test_images = np.where(test_images > 0 ,1 , 0)
#
#train_images_90 = np.rot90(train_images, 1, (1,2))
#train_images_180 = np.rot90(train_images_90, 1, (1,2))
#train_images_270 = np.rot90(train_images_180, 1, (1,2))
#train_images = np.append(train_images, train_images_90, axis = 0)
#train_images = np.append(train_images, train_images_180, axis = 0)
#train_images = np.append(train_images, train_images_270, axis = 0)
#train_labels_rot = train_labels
#train_labels=np.append(train_labels, train_labels_rot, axis = 0)
#train_labels=np.append(train_labels, train_labels_rot, axis = 0)
#train_labels=np.append(train_labels, train_labels_rot, axis = 0)

train_images = train_images / 255.0
test_images = test_images / 255.0

# %%

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# %% Model KMeans

train_images = train_images.reshape(np.shape(train_images)[0], -1)
test_images = test_images.reshape(np.shape(test_images)[0], -1)

output='replace' 
n_clusters = len(np.unique(train_labels))
clustering = KMeans(n_clusters = n_clusters, random_state=42)
clustering.fit(train_images)
y_labels_train = clustering.labels_
y_labels_test = clustering.predict(test_images)
if output == 'add':
    train_images['km_clust'] = y_labels_train
    test_images['km_clust'] = y_labels_test
elif output == 'replace':
    train_images =  y_labels_train[:, np.newaxis]
    test_images = y_labels_test[:, np.newaxis]


n_clusters = len(np.unique(train_labels))
train_images = to_categorical(train_images, num_classes = n_clusters)
test_images = to_categorical(test_images, num_classes = n_clusters)

model= svm.SVC()
history = model.fit(train_images, train_labels)
y_pred = model.predict(test_images)
print('KMeans + SVM: {}'.format(accuracy_score(test_labels, y_pred)))

# %% Model LogisticRegression

acc = []
model = LogisticRegression(random_state=42)
for i in range(2, 26):
    pca = PCA(n_components = i, svd_solver='full')
    train_images_pca = pca.fit_transform(train_images.reshape(np.shape(train_images)[0], -1)) 
    test_images_pca = pca.transform(test_images.reshape(np.shape(test_images)[0], -1))
    model.fit(train_images_pca, train_labels)
    y_pred = model.predict(test_images_pca)
    accuracy = accuracy_score(test_labels, y_pred)
    acc.append(accuracy)
    print('PCA: PC', i, 'Logistic Regression Test Accuracy: ', format(accuracy_score(test_labels, y_pred)))
    
plt.plot(acc)

# %% RandomForest

train_images = train_images.reshape(np.shape(train_images)[0], -1)
test_images = test_images.reshape(np.shape(test_images)[0], -1)

model = RandomForestClassifier(n_estimators=64, n_jobs=-1)
model.fit(train_images, train_labels.ravel())
y_pred = model.predict(test_images)
print(accuracy_score(test_labels, y_pred))

# %% Model 1

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# %% Model 2

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
    
model = keras.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding = 'Same', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding = 'Same'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# %% Model 2.PCA

pca = PCA(n_components = 256, svd_solver='full')
train_images = pca.fit_transform(train_images.reshape(60000, -1)) 
test_images = pca.transform(test_images.reshape(10000, -1))
    
train_images = train_images.reshape(-1, 16, 16, 1)
test_images = test_images.reshape(-1, 16, 16, 1)
    
model = keras.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding = 'Same', input_shape=(16, 16, 1)),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding = 'Same'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
    
# %% Model 3

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

model = keras.Sequential([
    keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding = 'Same', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding = 'Same'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding = 'Same'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# %% Model 4

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
    
model = keras.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding = 'Same', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding = 'Same'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding = 'Same'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
    
# %% Model 4.PCA

pca = PCA(n_components = 256, svd_solver='full')
train_images = pca.fit_transform(train_images.reshape(60000, -1)) 
test_images = pca.transform(test_images.reshape(10000, -1))
    
train_images = train_images.reshape(-1, 16, 16, 1)
test_images = test_images.reshape(-1, 16, 16, 1)
    
model = keras.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding = 'Same', input_shape=(16, 16, 1)),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding = 'Same'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding = 'Same'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# %%

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%

start_time = timeit.default_timer()

history = model.fit(train_images, train_labels, validation_split=0.3, epochs=10)

print(timeit.default_timer() - start_time)

# %%

print("Final Scores")
print("****************************")
print("Accuracy: {0:.3f}".format(history.history['acc'][-1]))
print("Validation Accuracy: {0:.3f}".format(history.history['val_acc'][-1]))
print("Loss: {0:.3f}".format(history.history['loss'][-1]))
print("Validation Loss: {0:.3f}".format(history.history['val_loss'][-1]))
print("****************************")

# %%

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)
print('Test Loss:', test_loss)

# %%


predictions = model.predict(test_images)

# %%

predictions[0]

# %%

np.argmax(predictions[0])

# %%

test_labels[0]
   
# %%

train_images = np.squeeze(train_images, axis=3)
test_images = np.squeeze(test_images, axis=3)

# %%

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# %%
    
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# %%

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# %%

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# %%

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# %%

img = test_images[0]
img = (np.expand_dims(img,0))

# %%

predictions_single = model.predict(img)

plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

# %%

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)