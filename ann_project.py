# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:34:01 2019

@author: 26022308
"""

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import timeit

# %%

df = pd.read_csv("fashion-mnist_train.csv",
                 sep=',',
                 header=0)

# %%
'''
df_sample = df.sample(3000)
df_sample.head()

# %%

sns.countplot(df_sample['label'],
              order = df_sample['label'].value_counts().index);
'''
# %% Create the label and features

y = df['label']
X = df.drop(df.columns[0], axis=1)

# %% Split data to training and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

# %%

X_train = X_train / 255.0
X_test = X_test / 255.0

# %%

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)
print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_test.shape)

# %%

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)

# %% Model 1

model = Sequential()

model.add(Conv2D(filters = 64,
                 kernel_size = (3,3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# %% Model 2

model = Sequential()

model.add(Conv2D(filters = 128,
                 kernel_size = (3,3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64,
                 kernel_size = (3,3),
                 activation='relu',))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

# %% Model 3

model = Sequential()

model.add(Conv2D(filters = 128,
                 kernel_size = (3,3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64,
                 kernel_size = (3,3),
                 activation='relu',
                 strides = 2))
#model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32,
                 kernel_size = (3,3),
                 activation='relu',
                 strides = 2))
#model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

# %%

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

# %%

start_time = timeit.default_timer()

history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_split=0.3,
                    verbose=1)

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

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# %%

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# %%

rf = RandomForestClassifier(n_estimators = 100,
                             criterion = 'entropy',
                             n_jobs = -1,
                             random_state=0)

cv = StratifiedKFold(n_splits=5, random_state=0)

scores = cross_val_score(rf, X_train, y_train,
                             scoring = 'accuracy',
                             cv = cv)

# %%
# For 95% Confidence Interval, 2 * std
print("Accuracy: {0:.3f} (+/- {1:.3f})".format(scores.mean(),
                                                      scores.std()*2))

# %%

y_pred = cross_val_predict(rf, X_train, y_train, cv=cv)

print(pd.crosstab(y_train, y_pred,
                rownames = ['Actual'],
                colnames = ['Predicted'],
                margins = True))

# %%

rf.fit(X_train, y_train)

# %%

y_pred = rf.predict(X_test)

print(pd.crosstab(y_test, y_pred,
                rownames = ['Actual'],
                colnames = ['Predicted'],
                margins = True))

# %%

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)












