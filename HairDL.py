# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:06:59 2024

@author: pchri
"""

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk(r'C:\Users\pchri\Downloads'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.callbacks import ReduceLROnPlateau

def get_images(directory):
    images = []
    
    for filename in os.listdir(directory):
        try:
            img = Image.open(os.path.join(directory, filename))
            img = img.resize((128, 128))
            img = img.convert('RGB')
            img = np.array(img) / 255.0
            images.append(img)
        except OSError as e:
            print(f"Error loading {os.path.join(directory, filename)}: {e}")
            continue
    return images

curly = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\curly')
dreadlocks = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\dreadlocks')
kinky = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\kinky')
straight = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\Straight')
wavy = get_images(r'C:\Users\pchri\Downloads\archive (5)\data\Wavy')

len(curly)
len(dreadlocks)
len(kinky)
len(straight)
len(wavy)

fig, ax = plt.subplots(1, 10, figsize=(20, 10))

ax[0].imshow(curly[0])
ax[1].imshow(curly[1])
ax[2].imshow(dreadlocks[0])
ax[3].imshow(dreadlocks[1])
ax[4].imshow(kinky[0])
ax[5].imshow(kinky[1])
ax[6].imshow(straight[0])
ax[7].imshow(straight[1])
ax[8].imshow(wavy[0])
ax[9].imshow(wavy[1])
ax[0].set_title('curly')
ax[1].set_title('curly')
ax[2].set_title('dreadlocks')
ax[3].set_title('dreadlocks')
ax[4].set_title('kinky')
ax[5].set_title('kinky')
ax[6].set_title('straight')
ax[7].set_title('straight')
ax[8].set_title('wavy')
ax[9].set_title('wavy')
                
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.show()

train1_images = np.concatenate((curly, dreadlocks))
train1_labels = np.concatenate((np.ones(len(curly)), np.zeros(len(dreadlocks))))

train1_ds = tf.data.Dataset.from_tensor_slices((train1_images, train1_labels))
batch_size = 32

train1 = train1_ds.shuffle(buffer_size=len(train1_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train1, validation_data = train1, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: curly v dreadlocks')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: curly v dreadlocks')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train1,verbose=2)

pred1 = model.predict(train1)
pred1_cnn = (pred1 > 0.5).astype("int32")

cm1 = confusion_matrix(train1_labels, pred1_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm1, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: curly v dreadlocks')
plt.show()

accuracy1 = accuracy_score(train1_labels, pred1_cnn)
model_report1 = classification_report(train1_labels, pred1_cnn)
print(f'Model accuracy: {round(accuracy1,4)}')
print('Classification Report: curly v dreadlocks')
print(f'{model_report1}')

train2_images = np.concatenate((curly, kinky))
train2_labels = np.concatenate((np.ones(len(curly)), np.zeros(len(kinky))))

train2_ds = tf.data.Dataset.from_tensor_slices((train2_images, train2_labels))
batch_size = 32

train2 = train2_ds.shuffle(buffer_size=len(train2_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train2, validation_data = train2, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: curly v kinky')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: curly v kinky')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train2,verbose=2)

pred2 = model.predict(train2)
pred2_cnn = (pred2 > 0.5).astype("int32")

cm2 = confusion_matrix(train2_labels, pred2_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm2, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: curly v kinky')
plt.show()

accuracy2 = accuracy_score(train2_labels, pred2_cnn)
model_report2 = classification_report(train2_labels, pred2_cnn)
print(f'Model accuracy: {round(accuracy2,4)}')
print('Classification Report: curly v kinky')
print(f'{model_report2}')

train3_images = np.concatenate((curly, straight))
train3_labels = np.concatenate((np.ones(len(curly)), np.zeros(len(straight))))

train3_ds = tf.data.Dataset.from_tensor_slices((train3_images, train3_labels))
batch_size = 32

train3 = train3_ds.shuffle(buffer_size=len(train3_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train3, validation_data = train3, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: curly v straight')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: curly v straight')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train3,verbose=2)

pred3 = model.predict(train3)
pred3_cnn = (pred3 > 0.5).astype("int32")

cm3 = confusion_matrix(train3_labels, pred3_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm3, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: curly v straight')
plt.show()

accuracy3 = accuracy_score(train3_labels, pred3_cnn)
model_report3 = classification_report(train3_labels, pred3_cnn)
print(f'Model accuracy: {round(accuracy3,4)}')
print('Classification Report: curly v straight')
print(f'{model_report3}')

train4_images = np.concatenate((curly, wavy))
train4_labels = np.concatenate((np.ones(len(curly)), np.zeros(len(wavy))))

train4_ds = tf.data.Dataset.from_tensor_slices((train4_images, train4_labels))
batch_size = 32

train4 = train4_ds.shuffle(buffer_size=len(train4_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train4, validation_data = train4, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: curly v wavy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: curly v wavy')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train4,verbose=2)

pred4 = model.predict(train4)
pred4_cnn = (pred4 > 0.5).astype("int32")

cm4 = confusion_matrix(train4_labels, pred4_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm4, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: curly v wavy')
plt.show()

accuracy4 = accuracy_score(train4_labels, pred4_cnn)
model_report4 = classification_report(train4_labels, pred4_cnn)
print(f'Model accuracy: {round(accuracy4,4)}')
print('Classification Report: curly v wavy')
print(f'{model_report4}')

train5_images = np.concatenate((dreadlocks, kinky))
train5_labels = np.concatenate((np.ones(len(dreadlocks)), np.zeros(len(kinky))))

train5_ds = tf.data.Dataset.from_tensor_slices((train5_images, train5_labels))
batch_size = 32

train5 = train5_ds.shuffle(buffer_size=len(train5_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train5, validation_data = train5, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: dreadlocks v kinky')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: dreadlocks v kinky')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train5,verbose=2)

pred5 = model.predict(train5)
pred5_cnn = (pred5 > 0.5).astype("int32")

cm5 = confusion_matrix(train5_labels, pred5_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm5, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: dreadlocks v kinky')
plt.show()

accuracy5 = accuracy_score(train5_labels, pred5_cnn)
model_report5 = classification_report(train5_labels, pred5_cnn)
print(f'Model accuracy: {round(accuracy5,4)}')
print('Classification Report: dreadlocks v kinky')
print(f'{model_report5}')

train6_images = np.concatenate((dreadlocks, straight))
train6_labels = np.concatenate((np.ones(len(dreadlocks)), np.zeros(len(straight))))

train6_ds = tf.data.Dataset.from_tensor_slices((train6_images, train6_labels))
batch_size = 32

train6 = train6_ds.shuffle(buffer_size=len(train6_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train6, validation_data = train6, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: dreadlocks v straight')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: dreadlocks v straight')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train6,verbose=2)

pred6 = model.predict(train6)
pred6_cnn = (pred6 > 0.5).astype("int32")

cm6 = confusion_matrix(train6_labels, pred6_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm6, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: dreadlocks v straight')
plt.show()

accuracy6 = accuracy_score(train6_labels, pred6_cnn)
model_report6 = classification_report(train6_labels, pred6_cnn)
print(f'Model accuracy: {round(accuracy6,4)}')
print('Classification Report: dreadlocks v straight')
print(f'{model_report6}')

train7_images = np.concatenate((dreadlocks, wavy))
train7_labels = np.concatenate((np.ones(len(dreadlocks)), np.zeros(len(wavy))))

train7_ds = tf.data.Dataset.from_tensor_slices((train7_images, train7_labels))
batch_size = 32

train7 = train7_ds.shuffle(buffer_size=len(train7_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train7, validation_data = train7, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: dreadlocks v wavy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: dreadlocks v wavy')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train7,verbose=2)

pred7 = model.predict(train7)
pred7_cnn = (pred7 > 0.5).astype("int32")

cm7 = confusion_matrix(train7_labels, pred7_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm7, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: dreadlocks v wavy')
plt.show()

accuracy7 = accuracy_score(train7_labels, pred7_cnn)
model_report7 = classification_report(train7_labels, pred7_cnn)
print(f'Model accuracy: {round(accuracy7,4)}')
print('Classification Report: dreadlocks v wavy')
print(f'{model_report7}')

train8_images = np.concatenate((kinky, straight))
train8_labels = np.concatenate((np.ones(len(kinky)), np.zeros(len(straight))))

train8_ds = tf.data.Dataset.from_tensor_slices((train8_images, train8_labels))
batch_size = 32

train8 = train8_ds.shuffle(buffer_size=len(train8_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train8, validation_data = train8, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: kinky v straight')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: kinky v straight')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train8,verbose=2)

pred8 = model.predict(train8)
pred8_cnn = (pred8 > 0.5).astype("int32")

cm8 = confusion_matrix(train8_labels, pred8_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm8, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: kinky v straight')
plt.show()

accuracy8 = accuracy_score(train8_labels, pred8_cnn)
model_report8 = classification_report(train8_labels, pred8_cnn)
print(f'Model accuracy: {round(accuracy8,4)}')
print('Classification Report: kinky v straight')
print(f'{model_report8}')

train9_images = np.concatenate((kinky, wavy))
train9_labels = np.concatenate((np.ones(len(kinky)), np.zeros(len(wavy))))

train9_ds = tf.data.Dataset.from_tensor_slices((train9_images, train9_labels))
batch_size = 32

train9 = train9_ds.shuffle(buffer_size=len(train9_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train9, validation_data = train9, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: kinky v wavy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: kinky v wavy')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train9,verbose=2)

pred9 = model.predict(train9)
pred9_cnn = (pred9 > 0.5).astype("int32")

cm9 = confusion_matrix(train9_labels, pred9_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm9, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: kinky v wavy')
plt.show()

accuracy9 = accuracy_score(train9_labels, pred9_cnn)
model_report9 = classification_report(train9_labels, pred9_cnn)
print(f'Model accuracy: {round(accuracy9,4)}')
print('Classification Report: kinky v wavy')
print(f'{model_report9}')

train10_images = np.concatenate((straight, wavy))
train10_labels = np.concatenate((np.ones(len(straight)), np.zeros(len(wavy))))

train10_ds = tf.data.Dataset.from_tensor_slices((train10_images, train10_labels))
batch_size = 32

train10 = train10_ds.shuffle(buffer_size=len(train10_images)).batch(batch_size)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128, 128, 3), padding='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train10, validation_data = train10, epochs = 10, verbose = 2)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Accuracy', 'Val Accuracy'], loc = 'upper right')
plt.title('Accuracy: straight v wavy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'Val Loss'], loc = 'upper right')
plt.title('Loss: straight v wavy')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

model.evaluate(train10,verbose=2)

pred10 = model.predict(train10)
pred10_cnn = (pred10 > 0.5).astype("int32")

cm10 = confusion_matrix(train10_labels, pred10_cnn)

plt.figure(figsize=(4, 4))
sns.heatmap(cm10, annot=True, fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: straight v wavy')
plt.show()

accuracy10 = accuracy_score(train10_labels, pred10_cnn)
model_report10 = classification_report(train10_labels, pred10_cnn)
print(f'Model accuracy: {round(accuracy10,4)}')
print('Classification Report: straight v wavy')
print(f'{model_report10}')