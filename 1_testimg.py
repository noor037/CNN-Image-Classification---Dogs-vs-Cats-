import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

train_datagen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, #20% as test set
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True)

train_generator=train_datagen.flow_from_directory(
    'Dataset',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='validation',  #This is nowour test set
    shuffle=True
)
import numpy as np

#Get one batch of imagesand labels 
images,labels=next(train_generator)

#plot a few argument images
plt.figure(figsize=(12,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i])
    plt.title(f"Label:{'Dog' if labels[i]==1 else 'Cat'}")
    plt.axis('off')
plt.suptitle("Augmented Training Images")
plt.tight_layout()
plt.show()
              
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2d(64,(3,3),actiavtion='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2d(128,(3,3),actiavtion='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,actiavtion='relu'),
    tf.keras.layers.Dense(1,actiavtion='sigmoid'),  #Binary Classification
    
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history=model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator)   #now using test generator
