# %%
from models.plants_model import PlantsModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from os.path import dirname, abspath
import pathlib
import cv2
import PIL
from utils.config import process_config
from utils.dirs import create_dirs

from data_loader.plants_data_loader import DataLoader
from models.plants_model import PlantsModel
from trainers.plants_train import PlantsTrainer


data_dir = os.path.join("./images/PlantVillage")
# data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

config_dir = os.path.join("./configs/plants_config.json")
# config_dir = pathlib.Path(config_dir)
config = process_config(config_dir)
create_dirs([config.summary_dir, config.checkpoint_dir])

train_data, val_data, test_data = DataLoader(config, data_dir)()

model_obj = PlantsModel(config)
model_obj.check_features_dimensions(train_data)
model = model_obj.create_model()
trainer = PlantsTrainer(model, train_data, val_data, config)
history = trainer.train()

# %%

loss0, accuracy0 = model.evaluate(val_data)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# %%
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

# %%
#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
#   plt.title(class_names[predictions[i]])
  plt.axis("off")

# %%
