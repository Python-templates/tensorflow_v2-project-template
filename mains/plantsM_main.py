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
from utils.utils import get_project_root
from utils.plot import display_training_accuracy
from data_loader.plantsM_data_loader import DataLoader
from models.plantsM_model import PlantsModel
from trainers.plantsM_train import PlantsTrainer
# %%
root_path = get_project_root()
data_dir = os.path.join(root_path, "images/PlantVillageM")
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

config_dir = os.path.join(root_path, "configs/plantsM_config.json")
config = process_config(config_dir)
# create_dirs([config.summary_dir, config.checkpoint_dir])

# %%
train_data, val_data, test_data = DataLoader(config, data_dir)()
model_obj = PlantsModel(config)
model_obj.check_features_dimensions(train_data)
model = model_obj.create_model()
trainer = PlantsTrainer(model, train_data, val_data, config)
history = trainer.train()

# %%
display_training_accuracy(history)
loss, accuracy = model.evaluate(test_data)
print('Test accuracy :', accuracy)

# %%
#Retrieve a batch of images from the test set
image_batch, label_batch = test_data.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)

# Apply a sigmoid since our model returns logits
probabilities = tf.nn.softmax(predictions)
predicted_indices = tf.argmax(probabilities, 1)
TARGET_LABELS = ["1", "2", "3", "4", "5"]
predicted_class = tf.gather(TARGET_LABELS, predicted_indices)



# %%
print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
#   plt.title(class_names[predictions[i]])
  plt.axis("off")

# %%


