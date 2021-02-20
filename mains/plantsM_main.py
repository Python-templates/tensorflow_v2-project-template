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

from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_project_root
from utils.plot import display_training_accuracy
from utils.plot import plot_confusion_matrix_from_dataset
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
create_dirs([config.summary_dir, config.checkpoint_dir])

train_data, val_data, test_data = DataLoader(config, data_dir)()
# # %%
# # Training
# model_obj = PlantsModel(config)
# model_obj.check_features_dimensions(train_data)
# model = model_obj.create_model()
# trainer = PlantsTrainer(model, train_data, val_data, config)
# history = trainer.train()

# display_training_accuracy(history)
# loss, accuracy = model.evaluate(test_data)
# print('Test accuracy :', accuracy)

# # %%
# # Fine tuning
# history_fine = trainer.train_fine()
# display_training_accuracy(history_fine)
# loss, accuracy = model.evaluate(test_data)
# print('Test accuracy :', accuracy)

# %%
model_obj_load = PlantsModel(config)
path_load = "C:\\Users\\janezla\\Documents\\_programs\\tensorflow_v2-project-template\\experiments\\plantsM\\checkpoint\\plant_modelM.h5"
model = model_obj_load.load_model(path_load)

loss, accuracy = model.evaluate(test_data)
print('Test accuracy :', accuracy)

# %%
plot_confusion_matrix_from_dataset(test_data, model, ["1", "2", "3", "4", "5"])


# %%
