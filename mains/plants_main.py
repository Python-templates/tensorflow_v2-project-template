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



# %%
