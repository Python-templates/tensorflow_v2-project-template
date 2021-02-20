import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
from utils.utils import get_project_root

class PlantsTrainer:
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.loss= tf.keras.losses.CategoricalCrossentropy()
        self.metrics=['accuracy']

        root_path = get_project_root()
        check_path = os.path.join(root_path, "experiments", self.config["exp_name"],
                                    "checkpoint", self.config["checkpoint_name"])
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path,
                                                        verbose=1, save_best_only=True)
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

    def train(self):

        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.history = self.model.fit(self.train_data,
                            epochs=self.config["num_epochs"],
                            callbacks = [self.checkpoint, self.callback],
                            validation_data=self.val_data)
        return self.history

    def train_fine(self):
        #fine tuning
        self.model.trainable = True
        print("Number of layers in the model: ", len(self.model.layers))

        # set base_model ([3]) trainable parameters
        for layer in self.model.layers[3].layers[:100]:
            layer.trainable =  False

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate/10)
        self.model.summary()
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)

        fine_tune_epochs = 5
        total_epochs =  self.config["num_epochs"] + fine_tune_epochs

        history_fine = self.model.fit(self.train_data,
                                epochs=total_epochs,
                                initial_epoch=self.history.epoch[-1],
                                callbacks = [self.checkpoint, self.callback],
                                validation_data=self.val_data)

        return history_fine
