import tensorflow as tf


class DataLoader:
    def __init__(self, config, data_dir):
        self.config = config
        self.train_data = tf.keras.preprocessing.image_dataset_from_directory(
                                    data_dir,
                                    validation_split=config["validation_split"],
                                    subset="training",
                                    # label_mode='categorical',
                                    seed=config["seed"],
                                    image_size=config["image_size"],
                                    batch_size=config["batch_size"])

        self.val_data = tf.keras.preprocessing.image_dataset_from_directory(
                                    data_dir,
                                    validation_split=config["validation_split"],
                                    subset="validation",
                                    # label_mode='categorical',
                                    seed=config["seed"],
                                    image_size=config["image_size"],
                                    batch_size=config["batch_size"])

        validation_batches = tf.data.experimental.cardinality(self.val_data)
        self.test_data = self.val_data.take(validation_batches // 5)
        self.val_data = self.val_data.skip(validation_batches // 5)

        # print('Number of validation batches: %d' % tf.data.experimental.cardinality(self.validation_data))
        # print('Number of test batches: %d' % tf.data.experimental.cardinality(self.test_dataset))

        # self.train_data = self.train_data.prefetch(tf.data.AUTOTUNE)
        # self.val_data = self.val_data.prefetch(tf.data.AUTOTUNE)
        # self.test_data = self.test_data.prefetch(tf.data.AUTOTUNE)
        # num_classes = len(self.train_data.class_names)

    def __call__(self):
        # self.preprocess()
        return self.train_data, self.val_data, self.test_data

    def preprocess(self):
        self.train_data = self.train_data.map(
            DataLoader.convert_types
        ).batch(self.config.batch_size)
        self.test_data = self.test_data.map(
            DataLoader.convert_types
        ).batch(self.config.batch_size)

    @staticmethod
    def convert_types(batch):
        image, label = batch.values()
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label



if __name__ == '__main__':
    from utils.config import process_config
    import os
    from os.path import dirname, abspath
    import pathlib

    data_dir = os.path.join(dirname(dirname(abspath(__file__))), 'images/PlantVillage/')
    data_dir = pathlib.Path(data_dir)
    config = process_config("configs/plants_config.json")
    train_data, validation_data, test_data = DataLoader(config, data_dir)()
    pass







