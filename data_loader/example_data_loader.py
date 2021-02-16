import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetGenerator:
    def __init__(self, config):
        # data = tfds.load("mnist", with_info=True)
        train_ds, test_ds = tfds.load('mnist', split=['train', 'test'])
        self.train_data = train_ds
        self.test_data = test_ds
        self.config = config
        assert isinstance(self.train_data, tf.data.Dataset)

    def __call__(self):
        self.preprocess()
        return self.train_data, self.test_data

    def preprocess(self):
        self.train_data = self.train_data.map(
            DatasetGenerator.convert_types
        ).batch(self.config.batch_size)
        self.test_data = self.test_data.map(
            DatasetGenerator.convert_types
        ).batch(self.config.batch_size)

    @staticmethod
    def convert_types(batch):
        image, label = batch.values()
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label



if __name__ == '__main__':
    from utils.config import process_config
    config = process_config("configs/example_config.json")
    train_data, test_data = DatasetGenerator(config)()
    pass






