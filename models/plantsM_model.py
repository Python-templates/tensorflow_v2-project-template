import tensorflow as tf


class PlantsModel():
    def __init__(self, config):

        self.model = None
        self.config = config
        IMG_SHAPE = tuple(config["image_size"])+(3,)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')
        self.base_model.trainable = False
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),])

        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.prediction_layer = tf.keras.layers.Dense(5, activation='softmax')

        # self.prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax',
        #                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)

    def create_model(self):

        inputs = tf.keras.Input(shape=(*self.config["image_size"], 3))
        # x = data_augmentation(inputs)
        x = inputs
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x = self.base_model(x, training=False)
        x = self.global_average_layer(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        # outputs = x
        # outputs = prediction_layer(x)
        outputs = self.prediction_layer(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.summary()

        return self.model

    # @staticmethod
    # def loss_object(prediction, label):
    #     loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    #     return loss_object(prediction, label)


    def check_features_dimensions(self, train_data):
        image_batch = train_data.take(1)
        image_batch = next(iter(train_data))

        feature_batch = self.base_model(image_batch)
        feature_batch_average = self.global_average_layer(feature_batch)
        prediction_batch = self.prediction_layer(feature_batch_average)

        print(feature_batch.shape)
        print(feature_batch_average.shape)
        print(prediction_batch.shape)

    def save_model(self):
        from utils.utils import get_project_root
        import os
        root_path = get_project_root()
        name = 'final_' + self.config["checkpoint_name"]
        check_path = os.path.join(root_path, "experiments", self.config["exp_name"],
                                    "checkpoint", name)
        self.model.save(check_path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
