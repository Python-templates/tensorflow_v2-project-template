import tensorflow as tf


class PlantsModel():
    def __init__(self, config):

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
        self.prediction_layer = tf.keras.layers.Dense(1)


    def create_model(self):

        inputs = tf.keras.Input(shape=(*self.config["image_size"], 3))
        # x = data_augmentation(inputs)
        x = inputs
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x = self.base_model(x, training=False)
        x = self.global_average_layer(x)
        x = tf.keras.layers.Dropout(rate=0.2)(x)
        # x = tf.keras.layers.Dense(num_classes, activation='softmax',
        #                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        # outputs = x
        # outputs = prediction_layer(x)
        outputs = self.prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)
        model.summary()

        return model

    # @staticmethod
    # def loss_object(prediction, label):
    #     loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    #     return loss_object(prediction, label)


    def check_features_dimensions(self, train_data):
        image_batch, label_batch = next(iter(train_data))

        feature_batch = self.base_model(image_batch)
        feature_batch_average = self.global_average_layer(feature_batch)
        prediction_batch = self.prediction_layer(feature_batch_average)

        print(feature_batch.shape)
        print(feature_batch_average.shape)
        print(prediction_batch.shape)
