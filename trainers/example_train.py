import tensorflow as tf
from tqdm import tqdm
import numpy as np

class ExampleTrainer:
    def __init__(self, model, train_data, test_data, config):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.optimizer = tf.optimizers.Adam(learning_rate=self.config.learning_rate)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.model.loss_object(y, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(y, predictions)


    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        for _, (batch_x, batch_y) in zip(loop, self.train_data):
            self.train_step(batch_x, batch_y)


    @tf.function
    def test_step(self, x, y):
        predictions = self.model(x, training=False)
        t_loss = self.model.loss_object(y, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(y, predictions)


    def test_epoch(self):
        for (batch_x, batch_y) in self.test_data:
            self.test_step(batch_x, batch_y)


    def train(self):
        template_train = "Epoch: {} | Train Loss: {}, Train Accuracy: {}"
        template_test =  "         | Test Loss:  {}, Test Accuracy: {}"
        epochs = self.config.num_epochs
        for epoch in range(1, epochs+1):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            self.train_epoch()
            self.test_epoch()

            if epoch % self.config.verbose_epochs == 0:
                # TODO: Using logger instead train_nt function
                print(template_train.format(epoch, self.train_loss.result(), self.train_accuracy.result()*100))
                print(template_test.format(self.test_loss.result(), self.test_accuracy.result()*100))



