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

        return self.train_loss.result(), self.train_accuracy.result()

    @tf.function
    def test_step(self, x, y):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(x, training=False)
        t_loss = self.model.loss_object(y, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(y, predictions)

        return self.test_loss.result(), self.test_accuracy.result()

    def test_epoch(self):
        losses = []
        accs = []
        for (batch_x, batch_y) in self.test_data:
            test_loss, test_accuracy = self.test_step(batch_x, batch_y)
            losses.append(test_loss)
            accs.append(test_accuracy)
        loss = np.mean(losses)
        acc = np.mean(accs)
        return loss, acc

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _, (batch_x, batch_y) in zip(loop, self.train_data):
            train_loss, train_accuracy = self.train_step(batch_x, batch_y)
            losses.append(train_loss)
            accs.append(train_accuracy)
        loss = np.mean(losses)
        acc = np.mean(accs)
        return loss, acc

    def train(self):
        template_train = "Epoch: {} | Train Loss: {}, Train Accuracy: {}"
        template_test =  "         | Test Loss:  {}, Test Accuracy: {}"
        epochs = self.config.num_epochs
        for epoch in range(1, epochs+1):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.test_epoch()
            if epoch % self.config.verbose_epochs == 0:
                # TODO: Using logger instead train_nt function
                print(template_train.format(epoch, train_loss, train_acc))
                print(template_test.format(test_loss, test_acc))
