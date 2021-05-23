""" [Tensorflow user-defined modules and functions]
- File name: tf_modules.py
- Last updated: 2021.5.23
"""

import tensorflow as tf

from copy import deepcopy
import matplotlib.pyplot as plt


def get_dataloaders(batch_size, seed, flatten=False):
    import mnist
    from sklearn.model_selection import train_test_split

    def preprocess(x, y, flatten):
        x = x.astype("float32")/255.
        y = y.astype("int64")
        x = x.reshape(-1, 28*28) if flatten else x.reshape(-1, 28, 28, 1)
        return x, y

    images, labels = mnist.train_images(), mnist.train_labels()
    x_train, x_valid, y_train, y_valid = train_test_split(images, labels,
        test_size=0.2, random_state=seed)

    x_train, y_train = preprocess(x_train, y_train, flatten)
    x_valid, y_valid = preprocess(x_valid, y_valid, flatten)

    train_loader = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(buffer_size=1024).batch(batch_size)

    valid_loader = tf.data.Dataset.from_tensor_slices(
        (x_valid, y_valid)).batch(batch_size)

    return train_loader, valid_loader


class Trainer():
    def __init__(self, model, optim, loss_fn):
        self.model, self.loss_fn, self.optim = model, loss_fn, optim
        self.best_loss, self.best_epoch, self.best_model = 1e12, 0, None
        self.history = {'train_loss':[], 'valid_loss':[]}

    def plot_history(self):
        plt.plot(self.history['train_loss'], 'ko-', lw=2, label="Train loss")
        plt.plot(self.history['valid_loss'], 'ro-', lw=2, label="Valid loss")
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()


class ManualTrainer(Trainer):

    def train(self, dataloader):
        batch_loss = []
        for xi, yi in dataloader:
            with tf.GradientTape() as tape:
                yi_hat = self.model(xi, training=True)
                loss = self.loss_fn(yi, yi_hat)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
            batch_loss.append(loss.numpy())

        return sum(batch_loss)/len(batch_loss)

    def evaluate(self, dataloader):
        batch_loss = []
        for xi, yi in dataloader:
            yi_hat = self.model(xi, training=False)
            loss = self.loss_fn(yi, yi_hat)
            batch_loss.append(loss.numpy())

        return sum(batch_loss)/len(batch_loss)

    def fit(self, train_dataloader, valid_dataloader, n_epochs):
        for epoch in range(n_epochs):
            train_loss = self.train(train_dataloader)
            valid_loss = self.evaluate(valid_dataloader)

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_epoch = epoch + 1
                self.best_model = deepcopy(self.model.get_weights())

            template  = "Epoch [%3d/%3d] >>> train_loss = %.2e, valid_loss = %.2e, "
            template += "lowest_loss = %.2e @epoch = %d"
            print(template % (epoch+1, n_epochs, train_loss,
                            valid_loss, self.best_loss, self.best_epoch))

        self.model.set_weights(self.best_model)


class SimpleDNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
            ])

    def call(self, x, training=False, mask=None):
        x = self.model(x)
        return x


class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'), # 28x28x16
            tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'), # 28x28x16
            tf.keras.layers.MaxPool2D((2,2)),                                      # 14x14x16
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'), # 14x14x32
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'), # 14x14x32
            tf.keras.layers.MaxPool2D((2,2)),                                      # 7x7x32
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'), # 7x7x64
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'), # 7x7x64
            tf.keras.layers.Flatten(),                                             # 1568
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

    def call(self, x, training=False, mask=None):
        x = self.model(x, training=training)
        return x


if __name__ == "__main__":

    pass