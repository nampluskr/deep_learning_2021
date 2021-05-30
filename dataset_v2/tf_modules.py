import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from copy import deepcopy


def tf_dataloader(data, batch_size, training=False):
    def preprocess(x, y):
        x = tf.cast(x, dtype=tf.float32)/255.
        y = tf.cast(y, dtype=tf.int64)
        return x, y

    dataloader = tf.data.Dataset.from_tensor_slices(data).map(preprocess)
    if training:
        return dataloader.shuffle(buffer_size=10000).batch(batch_size)
    else:
        return dataloader.batch(batch_size)


def keras_dataloader(data, batch_size, training=False):
    def preprocess(x, y):
        x = x.astype('float32')/255.
        y = y.astype('int64')
        return x, y

    x, y = data
    if training:
        datagenerator = ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # horizontal_flip=True,
        )
        datagenerator.fit(x)
    else:
        datagenerator = ImageDataGenerator()

    x, y = preprocess(x, y)
    return datagenerator.flow(x, y, batch_size=batch_size, shuffle=training)


class ConvClassifier(tf.keras.Model):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

        self.convblocks = models.Sequential([ # (bs, 32, 32,   3)
            ConvBlock(32),                    # (bs, 16, 16,  32)
            ConvBlock(64),                    # (bs,  8,  8,  64)
            ConvBlock(128),                   # (bs,  4,  4, 128)
            ConvBlock(256),                   # (bs,  2,  2, 256)
            ConvBlock(512),                   # (bs,  1,  1, 512)
        ])

        self.fclayers = models.Sequential([
            layers.Flatten(),
            layers.Dense(50),
            layers.Activation(tf.keras.activations.relu),
            layers.BatchNormalization(),
            layers.Dense(self.output_size),
            layers.Activation(tf.keras.activations.softmax),
        ])

    def call(self, x, training=False):
        assert x.ndim > 2:
        if x.ndim == 3:
            x = x[..., tf.newaxis]

        x = self.convblocks(x, training=training)
        y = self.fclayers(x, training=training)
        return y


class ConvBlock(tf.keras.Model):
    def __init__(self, out_channels):
        super().__init__()
        self.model = models.Sequential([
            layers.Conv2D(out_channels, (3, 3), padding='same'),
            layers.Activation(tf.keras.activations.relu),
            layers.BatchNormalization(),
            layers.Conv2D(out_channels, (3, 3), padding='same', strides=2),
            layers.Activation(tf.keras.activations.relu),
            layers.BatchNormalization(),
        ])

    def call(self, x, training=False):
        x = self.model(x, training=training)
        return x


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


class KerasTrainer(Trainer):
    def __init__(self, model, optim, loss_fn):
        super().__init__(model, optim, loss_fn)

    def fit(self, train_loader, valid_loader, n_epochs):
        self.model.compile(optimizer=self.optim, loss=self.loss_fn)
        self.model.fit(train_loader, validation_data=valid_loader,
                epochs=n_epochs)


class ManualTrainer(Trainer):

    @tf.function
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.model(x, training=True)
            loss = self.loss_fn(y, y_hat)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def train(self, dataloader):
        batch_loss = []
        for i, (xi, yi) in enumerate(dataloader):
            if i == len(dataloader):
                break
            loss = self._train_step(xi, yi)
            batch_loss.append(loss.numpy())

            if i+1 in [round(len(dataloader)*(i+1)/10) for i in range(10)]:
                print("Training %3.f%% [%3d/%3d] train_loss: %.4f" % (
                    round((i+1)/len(dataloader)*100), 
                    i+1, len(dataloader), loss.numpy()))

        return sum(batch_loss)/len(batch_loss)

    @tf.function
    def _evaluate_step(self, x, y):
        y_hat = self.model(x, training=False)
        loss = self.loss_fn(y, y_hat)
        return loss

    def evaluate(self, dataloader):
        batch_loss = []
        for i, (xi, yi) in enumerate(dataloader):
            if i == len(dataloader):
                break
            loss = self._evaluate_step(xi, yi)
            batch_loss.append(loss.numpy())

        return sum(batch_loss)/len(batch_loss)

    def fit(self, train_loader, valid_loader, n_epochs):
        for epoch in range(n_epochs):
            train_loss = self.train(train_loader)
            valid_loss = self.evaluate(valid_loader)

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_epoch = epoch + 1
                self.best_model = deepcopy(self.model.get_weights())

            template  = "Epoch [%3d/%3d] >>> train_loss = %.4f, valid_loss = %.4f, "
            template += "lowest_loss = %.4f @epoch = %d"
            print(template % (epoch+1, n_epochs, train_loss,
                            valid_loss, self.best_loss, self.best_epoch))

        self.model.set_weights(self.best_model)


# tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
def accuracy(y, y_hat):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_hat, 1), y), tf.float32))


def average(func, model, data, batch_size=0):
    batch_outputs = []
    if len(data) > 2:
        for i, (xi, yi) in enumerate(data):
            if i == len(data):
                break
            yi_hat = model(xi)
            output = func(yi, yi_hat)
            batch_outputs.append(output.numpy())

    elif batch_size > 0:
        x, y = data
        data_size = x.shape[0]
        steps_per_epoch = data_size // batch_size + (
                1 if data_size % batch_size else 0)

        for i in range(steps_per_epoch):
            xi = x[i*batch_size:(i+1)*batch_size]
            yi = y[i*batch_size:(i+1)*batch_size]

            yi_hat = model(xi)
            output = func(yi, yi_hat)
            batch_outputs.append(output.numpy())
    else:
        x, y = data
        y_hat = model(x)
        output = func(y, y_hat)
        batch_outputs.append(output.numpy())

    return sum(batch_outputs)/len(batch_outputs)