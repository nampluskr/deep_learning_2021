""" [Tensorflow manual training codes]
- File name: tf_train_manual.py
- Last updated: 2021.5.23
"""

import tensorflow as tf

from copy import deepcopy
import matplotlib.pyplot as plt


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
        x = self.model(x, training=training)
        return x


class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'), # 28x28x16
            tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'), # 28x28x16
            tf.keras.layers.MaxPool2D((2,2)), # 14x14x16
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'), # 14x14x32
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'), # 14x14x32
            tf.keras.layers.MaxPool2D((2,2)), # 7x7x32
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'), # 7x7x64
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'), # 7x7x64
            tf.keras.layers.Flatten(), # 1568
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])

    def call(self, x, training=False, mask=None):
        x = self.model(x, training=training)
        return x


if __name__ == "__main__":

    ## Parameters
    seed = 111
    tf.random.set_seed(seed)

    n_epochs = 5
    batch_size = 512

    model_type = 'cnn'
    assert model_type in ('dnn', 'cnn')
    flatten = True if model_type == 'dnn' else False

    ## Dataset
    import mnist
    from sklearn.model_selection import train_test_split

    def preprocess(x, y, flatten):
        x = x.astype("float32")/255.
        y = y.astype("int64")
        x = x.reshape(-1, 784) if flatten else x.reshape(-1, 28, 28, 1)
        return x, y

    images, labels = mnist.train_images(), mnist.train_labels()
    x_train, x_valid, y_train, y_valid = train_test_split(images, labels,
        test_size=0.2, random_state=seed)

    x_train, y_train = preprocess(x_train, y_train, flatten=flatten)
    x_valid, y_valid = preprocess(x_valid, y_valid, flatten=flatten)

    ## Model
    model = SimpleDNN() if flatten else SimpleCNN()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optim = tf.keras.optimizers.Adam()

    ## Training
    best_loss, best_epoch, best_model = 1e12, 0, None
    history = {'train_loss':[], 'valid_loss':[]}

    for epoch in range(n_epochs):

        ## Training
        data_size = x_train.shape[0]
        steps_per_epoch = data_size // batch_size + (1 if data_size % batch_size else 0)
        indices = tf.random.shuffle(tf.range(data_size))
        x_train = tf.gather(x_train, indices, axis=0)
        y_train = tf.gather(y_train, indices, axis=0)

        train_batch_loss = []
        for i in range(steps_per_epoch):
            xi = x_train[i*batch_size:(i+1)*batch_size]
            yi = y_train[i*batch_size:(i+1)*batch_size]

            with tf.GradientTape() as tape:
                yi_hat = model(xi, training=True)
                loss = loss_fn(yi, yi_hat)

            grads = tape.gradient(loss, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))
            train_batch_loss.append(loss.numpy())

        ## Validation
        data_size = x_valid.shape[0]
        steps_per_epoch = data_size // batch_size + (1 if data_size % batch_size else 0)

        valid_batch_loss = []
        for i in range(steps_per_epoch):
            xi = x_valid[i*batch_size:(i+1)*batch_size]
            yi = y_valid[i*batch_size:(i+1)*batch_size]

            yi_hat = model(xi, training=False)
            loss = loss_fn(yi, yi_hat)
            valid_batch_loss.append(loss.numpy())

        # Print results
        train_loss = sum(train_batch_loss)/len(train_batch_loss)
        valid_loss = sum(valid_batch_loss)/len(valid_batch_loss)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_model = deepcopy(model.get_weights())

        template  = "Epoch [%3d/%3d] >> train_loss = %.4f, valid_loss = %.4f, "
        template += "lowest_loss = %.4f @epoch = %d"
        print(template % (epoch+1, n_epochs, train_loss, valid_loss,
                        best_loss, best_epoch))

    model.set_weights(best_model)

    plt.plot(history['train_loss'], 'ko-', lw=2, label="Train loss")
    plt.plot(history['valid_loss'], 'ro-', lw=2, label="Valid loss")
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    model.save("tf_model_munual_%s" % model_type) # folder_name
    ## new_model = tf.keras.models.load_model(folder_name)
