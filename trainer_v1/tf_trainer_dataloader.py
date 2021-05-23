import argparse
import tensorflow as tf

import matplotlib.pyplot as plt
from copy import deepcopy


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--n_epochs', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=512)

    return args.parse_args()


def preprocess(x, y):
    x = x.astype("float32")/255.
    y = y.astype("int64")
    return x.reshape(x.shape[0], -1), y


def get_dataloaders(batch_size, valid_ratio=0.2, seed=0):
    import mnist
    from sklearn.model_selection import train_test_split

    images, labels = mnist.train_images(), mnist.train_labels()
    x_train, x_valid, y_train, y_valid = train_test_split(
            images, labels, test_size=valid_ratio, random_state=seed)

    x_train, y_train = preprocess(x_train, y_train)
    x_valid, y_valid = preprocess(x_valid, y_valid)

    train_dataloader = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(buffer_size=10000).batch(batch_size)
    valid_dataloader = tf.data.Dataset.from_tensor_slices(
        (x_valid, y_valid)).batch(batch_size)

    return train_dataloader, valid_dataloader


class Trainer():
    def __init__(self, model, loss_fn, optim):
        self.model, self.loss_fn, self.optim = model, loss_fn, optim

    def train(self, dataloader):

        batch_loss = []
        for xi, yi in dataloader:
            with tf.GradientTape() as tape:
                yi_hat = self.model(xi, training=True)
                loss = self.loss_fn(yi, yi_hat)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
            batch_loss.append(loss)

        return sum(batch_loss)/len(batch_loss)

    def evaluate(self, dataloader):

        batch_loss = []
        for xi, yi in dataloader:
            yi_hat = self.model(xi, training=False)
            loss = self.loss_fn(yi, yi_hat)
            batch_loss.append(loss)

        return sum(batch_loss)/len(batch_loss)

    def fit(self, train_dataloader, valid_dataloader, n_epochs):
        lowest_loss = 1e12
        best_epoch, best_model = 0, None

        self.history = {'train_loss':[], 'valid_loss':[]}
        for epoch in range(n_epochs):
            train_loss = self.train(train_dataloader)
            valid_loss = self.evaluate(valid_dataloader)

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_epoch = epoch + 1
                best_model = deepcopy(self.model.get_weights())

            template  = "Epoch [%3d/%3d] >>> train_loss = %.2e, valid_loss = %.2e, "
            template += "lowest_loss = %.2e @epoch = %d"
            print(template % (epoch+1, n_epochs, train_loss,
                            valid_loss, lowest_loss, best_epoch))

        self.model.set_weights(best_model)
        return best_epoch, lowest_loss

    def plot_history(self):
        fig, ax = plt.subplots()
        ax.plot(self.history['train_loss'], 'ko-', lw=2, label="Train loss")
        ax.plot(self.history['valid_loss'], 'ro-', lw=2, label="Valid loss")
        ax.legend(fontsize=12)
        ax.grid()
        plt.show()


if __name__ == "__main__":

    seed = 111
    tf.random.set_seed(seed)

    ## Hyper parameters
    args = get_args()
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    ## Load dataset
    train_dataloader, valid_dataloader = get_dataloaders(batch_size, seed=seed)

    ## Model / Loss function / Optimizer / Trainner
    input_size = 784
    output_size = 10

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='softmax'),
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optim = tf.keras.optimizers.Adam()

    ## Training with Keras
    # model.compile(loss=loss_fn, optimizer=optim)
    # model.fit(train_dataloader, validation_data=valid_dataloader,
    #     epochs=n_epochs, batch_size=batch_size)

    ## Training
    print("\n[Training with validation]")
    trainer = Trainer(model, loss_fn, optim)
    epoch, loss = trainer.fit(train_dataloader, valid_dataloader, n_epochs)
    model_name = "tf_model.valid_loss-%.4f.epoch-%d.h5" % (loss, epoch)
    trainer.model.save(model_name)

    trainer.plot_history()
    ## new_model = tf.keras.models.load_model(model_name)