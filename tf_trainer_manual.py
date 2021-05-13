import tensorflow as tf


def load_train_valid_data(valid_ratio=0.2, show_info=False):
    import mnist
    from sklearn.model_selection import train_test_split

    images, labels = mnist.train_images(), mnist.train_labels()
    x_train, x_valid, y_train, y_valid = train_test_split(
            images, labels, test_size=valid_ratio, random_state=111)

    if show_info:
        print("Train images:", type(x_train), x_train.shape, x_train.dtype)
        print("Train labels:", type(y_train), y_train.shape, y_train.dtype)
        print("Valid images:", type(x_valid), x_valid.shape, x_valid.dtype)
        print("Valid labels:", type(y_valid), y_valid.shape, y_valid.dtype)

    return (x_train, y_train), (x_valid, y_valid)


def load_test_data():
    import mnist

    return mnist.test_images(), mnist.test_labels()


def preprocess(x, y):
    x = x.reshape(-1, 784).astype("float32")/255.
    return x, y


if __name__ == "__main__":

    ## Hyper parameters
    n_epochs = 10
    batch_size = 512


    ## Load dataset
    (x_train, y_train), (x_valid, y_valid) = load_train_valid_data()
    x_train, y_train = preprocess(x_train, y_train)
    x_valid, y_valid = preprocess(x_valid, y_valid)

    print("Train images:", type(x_train), x_train.shape, x_train.dtype)
    print("Train labels:", type(y_train), y_train.shape, y_train.dtype)
    print("Valid images:", type(x_valid), x_valid.shape, x_valid.dtype)
    print("Valid labels:", type(y_valid), y_valid.shape, y_valid.dtype)


    ## Model / Loss function / Optimizer
    input_size = x_train.shape[0]
    output_size = int(y_train.max()) + 1

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_size, activation='softmax'),
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optim = tf.keras.optimizers.Adam()

    # model.compile(loss=loss_fn, optimizer=optim)
    # model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
    #     epochs=n_epochs, batch_size=batch_size)


    print("\n[Training with validation]")
    history = {'train_loss':[], 'valid_loss':[]}
    for epoch in range(n_epochs):

        ## Training
        data_size = x_train.shape[0]
        steps_per_epoch = data_size // batch_size + (1 if data_size % batch_size else 0)
        indices = tf.random.shuffle(tf.range(data_size))
        x_train = tf.gather(x_train, indices, axis=0)
        y_train = tf.gather(y_train, indices, axis=0)

        batch_loss = []
        for i in range(steps_per_epoch):
            xi = x_train[i*batch_size:(i+1)*batch_size]
            yi = y_train[i*batch_size:(i+1)*batch_size]

            with tf.GradientTape() as tape:
                yi_hat = model(xi, training=True)
                loss = loss_fn(yi, yi_hat)

            grads = tape.gradient(loss, model.trainable_weights)
            optim.apply_gradients(zip(grads, model.trainable_weights))
            batch_loss.append(float(loss))

        train_loss = sum(batch_loss)/len(batch_loss)

        ## Validation
        data_size = x_valid.shape[0]
        steps_per_epoch = data_size // batch_size + (1 if data_size % batch_size else 0)

        batch_loss = []
        for i in range(steps_per_epoch):
            xi = x_valid[i*batch_size:(i+1)*batch_size]
            yi = y_valid[i*batch_size:(i+1)*batch_size]

            yi_hat = model(xi, training=False)
            loss = loss_fn(yi, yi_hat)
            batch_loss.append(float(loss))

        valid_loss = sum(batch_loss)/len(batch_loss)

        ## Print log
        print("Epoch [%3d/%3d] >>> train_loss = %.2e, valid_loss = %.2e"
            % (epoch+1, n_epochs, train_loss, valid_loss))

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)


    ## Plot results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(range(0, n_epochs), history['train_loss'], 'ko-', lw=2, label="Train loss")
    ax.plot(range(0, n_epochs), history['valid_loss'], 'ro-', lw=2, label="Valid loss")
    ax.legend(fontsize=12)
    ax.grid()
    fig.tight_layout()
    plt.show()
