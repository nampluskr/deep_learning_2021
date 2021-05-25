""" [Tensorflow training codes]
- File name: tf_train.py
- Last updated: 2021.5.25
"""

import argparse
import tensorflow as tf

import matplotlib.pyplot as plt
import tf_modules as tfmd


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model", required=True, help="cnn or dnn")
    args.add_argument("--n_epochs", type=int, default=5)
    args.add_argument("--batch_size", type=int, default=512)
    args.add_argument("--keras", action="store_true")
    args.add_argument("--save", action="store_true")
    args.add_argument("--history", action="store_true")
    return args.parse_args()


if __name__ == "__main__":

    ## Parameters
    seed = 11
    tf.random.set_seed(seed)
    args = get_args()

    ## Dataset
    assert args.model in ('cnn', 'dnn')
    flatten = True if args.model == 'dnn' else False
    train_loader, valid_loader, test_loader = tfmd.get_dataloaders(
            args.batch_size, seed=seed, flatten=flatten)

    # Model
    model = tfmd.SimpleDNN() if flatten else tfmd.SimpleCNN()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optim = tf.keras.optimizers.Adam()

    ## Training
    print("\n[Training with validation]")

    if args.keras:
        model.compile(optimizer=optim, loss=loss_fn)
        hist = model.fit(train_loader, validation_data=valid_loader,
                epochs=args.n_epochs, batch_size=args.batch_size)    

        if args.history:
            plt.plot(hist.history['loss'], 'ko-', lw=2, label="Train loss")
            plt.plot(hist.history['val_loss'], 'ro-', lw=2, label="Valid loss")
            plt.legend(fontsize=12)
            plt.grid()
            plt.show()

        if args.save:
            model.save("tf_model_%s" % args.model) # folder_name
    else:
        trainer = tfmd.ManualTrainer(model, optim, loss_fn)
        trainer.fit(train_loader, valid_loader, args.n_epochs)

        if args.history:
            trainer.plot_history()

        if args.save:
            trainer.model.save("tf_model_%s" % args.model) # folder_name
            ## new_model = tf.keras.models.load_model(folder_name)

    ## Test
    print("\n[Test]")

    train_loss = tfmd.average(loss_fn, model, train_loader)
    valid_loss = tfmd.average(loss_fn, model, valid_loader)
    test_loss  = tfmd.average(loss_fn, model, test_loader)

    train_acc = tfmd.average(tfmd.accuracy, model, train_loader)
    valid_acc = tfmd.average(tfmd.accuracy, model, valid_loader)
    test_acc  = tfmd.average(tfmd.accuracy, model, test_loader)

    print(">> train_loss = %.4f, train_acc = %.4f" % (train_loss, train_acc))
    print(">> valid_loss = %.4f, valid_acc = %.4f" % (valid_loss, valid_acc))
    print(">> test_loss  = %.4f, test_acc  = %.4f" % (test_loss, test_acc))