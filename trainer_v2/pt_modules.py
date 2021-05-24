""" [Pytorch user-defined modules and functions]
- File name: pt_modules.py
- Last updated: 2021.5.24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import (Events,
        create_supervised_trainer, create_supervised_evaluator)
from ignite.metrics import Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar

from copy import deepcopy
import matplotlib.pyplot as plt


def acc_fn(y_hat, y):
    return torch.eq(y_hat.argmax(-1), y).float().mean()


def average(func, model, data, batch_size=0):
    device = next(model.parameters()).device
    with torch.no_grad():

        batch_outputs = []
        if isinstance(data, torch.utils.data.DataLoader):
            for xi, yi in data:
                xi, yi = xi.to(device), yi.to(device)
                yi_hat = model(xi)
                output = func(yi_hat, yi.squeeze())
                batch_outputs.append(output.item())

        elif batch_size > 0:
            x = data[0].split(batch_size, dim=0)
            y = data[1].split(batch_size, dim=0)

            for xi, yi in zip(x, y):
                xi, yi = xi.to(device), yi.to(device)
                yi_hat = model(xi)
                output = func(yi_hat, yi.squeeze())
                batch_outputs.append(output.item())
        else:
            x, y = data[0].to(device), data[1].to(device)
            y_hat = model(x)
            output = func(y_hat, y.squeeze())
            batch_outputs.append(output.item())

        return sum(batch_outputs)/len(batch_outputs)


def get_dataloaders(batch_size, seed, flatten=False):
    import mnist
    from sklearn.model_selection import train_test_split

    def preprocess(x, y, flatten):
        x = x.astype("float32")/255.
        y = y.astype("int64")
        x = x.reshape(-1, 784) if flatten else x.reshape(-1, 1, 28, 28)
        return x, y

    images, labels = mnist.train_images(), mnist.train_labels()
    x_train, x_valid, y_train, y_valid = train_test_split(images, labels,
            test_size=0.2, random_state=seed)
    x_test, y_test = mnist.test_images(), mnist.test_labels()

    x_train, y_train = preprocess(x_train, y_train, flatten)
    x_valid, y_valid = preprocess(x_valid, y_valid, flatten)
    x_test,  y_test  = preprocess(x_test, y_test, flatten)

    train_loader = torch.utils.data.DataLoader(ImageDataset(x_train, y_train),
            batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(ImageDataset(x_valid, y_valid),
            batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(ImageDataset(x_test, y_test),
            batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images, self.labels = images, labels

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.images.shape[0]


class Trainer():
    def __init__(self, model, optim, loss_fn, device):
        self.model, self.loss_fn, self.optim = model, loss_fn, optim
        self.device = device
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
        self.model.train()
        batch_loss = []
        for xi, yi in dataloader:
            xi, yi = xi.to(self.device), yi.to(self.device)
            yi_hat = self.model(xi)
            loss = self.loss_fn(yi_hat, yi.squeeze())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            batch_loss.append(loss.item())

        return sum(batch_loss)/len(batch_loss)

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            batch_loss = []
            for xi, yi in dataloader:
                xi, yi = xi.to(self.device), yi.to(self.device)
                yi_hat = self.model(xi)
                loss = self.loss_fn(yi_hat, yi.squeeze())
                batch_loss.append(loss.item())

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
                self.best_model = deepcopy(self.model.state_dict())

            template  = "Epoch [%3d/%3d] >> train_loss = %.4f, valid_loss = %.4f, "
            template += "lowest_loss = %.4f @epoch = %d"
            print(template % (epoch+1, n_epochs, train_loss, valid_loss,
                            self.best_loss, self.best_epoch))

        self.model.load_state_dict(self.best_model)


class IgniteTrainer(Trainer):

    def fit(self, train_loader, valid_loader, n_epochs):
        trainer = create_supervised_trainer(self.model,
            self.optim, self.loss_fn, device=self.device)

        evaluator = create_supervised_evaluator(self.model,
            metrics={'loss': Loss(self.loss_fn)}, device=self.device)

        RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
        pbar = ProgressBar(persist=False)
        pbar.attach(trainer, metric_names="all")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results():
            train_loss = trainer.state.metrics['loss']
            evaluator.run(valid_loader)
            valid_loss = evaluator.state.metrics['loss']

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_epoch = trainer.state.epoch
                self.best_model = deepcopy(self.model.state_dict())

            template  = "Epoch [%3d/%3d] >> train_loss = %.4f, valid_loss = %.4f, "
            template += "lowest_loss = %.4f @epoch = %d"
            pbar.log_message(template % (trainer.state.epoch, 
                    trainer.state.max_epochs, trainer.state.output, 
                    valid_loss, self.best_loss, self.best_epoch))

        trainer.run(train_loader, max_epochs=n_epochs)
        self.model.load_state_dict(self.best_model)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class SimpleDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":

    pass