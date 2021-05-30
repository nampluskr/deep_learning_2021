import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ignite.engine import (Events,
        create_supervised_trainer, create_supervised_evaluator)
from ignite.metrics import Loss, RunningAverage
from ignite.contrib.handlers import ProgressBar

from copy import deepcopy
import matplotlib.pyplot as plt


def pt_dataloader(data, batch_size, training=True):
    def preprocess(data):
        x, y = data
        x = torch.tensor(x).float().permute(0, 3, 1, 2)/255.
        y = torch.tensor(y).long()
        return (x, y)

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if training:
        return torch.utils.data.DataLoader(
            dataset=ImageDataset(preprocess(data), transform=transform_train),
            batch_size=batch_size,
            shuffle=True)
    else:
        return torch.utils.data.DataLoader(
            dataset=ImageDataset(preprocess(data), transform=transform_test),
            batch_size=batch_size,
            shuffle=False)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.images, self.labels = data
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        images, labels= self.images[idx], self.labels[idx]
        if self.transform:
            images = self.transform(images)
        return images, labels


class ConvClassifier(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

        self.blocks = nn.Sequential( # (bs,   3, 32, 32)
            ConvBlock(3, 32),        # (bs,  32, 16, 16)
            ConvBlock(32, 64),       # (bs,  64,  8,  8)
            ConvBlock(64, 128),      # (bs, 128,  4,  4)
            ConvBlock(128, 256),     # (bs, 256,  2,  2)
            ConvBlock(256, 512)      # (bs, 512,  1,  1)
        )

        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
        )

    def forward(self, x):
        assert x.ndim > 2
        if x.ndim == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))

        z = self.blocks(x)
        y = self.layers(z.squeeze())
        return F.log_softmax(y, dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):  # |x| = (batch_size, in_channels, h, w)
        y = self.layers(x) # |y| = (batch_size, out_channels, h/2, w/2)
        return y


class Trainer():
    def __init__(self, model, optim, loss_fn):
        self.model, self.loss_fn, self.optim = model, loss_fn, optim
        self.device = next(model.parameters()).device
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
        for i, (xi, yi) in enumerate(dataloader):
            xi, yi = xi.to(self.device), yi.to(self.device)
            yi_hat = self.model(xi)
            loss = self.loss_fn(yi_hat, yi.squeeze())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            batch_loss.append(loss.item())

            if i+1 in [round(len(dataloader)*(i+1)/10) for i in range(10)]:
                print("Training %3.f%% [%3d/%3d] train_loss: %.4f" % (
                        round((i+1)/len(dataloader)*100),
                        i+1, len(dataloader), loss))

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


def accuracy(y_hat, y):
    return torch.eq(y_hat.argmax(-1), y).float().mean()


def average(func, model, data, batch_size=0):
    device = next(model.parameters()).device
    model.eval()
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