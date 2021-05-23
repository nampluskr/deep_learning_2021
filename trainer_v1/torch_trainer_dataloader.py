import argparse
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--n_epochs', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=512)

    return args.parse_args()


def preprocess(x, y):
    x = x.astype("float32")/255.  # float
    y = y.astype("int64")         # long
    return x.reshape(x.shape[0], -1), y


def get_dataloaders(batch_size, valid_ratio=0.2, seed=0):
    import mnist
    from sklearn.model_selection import train_test_split

    images, labels = mnist.train_images(), mnist.train_labels()
    x_train, x_valid, y_train, y_valid = train_test_split(
            images, labels, test_size=valid_ratio, random_state=seed)

    x_train, y_train = preprocess(x_train, y_train)
    x_valid, y_valid = preprocess(x_valid, y_valid)

    train_dataloader = DataLoader(dataset=ImageDataset(x_train, y_train),
                              batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=ImageDataset(x_valid, y_valid),
                              batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        super().__init__()
        self.images, self.labels = images, labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class Trainer():
    def __init__(self, model, loss_fn, optim, device):
        self.model, self.loss_fn, self.optim = model, loss_fn, optim
        self.device = device

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
                best_model = deepcopy(self.model.state_dict())

            template  = "Epoch [%3d/%3d] >>> train_loss = %.2e, valid_loss = %.2e, "
            template += "lowest_loss = %.2e @epoch = %d"
            print(template % (epoch+1, n_epochs, train_loss,
                            valid_loss, lowest_loss, best_epoch))

        self.model.load_state_dict(best_model)
        return lowest_loss, best_epoch

    def plot_history(self):
        fig, ax = plt.subplots()
        ax.plot(self.history['train_loss'], 'ko-', lw=2, label="Train loss")
        ax.plot(self.history['valid_loss'], 'ro-', lw=2, label="Valid loss")
        ax.legend(fontsize=12)
        ax.grid()
        plt.show()


if __name__ == "__main__":

    seed = 111
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Hyper parameters
    args = get_args()
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    ## Load dataset
    train_dataloader, valid_dataloader = get_dataloaders(batch_size, seed=seed)

    ## Model / Loss function / Optimizer / Trainner
    input_size = 784
    output_size = 10

    model = nn.Sequential(
        nn.Linear(input_size, 200),
        nn.LeakyReLU(),
        nn.Linear(200, 200),
        nn.LeakyReLU(),
        nn.Linear(200, output_size),
        nn.LogSoftmax(dim=-1),
    ).to(device)

    loss_fn = nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters())

    ## Training
    print("\n[Training with validation]")
    trainer = Trainer(model, loss_fn, optim, device=device)
    loss, epoch = trainer.fit(train_dataloader, valid_dataloader, n_epochs)
    model_name = "torch_model.valid_loss-%.4f.epoch-%d.pth" % (loss, epoch)
    torch.save(trainer.model, model_name)

    trainer.plot_history()
    ## new_model = torch.load(model_name)
