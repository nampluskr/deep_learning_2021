""" [Pytorch manual training codes]
- File name: pt_train_manual.py
- Last updated: 2021.5.23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import matplotlib.pyplot as plt


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

    ## Parameters
    seed = 111
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 20
    batch_size = 512

    model_type = 'dnn'
    assert model_type in ('dnn', 'cnn')
    flatten = True if model_type == 'dnn' else False

    ## Dataset
    import mnist
    from sklearn.model_selection import train_test_split

    def preprocess(x, y, flatten):
        x = x.astype("float32")/255.
        y = y.astype("int64")
        x = x.reshape(-1, 784) if flatten else x.reshape(-1, 1, 28, 28)
        return torch.tensor(x), torch.tensor(y)

    images, labels = mnist.train_images(), mnist.train_labels()
    x_train, x_valid, y_train, y_valid = train_test_split(images, labels,
        test_size=0.2, random_state=seed)

    x_train, y_train = preprocess(x_train, y_train, flatten=flatten)
    x_valid, y_valid = preprocess(x_valid, y_valid, flatten=flatten)

    ## Model
    model = SimpleDNN() if flatten else SimpleCNN()
    model.to(device)
    loss_fn = torch.nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters())

    ## Training
    best_loss, best_epoch, best_model = 1e12, 0, None
    history = {'train_loss':[], 'valid_loss':[]}

    for epoch in range(n_epochs):
        ## Training
        model.train()
        indices = torch.randperm(x_train.size(0), device=x_train.device)
        x_train = torch.index_select(x_train, dim=0, index=indices)
        y_train = torch.index_select(y_train, dim=0, index=indices)
        x = x_train.split(batch_size, dim=0)
        y = y_train.split(batch_size, dim=0)

        train_batch_loss = []
        for xi, yi in zip(x, y):
            xi, yi = xi.to(device), yi.to(device)
            yi_hat = model(xi)
            loss = loss_fn(yi_hat, yi.squeeze())
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_batch_loss.append(loss.item())

        ## Validation
        model.eval()
        with torch.no_grad():
            x = x_valid.split(batch_size, dim=0)
            y = y_valid.split(batch_size, dim=0)

            valid_batch_loss = []
            for xi, yi in zip(x, y):
                xi, yi = xi.to(device), yi.to(device)
                yi_hat = model(xi)
                loss = loss_fn(yi_hat, yi.squeeze())
                valid_batch_loss.append(loss.item())

        ## Print results
        train_loss = sum(train_batch_loss)/len(train_batch_loss)
        valid_loss = sum(valid_batch_loss)/len(valid_batch_loss)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_model = deepcopy(model.state_dict())

        template  = "Epoch [%3d/%3d] >> train_loss = %.4f, valid_loss = %.4f, "
        template += "lowest_loss = %.4f @epoch = %d"
        print(template % (epoch+1, n_epochs, train_loss, valid_loss,
                        best_loss, best_epoch))

    model.load_state_dict(best_model)

    plt.plot(history['train_loss'], 'ko-', lw=2, label="Train loss")
    plt.plot(history['valid_loss'], 'ro-', lw=2, label="Valid loss")
    plt.legend(fontsize=12)
    plt.grid()
    plt.show()

    torch.save(model, "pt_model_manual_%s.pth" % model_type)
    ## new_model = torch.load(model_name)
