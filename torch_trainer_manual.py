import torch
import torch.nn as nn
import torch.optim as optim


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
    x = torch.from_numpy(x).float()/255.
    y = torch.from_numpy(y).long()

    return x.view(-1, 784), y


if __name__ == "__main__":

    ## Hyper parameters
    n_epochs = 10
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ## Load dataset
    (x_train, y_train), (x_valid, y_valid) = load_train_valid_data()
    x_train, y_train = preprocess(x_train, y_train)
    x_valid, y_valid = preprocess(x_valid, y_valid)

    print("Train images:", type(x_train), x_train.shape, x_train.dtype)
    print("Train labels:", type(y_train), y_train.shape, y_train.dtype)
    print("Valid images:", type(x_valid), x_valid.shape, x_valid.dtype)
    print("Valid labels:", type(y_valid), y_valid.shape, y_valid.dtype)


    ## Model / Loss function / Optimizer
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_valid, y_valid = x_valid.to(device), y_valid.to(device)

    input_size = x_train.size(-1)
    output_size = int(y_train.max()) + 1

    model = nn.Sequential(
        nn.Linear(input_size, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, output_size),
        nn.LogSoftmax(dim=-1),
    ).to(device)

    loss_fn = nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters())

    print("\n[Training with validation]")
    history = {'train_loss':[], 'valid_loss':[]}
    for epoch in range(n_epochs):

        ## Training
        model.train()
        indices = torch.randperm(x_train.size(0), device=x_train.device)
        x_train = torch.index_select(x_train, dim=0, index=indices)
        y_train = torch.index_select(y_train, dim=0, index=indices)
        x = x_train.split(batch_size, dim=0)
        y = y_train.split(batch_size, dim=0)

        batch_loss = []
        for xi, yi in zip(x, y):
            yi_hat = model(xi)
            loss = loss_fn(yi_hat, yi.squeeze())
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_loss.append(loss.item())

        train_loss = sum(batch_loss)/len(batch_loss)

        ## Validation
        model.eval()
        with torch.no_grad():
            x = x_valid.split(batch_size, dim=0)
            y = y_valid.split(batch_size, dim=0)

            batch_loss = []
            for xi, yi in zip(x, y):
                yi_hat = model(xi)
                loss = loss_fn(yi_hat, yi.squeeze())
                batch_loss.append(loss.item())

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