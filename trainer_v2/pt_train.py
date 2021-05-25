""" [Pytorch training codes]
- File name: pt_train.py
- Last updated: 2021.5.25
"""

import argparse
import torch

import pt_modules as ptmd


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model", required=True, help="cnn or dnn")
    args.add_argument("--n_epochs", type=int, default=5)
    args.add_argument("--batch_size", type=int, default=512)
    args.add_argument("--ignite", action="store_true")
    args.add_argument("--save", action="store_true")
    args.add_argument("--history", action="store_true")
    return args.parse_args()


if __name__ == "__main__":

    ## Parameters
    seed = 11
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    ## Dataset
    assert args.model in ('cnn', 'dnn')
    flatten = True if args.model == 'dnn' else False
    train_loader, valid_loader, test_loader = ptmd.get_dataloaders(
            args.batch_size, seed=seed, flatten=flatten)

    ## Model
    model = ptmd.SimpleDNN() if flatten else ptmd.SimpleCNN()
    model.to(device)
    loss_fn = torch.nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters())

    ## Training
    print("\n[Training with validation]")

    if args.ignite:
        trainer = ptmd.IgniteTrainer(model, optim, loss_fn, device)
    else:
        trainer = ptmd.ManualTrainer(model, optim, loss_fn, device)

    trainer.fit(train_loader, valid_loader, args.n_epochs)

    if args.history:
        trainer.plot_history()

    if args.save:
        torch.save(trainer.model, "pt_model_%s.pth" % args.model)
        ## new_model = torch.load(model_name)

    ## Test
    print("\n[Test]")

    train_loss = ptmd.average(loss_fn, model, train_loader)
    valid_loss = ptmd.average(loss_fn, model, valid_loader)
    test_loss  = ptmd.average(loss_fn, model, test_loader)

    train_acc = ptmd.average(ptmd.accuracy, model, train_loader)
    valid_acc = ptmd.average(ptmd.accuracy, model, valid_loader)
    test_acc  = ptmd.average(ptmd.accuracy, model, test_loader)

    print(">> train_loss = %.4f, train_acc = %.4f" % (train_loss, train_acc))
    print(">> valid_loss = %.4f, valid_acc = %.4f" % (valid_loss, valid_acc))
    print(">> test_loss  = %.4f, test_acc  = %.4f" % (test_loss, test_acc))