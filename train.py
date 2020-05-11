"""
ConGCN/train.py

Simulate Synthetic data.
"""
import random
import pandas as pd
from torch.optim import Adam, Adagrad
from model.mlp import MLP
from model.gcn import GCN
from model.gat import GAT
from utils.data_utils import *
from utils.train_utils import *
from utils.visualize_utils import *
from utils.NLLLoss import NLLLoss


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _train_epoch(data, model, loss_fn, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `loss_fn`
    """
    model.train()
    optimizer.zero_grad()
    output = model(data)
    if hasattr(loss_fn, 'requires_conv'):
        loss = loss_fn(
            output[data.train_mask],
            data.y[data.train_mask],
            data.sigma_inv_train)
    else:
        loss = loss_fn(
            output[data.train_mask],
            data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def _get_loss(data, model, criterion):
    """
    Get the loss of `model` on set.
    """
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss_train = criterion(
            output[data.train_mask],
            data.y[data.train_mask]).item()
        loss_valid = criterion(
            output[data.val_mask],
            data.y[data.val_mask]).item()
        loss_test = criterion(
            output[data.test_mask],
            data.y[data.test_mask]).item()
    return loss_train, loss_valid, loss_test


def _evaluate_epoch(axes, data, model, criterion, epoch, stats, log_interval, verbose=True):
    """
    Evaluates the `model` on the train and validation set.
    """
    train_loss, val_loss, test_loss = _get_loss(data, model, criterion)

    stats.append([val_loss, train_loss, test_loss])
    if verbose:
        log_training(epoch, stats)
    if axes:
        update_training_plot(axes, epoch, stats, log_interval)


def train_model(data,
                model_name="mlp",
                log_interval=10,
                loss="mse",
                optim="adam",
                store=False,
                visual=False,
                verbose=False,
                log=False):

    if verbose and log:
        print("\nTraining model", model_name)

    if store:
        # Create directory
        checkpoint_path = os.path.join("checkpoints/", model_name)
        if not os.path.isdir(checkpoint_path):
            try:
                os.makedirs(checkpoint_path)
            except OSError:
                sys.exit("Creation of checkpoint directory failed.")

    # Model
    model = None
    if model_name == "mlp":
        model = MLP(
            num_features=data.X.shape[1],
            hidden_size=config("{}.hidden_layer".format(model_name)),
        )
    elif model_name == "gcn":
        model = GCN(
            num_features=data.X.shape[1],
            hidden_size=config("{}.hidden_layer".format(model_name)),
        )
    elif model_name == "gat":
        model = GAT(
            num_features=data.X.shape[1],
            hidden_size=config("{}.hidden_layer".format(model_name)),
        )

    # Criterion and Loss Function
    criterion = torch.nn.MSELoss()
    loss_fn = None
    if loss == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss == "nll":
        loss_fn = NLLLoss()

    # Optimizer
    optimizer = None
    if optim == "adam":
        optimizer = Adam(
            model.parameters(), lr=config('{}.learning_rate'.format(model_name))
        )
    elif optim == "adagrad":
        optimizer = Adagrad(
            model.parameters(), lr=config('{}.learning_rate'.format(model_name))
        )

    # Setup training
    fig, axes = None, None
    if visual:
        fig, axes = make_training_plot(model_name)

    start_epoch = 0
    stats = []
    if store:
        # Attempts to restore the latest checkpoint if exists
        print('Loading {}...'.format(model_name))
        model, start_epoch, stats = restore_checkpoint(
            model, config('{}.checkpoint'.format(model_name))
        )

        # Evaluate the randomly initialized model
        _evaluate_epoch(
            axes, data, model, criterion, start_epoch, stats, log_interval, log
        )

    # Loop over the entire dataset multiple times
    patience = config("patience")
    best_loss = float('inf')
    idx = -1

    for epoch in range(start_epoch, config('{}.num_epochs'.format(model_name))):
        # Early stop
        if patience < 0:
            break

        # Train model
        _train_epoch(
            data, model, loss_fn, optimizer
        )

        # Evaluate model
        if (epoch + 1) % log_interval == 0:
            _evaluate_epoch(
                axes, data, model, criterion, epoch+1, stats, log_interval, log
            )

            if store:
                # Save model parameters
                save_checkpoint(
                    model, epoch+1, config('{}.checkpoint'.format(model_name)), stats
                )

            valid_loss = stats[-1][0]
            if valid_loss < best_loss:
                patience = config("patience")
                best_loss = valid_loss
                idx = epoch

            patience -= 1

    epoch = idx
    idx = min(int((idx+1)/log_interval), len(stats)-1)

    if verbose:
        print("The loss on test dataset is:", stats[idx][2],
              "obtained in epoch", epoch)

    # Save figure and keep plot open
    if visual:
        save_training_plot(fig, model_name)
        hold_training_plot()

    return stats[idx][2]


def standard_simulate():
    checkpoint_path = "checkpoints/"
    output_path = "output/"
    if not os.path.isdir(checkpoint_path):
        try:
            os.makedirs(checkpoint_path)
        except OSError:
            sys.exit("Creation of checkpoint directory failed.")
    if not os.path.isdir(output_path):
        try:
            os.makedirs(output_path)
        except OSError:
            sys.exit("Creation of output directory failed.")

    d0 = config("d0")
    d1 = config("d1")
    n = config("node")
    m = config("feature")
    gamma = config("gamma")
    _seed = range(config("seed"))
    _tau = config("tau")
    _setting = config("setting")

    for setting in _setting:
        model = [[], [], []]
        mean = setting[0]
        covariance = setting[1]
        if covariance == "i":
            _tau = [1]

        for tau in _tau:
            loss = [[], [], []]

            for seed in _seed:
                _set_seed(seed)

                print("\nData info: {}_{}_n{}_d0{}_d1{}_m{}_g{}_t{}_s{}".format(
                    mean, covariance, n, d0, d1, m, gamma, tau, seed))

                data = get_dataset(seed=seed,
                                   m=m,
                                   gamma=gamma,
                                   tau=tau,
                                   mean_mode=mean,
                                   cov_mode=covariance)

                loss[0].append(train_model(data, "mlp", visual=False, loss="nll", verbose=True))
                loss[1].append(train_model(data, "gcn", visual=False, loss="nll", verbose=True))
                loss[2].append(train_model(data, "gat", visual=False, loss="nll", verbose=True))

            mean_loss = np.mean(loss, axis=1)
            std_loss = np.std(loss, axis=1)

            for i in range(3):
                model[i].append([mean_loss[i], std_loss[i]])

        df = {"tau": _tau, "mlp": model[0], "gcn": model[1], "gat": model[2]}
        pd_writer = pd.DataFrame(df)
        pd_writer.to_csv('output/{}_{}.csv'.format(mean, covariance),
                         index=False, header=False)


def noisy_simulate():
    checkpoint_path = "checkpoints/"
    output_path = "output/"
    if not os.path.isdir(checkpoint_path):
        try:
            os.makedirs(checkpoint_path)
        except OSError:
            sys.exit("Creation of checkpoint directory failed.")
    if not os.path.isdir(output_path):
        try:
            os.makedirs(output_path)
        except OSError:
            sys.exit("Creation of output directory failed.")

    tau = 0.1
    d0 = config("d0")
    d1 = config("d1")
    n = config("node")
    m = config("feature")
    gamma = config("gamma")
    _seed = range(config("seed"))
    _drop_rate = config("drop_rate")
    _setting = config("setting")

    for setting in _setting:
        model = [[], [], []]
        mean = setting[0]
        covariance = setting[1]

        for dr in _drop_rate:
            loss = [[], [], []]

            for seed in _seed:
                _set_seed(seed)

                print("\nData info: {}_{}_n{}_d0{}_d1{}_m{}_g{}_t{}_dr{}_s{}".format(
                    mean, covariance, n, d0, d1, m, gamma, tau, dr, seed))

                data = get_dataset(seed=seed,
                                   m=m,
                                   gamma=gamma,
                                   tau=tau,
                                   mean_mode=mean,
                                   cov_mode=covariance,
                                   noise="flip",
                                   drop_rate=dr)

                loss[0].append(train_model(data, "mlp", verbose=True))
                loss[1].append(train_model(data, "gcn", verbose=True))
                loss[2].append(train_model(data, "gat", verbose=True))

            mean_loss = np.mean(loss, axis=1)
            std_loss = np.std(loss, axis=1)

            for i in range(len(loss)):
                model[i].append([mean_loss[i], std_loss[i]])

        df = {"drop_rate": _drop_rate, "mlp": model[0], "gcn": model[1], "gat": model[2]}
        pd_writer = pd.DataFrame(df)
        pd_writer.to_csv('output/{}_{}.csv'.format(mean, covariance),
                         index=False, header=False)


if __name__ == '__main__':
    standard_simulate()
