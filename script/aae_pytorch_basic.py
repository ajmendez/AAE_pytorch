import argparse
import os
import torch
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

args = parser.parse_args()
cuda = torch.cuda.is_available()

seed = 10


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 10
weight_init = {'distro': 'gaussian', 'params': {'mu':0, 'std': 0.01}}
z_dim = 8
z_prior_std = 5.
X_dim = 784
y_dim = 10
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
N = 1000
epochs = args.epochs


##################################
# Load data and create Data loaders
##################################
def load_data():
    from .create_datasets import split_dataset, load_mnist

    print('Loading data!')
    trainset_labeled, trainset_unlabeled, validset = \
        split_dataset(load_mnist(), n_train_labels_pc=10, n_validation_pc=1000)

    if trainset_unlabeled.train_labels is None:
        n = trainset_unlabeled.train_data.size()[0]
        trainset_unlabeled.train_labels = torch.from_numpy(np.array([-1] * n))

    # trainset_labeled = pickle.load(open(data_path + "train_labeled.p", "rb"))
    # trainset_unlabeled = pickle.load(open(data_path + "train_unlabeled.p", "rb"))
    # # Set -1 as labels for unlabeled data
    # trainset_unlabeled.train_labels = torch.from_numpy(np.array([-1] * 47000))
    # validset = pickle.load(open(data_path + "validation.p", "rb"))

    train_labeled_loader = torch.utils.data.DataLoader(trainset_labeled,
                                                       batch_size=train_batch_size,
                                                       shuffle=True, **kwargs)

    train_unlabeled_loader = torch.utils.data.DataLoader(trainset_unlabeled,
                                                         batch_size=train_batch_size,
                                                         shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(validset, batch_size=valid_batch_size, shuffle=True)

    return train_labeled_loader, train_unlabeled_loader, valid_loader


##################################
# Define Networks
##################################
# Encoder - Generator
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        # Gaussian code (z)
        self.lin3gauss = nn.Linear(N, z_dim)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)

        return xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self, input_dim=z_dim):
        """
        input_dim : int
            could be either z_dim (basic), z_dim + n_classes (supervised)
        """
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)

        return F.sigmoid(self.lin3(x))


####################
# Utility functions
####################


def create_latent(Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    Q.eval()
    labels = []
    z_values = np.array([]).reshape(0, z_dim)

    for batch_idx, (X, target) in enumerate(loader):

        X = X * 0.3081 + 0.1307
        # X.resize_(loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        labels.extend(target.data.tolist())
        if cuda:
            X, target = X.cuda(), target.cuda()
        # Reconstruction phase
        z_sample = Q(X)
        if batch_idx > 0:
            z_values = np.concatenate((z_values, np.array(z_sample.data.tolist())))
        else:
            z_values = np.array(z_sample.data.tolist())
    labels = np.array(labels)

    return z_values, labels


# Used for supervised classification
def get_categorical(labels, n_classes=10):
    cat = np.array(labels.data.tolist())
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)


####################
# Train procedure
####################
def train(P, Q, D_gauss, P_decoder, Q_encoder, Q_generator, D_gauss_solver, data_loader):
    '''
    Train procedure for one epoch.
    '''
    TINY = 1e-15
    # Set the networks in train mode (apply dropout when needed)
    Q.train()
    P.train()
    D_gauss.train()

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for X, target in data_loader:

        # Load batch and normalize samples to be between 0 and 1
        X = X * 0.3081 + 0.1307
        X.resize_(train_batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        # Init gradients
        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        #######################
        # Reconstruction phase
        #######################

        if mode == 'basic':
            z_sample = Q(X)
        elif mode == 'supervised':
            z_gauss = Q(X)
            z_cat = get_categorical(target, n_classes=10)
            if cuda:
                z_cat = z_cat.cuda()

            z_sample = torch.cat((z_cat, z_gauss), 1)

        X_sample = P(z_sample)
        recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)

        recon_loss.backward()
        P_decoder.step()
        Q_encoder.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        #######################
        # Regularization phase
        #######################
        # Discriminator
        Q.eval()
        z_real_gauss = Variable(torch.randn(train_batch_size, z_dim) *
                                z_prior_std)
        if cuda:
            z_real_gauss = z_real_gauss.cuda()

        z_fake_gauss = Q(X)

        D_real_gauss = D_gauss(z_real_gauss)
        D_fake_gauss = D_gauss(z_fake_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

        D_loss.backward()
        D_gauss_solver.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

        # Generator
        Q.train()
        z_fake_gauss = Q(X)

        D_fake_gauss = D_gauss(z_fake_gauss)
        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))

        G_loss.backward()
        Q_generator.step()

        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()

    return D_loss, G_loss, recon_loss


def save_model(model, filename, models_path='../models/unsupervised'):
    # print('Best model so far, saving it...')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    path = os.path.join(models_path, filename)
    torch.save(model.state_dict(), path)


def report_loss(report_fmt, **kwargs):
    '''
    Print loss
    '''
    print(report_fmt.format(**kwargs))


def generate_model():
    torch.manual_seed(10)

    # Construct networks Q (encoder), P (decoder), D_gauss (discriminator)
    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net()
        P = P_net()
        D_gauss = D_net_gauss()

    # Set learning rates
    gen_lr = 0.0001
    reg_lr = 0.00005

    # Set optimizers
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

    header_fields = ['Epoch', 'D_loss_gauss', 'G_loss', 'Recon_loss', 'dt']

    header_fmt = '{:>10} {:>15} {:>15} {:>15} {:>10}'
    report_fmt = '{epoch:>10} {d_loss_gauss:>15.6e} {g_loss:>15.6e} ' \
                 '{recon_loss:>15.6e} {dt:>10.2f}'

    header = header_fmt.format(*header_fields)
    print('\n{}\n{}'.format(header, '-' * len(header)))

    t_start = time.time()
    for epoch in range(epochs):
        D_loss_gauss, G_loss, recon_loss = train(P, Q, D_gauss,
                                                 P_decoder,
                                                 Q_encoder,
                                                 Q_generator,
                                                 D_gauss_solver,
                                                 train_unlabeled_loader)
        if epoch % 10 == 0:
            dt = time.time() - t_start

            report_args = {'epoch': epoch,
                           'd_loss_gauss': D_loss_gauss.data[0],
                           'g_loss': G_loss.data[0],
                           'recon_loss': recon_loss.data[0],
                           'dt': dt}

            report_loss(report_fmt, **report_args)

            t_save = time.time()
            save_model(Q, 'Q_net_{:05d}.p'.format(epoch))
            save_model(P, 'P_net_{:05d}.p'.format(epoch))
            save_model(D_gauss, 'D_gauss_{:05d}.p'.format(epoch))
            t_save = time.time() - t_save
            print('Took {:10.2f} to save the models.\n'.format(t_save))
            t_start = time.time()

    return Q, P, D_gauss


if __name__ == '__main__':
    mode = 'basic'
    models_path = '../models/unsupervised'
    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data()
    Q, P, D_gauss = generate_model()
