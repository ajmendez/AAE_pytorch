import torch
import numpy as np
import argparse
import time
import pickle
import itertools
# from .viz import *
from aae_pytorch_basic import save_model, report_loss, load_data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs to train (default: 5000)')

args = parser.parse_args()
cuda = torch.cuda.is_available()

seed = 10


kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
n_classes = 10
X_dim = 784
z_dim = 10
y_dim = 10
train_batch_size = args.batch_size
valid_batch_size = args.batch_size
N = 1000
epochs = args.epochs

params = {'n_classes': n_classes,
          'z_dim': z_dim,
          'X_dim': X_dim,
          'y_dim': y_dim,
          'train_batch_size': train_batch_size,
          'valid_batch_size': valid_batch_size,
          'N': N,
          'epochs': epochs,
          'cuda': cuda}


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
        # Categorical code (y)
        self.lin3cat = nn.Linear(N, n_classes)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        xcat = F.softmax(self.lin3cat(x))

        return xcat, xgauss


# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim + n_classes, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)


# Discriminator networks
class D_net_cat(nn.Module):
    def __init__(self):
        super(D_net_cat, self).__init__()
        self.lin1 = nn.Linear(n_classes, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
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



def create_latent(Q, loader):
    '''
    Creates the latent representation for the samples in loader
    return:
        z_values: numpy array with the latent representations
        labels: the labels corresponding to the latent representations
    '''
    Q.eval()
    labels = []

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


def classification_accuracy(Q, data_loader):
    Q.eval()
    labels = []
    test_loss = 0
    correct = 0

    for batch_idx, (X, target) in enumerate(data_loader):
        X = X * 0.3081 + 0.1307
        X.resize_(data_loader.batch_size, X_dim)
        X, target = Variable(X), Variable(target)
        if cuda:
            X, target = X.cuda(), target.cuda()

        labels.extend(target.data.tolist())
        # Reconstruction phase
        output = Q(X)[0]

        test_loss += F.nll_loss(output, target).data[0]

        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader)
    return 100. * correct / len(data_loader.dataset)


def eval_model():
    import os
    all_model_files = os.listdir(models_path)
    model_names = {'Q_net', 'P_net', 'D_gauss'}
    model_files = {}

    for name in model_names:
        k_model_files = [mf for mf in all_model_files if mf.startswith(name)]
        k_model_files = [os.path.join(models_path, mf) for mf in k_model_files]
        model_file = max(k_model_files, key=os.path.getctime)
        model_files[name] = model_file

    # Construct networks Q (encoder), P (decoder), D_gauss (discriminator)
    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net()
        P = P_net()
        D_gauss = D_net_gauss()

    # Load models weights
    Q.load_state_dict(torch.load(model_files['Q_net']))
    # P.load_state_dict(torch.load(model_files['P_net']))
    # D_gauss.load_state_dict(torch.load(model_files['D_gauss']))

    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),
                                                         (0.3081,))])

    testset = datasets.MNIST('../data', train=False, download=True,
                             transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                              shuffle=True, num_workers=2)

    test_acc = classification_accuracy(Q, testloader)
    print('Test accuracy: {:5.2f}%'.format(test_acc))

    return test_acc

####################
# Train procedure
####################
def train(P, Q, D_cat, D_gauss, P_decoder, Q_encoder, Q_semi_supervised,
          Q_generator, D_cat_solver, D_gauss_solver, train_labeled_loader,
          train_unlabeled_loader):
    '''
    Train procedure for one epoch.
    '''
    TINY = 1e-15
    # Set the networks in train mode (apply dropout when needed)
    Q.train()
    P.train()
    D_cat.train()
    D_gauss.train()

    if train_unlabeled_loader is None:
        train_unlabeled_loader = train_labeled_loader

    # Loop through the labeled and unlabeled dataset getting one batch of samples from each
    # The batch size has to be a divisor of the size of the dataset or it will return
    # invalid samples
    for (X_l, target_l), (X_u, target_u) in zip(train_labeled_loader, train_unlabeled_loader):

        for X, target in [(X_u, target_u), (X_l, target_l)]:
            if target[0] == -1:
                labeled = False
            else:
                labeled = True

            # Load batch and normalize samples to be between 0 and 1
            X = X * 0.3081 + 0.1307
            X.resize_(train_batch_size, X_dim)

            X, target = Variable(X), Variable(target)
            if cuda:
                X, target = X.cuda(), target.cuda()

            # Init gradients
            P.zero_grad()
            Q.zero_grad()
            D_cat.zero_grad()
            D_gauss.zero_grad()

            #######################
            # Reconstruction phase
            #######################
            if not labeled:
                z_sample = torch.cat(Q(X), 1)
                X_sample = P(z_sample)

                recon_loss = F.binary_cross_entropy(X_sample + TINY, X.resize(train_batch_size, X_dim) + TINY)
                recon_loss = recon_loss
                recon_loss.backward()
                P_decoder.step()
                Q_encoder.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()
                recon_loss = recon_loss
                #######################
                # Regularization phase
                #######################
                # Discriminator
                Q.eval()
                z_real_cat = sample_categorical(train_batch_size, n_classes=n_classes)
                z_real_gauss = Variable(torch.randn(train_batch_size, z_dim))
                if cuda:
                    z_real_cat = z_real_cat.cuda()
                    z_real_gauss = z_real_gauss.cuda()

                z_fake_cat, z_fake_gauss = Q(X)

                D_real_cat = D_cat(z_real_cat)
                D_real_gauss = D_gauss(z_real_gauss)
                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                D_loss_cat = -torch.mean(torch.log(D_real_cat + TINY) + torch.log(1 - D_fake_cat + TINY))
                D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))

                D_loss = D_loss_cat + D_loss_gauss
                D_loss = D_loss

                D_loss.backward()
                D_cat_solver.step()
                D_gauss_solver.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()

                # Generator
                Q.train()
                z_fake_cat, z_fake_gauss = Q(X)

                D_fake_cat = D_cat(z_fake_cat)
                D_fake_gauss = D_gauss(z_fake_gauss)

                G_loss = - torch.mean(torch.log(D_fake_cat + TINY)) - torch.mean(torch.log(D_fake_gauss + TINY))
                G_loss = G_loss
                G_loss.backward()
                Q_generator.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()

            #######################
            # Semi-supervised phase
            #######################
            if labeled:
                pred, _ = Q(X)
                class_loss = F.cross_entropy(pred, target)
                class_loss.backward()
                Q_semi_supervised.step()

                P.zero_grad()
                Q.zero_grad()
                D_cat.zero_grad()
                D_gauss.zero_grad()

    return D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss


####################
# Utility functions
####################
def sample_categorical(batch_size, n_classes=10):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.autograd.Variable with the sample
    '''
    cat = np.random.randint(0, 10, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return Variable(cat)


def generate_model():
    torch.manual_seed(10)

    if cuda:
        Q = Q_net().cuda()
        P = P_net().cuda()
        D_cat = D_net_cat().cuda()
        D_gauss = D_net_gauss().cuda()
    else:
        Q = Q_net()
        P = P_net()
        D_gauss = D_net_gauss()
        D_cat = D_net_cat()

    # Set learning rates
    gen_lr = 0.0006
    semi_lr = 0.001
    reg_lr = 0.0008

    # Set optimizers
    P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
    Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)

    Q_semi_supervised = optim.Adam(Q.parameters(), lr=semi_lr)

    Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
    D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)
    D_cat_solver = optim.Adam(D_cat.parameters(), lr=reg_lr)

    header_fields = ['Epoch', 'D_loss_gauss', 'G_loss', 'Recon_loss',
                     'Class_loss', 'Train_acc', 'Valid_acc', 'dt']

    header_fmt = '{:>10} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>10}'
    report_fmt = '{epoch:>10} {d_loss_gauss:>15.6e} {g_loss:>15.6e} ' \
                 '{recon_loss:>15.6e} {class_loss:>15.6e} {train_acc:>15.6e} '\
                 '{val_acc:>15.6e} {dt:>10.2f}'

    header = header_fmt.format(*header_fields)
    print('\n{}\n{}'.format(header, '-' * len(header)))

    t_start = time.time()
    t_total = t_start
    for epoch in range(epochs):
        D_loss_cat, D_loss_gauss, G_loss, recon_loss, class_loss = \
            train(P, Q, D_cat, D_gauss, P_decoder, Q_encoder,
                  Q_semi_supervised, Q_generator, D_cat_solver,
                  D_gauss_solver, train_labeled_loader, train_unlabeled_loader)

        if epoch % 10 == 0:
            dt = time.time() - t_start
            train_acc = classification_accuracy(Q, train_labeled_loader)
            val_acc = classification_accuracy(Q, valid_loader)

            report_args = {'epoch': epoch,
                           'd_loss_gauss': D_loss_gauss.data[0],
                           'g_loss': G_loss.data[0],
                           'recon_loss': recon_loss.data[0],
                           'class_loss': class_loss.data[0],
                           'dt': dt,
                           'train_acc': train_acc,
                           'val_acc': val_acc}

            report_loss(report_fmt, **report_args)

            save_model(Q, 'Q_net_{:05d}.p'.format(epoch), models_path)
            save_model(P, 'P_net_{:05d}.p'.format(epoch), models_path)
            save_model(D_gauss, 'D_gauss_{:05d}.p'.format(epoch), models_path)
            save_model(D_cat, 'D_cat_{:05d}.p'.format(epoch), models_path)
            t_start = time.time()

    t_total = time.time() - t_total
    print('Training time: {:10.2f} seconds.\n'.format(t_total))

    return Q, P, D_gauss, D_cat


if __name__ == '__main__':
    models_path = '../models/semisup'
    train_labeled_loader, train_unlabeled_loader, valid_loader = load_data()
    # Q, P, D_gauss, D_cat = generate_model()
    eval_model()