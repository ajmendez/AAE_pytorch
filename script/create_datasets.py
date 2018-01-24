from __future__ import print_function
import pickle
import numpy as np
import torch
from torchvision import datasets, transforms

from sub import subMNIST


mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

trainset_original = datasets.MNIST('../data', train=True, download=True,
                                   transform=mnist_transform)


def load_mnist():
    return trainset_original


def split_dataset(trainset_original, n_train_labels_pc=10,
                  n_train_unlabeled_pc=None, n_validation_pc=1000):
    """

    Parameters
    ----------
    trainset_original : torch.utils.data.Dataset
        A dataset object as defined by torch

    n_train_labels_pc : int
        Number of labeled samples per class to use for training.

    n_train_unlabeled_pc : int
        Number of unlabeled samples per class to use for training.

    n_validation_pc : int
        Number of labeled samples per class to use for validation.

    Returns
    -------
    trainset_new, validset, trainset_new_unl: torch.utils.data.Dataset objects

    """

    train_label_index = []
    train_unlabel_index = []
    valid_label_index = []

    classes = np.unique(trainset_original.train_labels.numpy())
    n_classes = len(classes)

    for i in range(n_classes):
        train_label_list = trainset_original.train_labels.numpy()
        label_index = np.where(train_label_list == i)[0]
        n_class_samples = len(label_index)

        n_tv = n_train_labels_pc + n_validation_pc
        if n_train_unlabeled_pc is not None:
            n_tv += n_train_unlabeled_pc

        if n_tv > n_class_samples:
            raise ValueError('Class {} has not enough samples ({}) to split '
                             'in training labeled, training unlabeled and '
                             'validation set'.format(classes[i], n_class_samples))


        label_subindex = list(label_index[:n_train_labels_pc])
        ind_end = n_train_labels_pc + n_validation_pc
        valid_subindex = list(label_index[n_train_labels_pc:ind_end])
        ind_start = ind_end

        if n_train_unlabeled_pc is not None:
            ind_end += n_train_labels_pc
        else:
            ind_end = n_class_samples

        unlabel_subindex = list(label_index[ind_start:ind_end])
        train_label_index += label_subindex
        valid_label_index += valid_subindex
        train_unlabel_index += unlabel_subindex


    trainset_np = trainset_original.train_data.numpy()
    trainset_label_np = trainset_original.train_labels.numpy()
    train_data_sub = torch.from_numpy(trainset_np[train_label_index])
    train_labels_sub = torch.from_numpy(trainset_label_np[train_label_index])

    trainset_new = subMNIST(root='./../data', train=True, download=True,
                            transform=mnist_transform,
                            k=n_train_labels_pc * n_classes)
    trainset_new.train_data = train_data_sub.clone()
    trainset_new.train_labels = train_labels_sub.clone()

    # pickle.dump(trainset_new, open("./../data/train_labeled.p", "wb"))

    validset_np = trainset_original.train_data.numpy()
    validset_label_np = trainset_original.train_labels.numpy()
    valid_data_sub = torch.from_numpy(validset_np[valid_label_index])
    valid_labels_sub = torch.from_numpy(validset_label_np[valid_label_index])

    validset = subMNIST(root='./../data', train=False, download=True,
                        transform=mnist_transform, k=n_validation_pc * n_classes)
    validset.test_data = valid_data_sub.clone()
    validset.test_labels = valid_labels_sub.clone()

    # pickle.dump(validset, open("./../data/validation.p", "wb"))

    n_unlabeled_set = len(train_unlabel_index)
    trainset_np = trainset_original.train_data.numpy()
    trainset_label_np = trainset_original.train_labels.numpy()
    train_data_sub_unl = torch.from_numpy(trainset_np[train_unlabel_index])
    # train_labels_sub_unl = torch.from_numpy(trainset_label_np[train_unlabel_index])

    trainset_new_unl = subMNIST(root='./../data', train=True, download=True,
                                transform=mnist_transform, k=n_unlabeled_set)
    trainset_new_unl.train_data = train_data_sub_unl.clone()
    trainset_new_unl.train_labels = None      # Unlabeled

    # pickle.dump(trainset_new_unl, open("./../data/train_unlabeled.p", "wb"))

    return trainset_new, trainset_new_unl, validset