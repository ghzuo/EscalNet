#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2023
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2023-05-20 13:55:16
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2026-10-01 Thursday 12:38:58
'''

import sys
import logging
import numpy as np
import torch
import copy
import os
from sklearn.model_selection import KFold


def normlize(X):
    Xm = X.mean(dim=0)
    Xs = X.std(dim=0)
    Xx = (X - Xm)/Xs
    return Xx, Xm, Xs


def deNormlize(Xx, Xm, Xs):
    return Xx*Xs + Xm


class ModKFold:
    """K-Fold like splitter using index % n_splits to partition data.

    Unlike sklearn's KFold which splits data into contiguous blocks,
    this class interleaves samples by their index modulo n_splits.
    For the k-th fold, samples whose index % n_splits == k form the
    test set, while the remaining samples form the training set.
    """

    def __init__(self, n_splits):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits

    def get_n_splits(self, X=None):
        return self.n_splits

    def split(self, X):
        n = len(X)
        indices = np.arange(n)
        for k in range(self.n_splits):
            test_index = indices[indices % self.n_splits == k]
            train_index = indices[indices % self.n_splits != k]
            yield train_index, test_index


class EncoderNet:
    def __init__(self, net, fft,
                 cachePref=None,
                 kfold=5, num_epochs=10,
                 loss=torch.nn.MSELoss(),
                 batch_size=50, shuffle=True,
                 optim=torch.optim.Adam, lr=0.001,
                 balance=0.01, decorr=0.01,
                 device=None):
        self.net = net
        self.fft = fft
        self.cachePref = cachePref
        os.makedirs(os.path.dirname(cachePref), exist_ok=True)
        self.kfold = kfold
        self.num_epochs = num_epochs
        self.loss = loss
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance = balance
        self.decorr = decorr
        self.optim = optim(self.net.parameters(), lr, weight_decay=0.0005)
        self.X = torch.tensor(self.fft.Xx, dtype=torch.float)
        # set device
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.net.to(device)

    def info(self):
        logging.info("\n=== EncoderNet Info ===")
        logging.info(f"--- The input data:\n*** The X shape: {self.X.shape}")
        logging.info(f"*** The FFT base shape: {self.fft.F[0].shape}")
        logging.info(f"\n--- The network is:\n{self.net}")
        logging.info(f"\n--- The training information is as follows:")
        logging.info(f"*** KFold: {self.kfold}")
        logging.info(f"*** Num Epochs: {self.num_epochs}")
        logging.info(f"*** Loss: {self.loss.__class__.__name__}")
        logging.info(f"*** Batch Size: {self.batch_size}")
        logging.info(f"*** Shuffle: {self.shuffle}")
        logging.info(f"*** Optimizer: {self.optim.__class__.__name__}")
        logging.info(
            f"*** Learning Rate: {self.optim.param_groups[0]['lr']}")
        logging.info(f"*** Balance: {self.balance}")
        logging.info(f"*** Decorrelation: {self.decorr}")
        logging.info(f"*** Device: {self.device}")
        if self.cachePref is not None:
            logging.info(f"*** Cache Pref: {self.cachePref}")
        else:
            logging.info("*** Don't cache models.")

    def setY(self, kappa):
        self.y, self.ym, self.ys = normlize(
            torch.tensor(self.fft.multi_exiFFT(kappa), dtype=torch.float))

    def _train(self, dtloader):
        self.net.train()
        running_loss = 0.0
        # collect feature activations for feature regularization
        feat_holder = [None]
        handle = None
        if self.balance > 0 or self.decorr > 0:
            handle = self.net.output.register_forward_hook(
                hook=lambda m, fin, fout: feat_holder.__setitem__(0, fin[0]))
        for X, y in dtloader:
            # set device
            X = X.to(self.device)
            y = y.to(self.device)
            # Forward pass
            output = self.net(X)
            loss = self.loss(output, y.view(-1, 1))
            # feature regularization
            feat = feat_holder[0]
            if feat is not None:
                feat_c = feat - feat.mean(dim=0, keepdim=True)
                cov = (feat_c.T @ feat_c) / feat.size(0)
                if self.decorr > 0:
                    offdiag = cov - torch.diag(cov.diag())
                    loss = loss + self.decorr * (offdiag ** 2).sum()
                if self.balance > 0:
                    loss = loss + self.balance * cov.diag().var()
            running_loss += loss.item()
            # Backward pass
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        if handle is not None:
            handle.remove()
        return running_loss / len(dtloader)

    def _validate(self, dtloader):
        self.net.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X, y in dtloader:
                # set device
                X = X.to(self.device)
                y = y.to(self.device)
                # Forward pass
                output = self.net(X)
                loss = self.loss(output, y.view(-1, 1))
                running_loss += loss.item()
        return running_loss / len(dtloader)

    def train(self, X_train, y_train, X_valid, y_valid):
        # set train dataloader
        train_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=self.batch_size,  # mini batch size
            shuffle=self.shuffle,
        )

        # set valid dataloader
        valid_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X_valid, y_valid),
            batch_size=self.batch_size,  # mini batch size
            shuffle=False,  # no shuffle for valid
        )

        # for the best model
        lowest_loss = sys.float_info.max
        best_model = None
        # start training
        for epoch in range(self.num_epochs):
            # training, validating and evaluating
            train_loss = self._train(train_loader)
            val_loss = self._validate(valid_loader)
            # save the best model
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                best_model = copy.deepcopy(self.net.state_dict())
            # output epoch info
            logging.info(
                f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}")
        self.net.load_state_dict(best_model)
        return lowest_loss

    def test(self, X_test, y_test):
        # set test dataloader
        test_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=self.batch_size,  # mini batch size
            shuffle=False,  # no shuffle for test
        )
        return self._validate(test_loader)

    def trainOneKappa(self, kappa=2):
        # set dataset
        self.setY(kappa)
        best_loss = sys.float_info.max
        best_model = None
        logging.info(f"\n=== Start training for kappa = {kappa:.0f} ===")

        if (self.kfold < 2):
            logging.info("No KFold, train and validate on all data")
            best_loss = self.train(self.X, self.y, self.X, self.y)
        else:
            # kfold = KFold(n_splits=self.kfold, shuffle=True, random_state=42)
            kfold = ModKFold(self.kfold)
            for i, (train_index, test_index) in enumerate(kfold.split(self.X)):
                logging.info(f"--- Fold {i+1}/{self.kfold} ---")
                X_train, X_valid = self.X[train_index], self.X[test_index]
                y_train, y_valid = self.y[train_index], self.y[test_index]
                self.train(X_train, y_train, X_valid, y_valid)
                # get the best model for every fold
                test_loss = self.test(self.X, self.y)
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model = copy.deepcopy(self.net.state_dict())
            self.net.load_state_dict(best_model)
        logging.info(f"=== Training for kappa = {kappa:.0f} is done ===")

        if self.cachePref is not None:
            netpt = f"{self.cachePref}-K{kappa:.0f}.pt"
            torch.save(self.net.state_dict(), netpt)
            logging.info(f"*** Save best model to: {netpt}")
        return best_loss

    def oneKappa(self, kappa=2, retrain=False):
        if retrain or self.cachePref is None:
            best_loss = self.trainOneKappa(kappa)
        else:
            netpt = f"{self.cachePref}-K{kappa:.0f}.pt"
            if not os.path.exists(netpt):
                best_loss = self.trainOneKappa(kappa)
            else:
                self.setY(kappa)
                self.net.load_state_dict(torch.load(netpt, weights_only=True))
                best_loss = self.test(self.X, self.y)
                logging.info(f"\n=== Loading the best model from: {netpt} ===")
    
        r = np.sqrt(1 - best_loss) if best_loss < 1 else 0
        logging.info(f"*** Best Model for kappa = {kappa:.0f} | Loss: {best_loss:.4f} "
                    f"| Coefficient: {r:.4f}")
        return r

    def score(self, qmclist):
        # Calculate the QMC from the qmc dict.
        logging.info(
            f"\n=== Start scoring for all kappas ===")
        for item in qmclist:
            if np.isnan(item[1]):
                item[1] = self.oneKappa(item[0])
            else:
                logging.info(f"*** Skip kappa = {item[0]:.0f} as it has been set as {item[1]:.4f}")
        qmc = {'list': qmclist}
        # Get the peak value of qmc
        qmc['KappaMax'], qmc['qmcMax'] = qmclist[np.argmax(qmclist[:, 1])]
        logging.info(
            f"Best kappa = {qmc['KappaMax']:.0f}, QMC = {qmc['qmcMax']:.4f}")
        return qmc

    def scale(self):
        # work on a CPU copy to avoid cuBLAS context warnings;
        # the original self.net stays on its training device untouched.
        net_cpu = copy.deepcopy(self.net).to("cpu")
        X_cpu = self.X.clone()
        # add hook on features layer
        features = []
        handle = net_cpu.output.register_forward_hook(
            hook=lambda module, fin, fout: features.append(fin))
        # get the effect energy and features
        Ee = deNormlize(net_cpu(X_cpu), self.ym, self.ys).detach().numpy()
        rescl = {
            "Ee": np.split(Ee, len(self.fft.X0)) if self.fft.X0 is list else Ee,
            'X': (features[0][0] * net_cpu.output.weight).detach().numpy(),
            'A': net_cpu.output.weight.detach().numpy()[0]
        }
        handle.remove()
        # output the rescaled info
        logging.info(f"\n=== The result ===\n*** rescl keys: {list(rescl.keys())}")
        for k, v in rescl.items():
            arr = np.asarray(v)
            logging.info(
                f"*** rescl['{k}']: shape={arr.shape}, dtype={arr.dtype}")

        # get saliency (reuse the same CPU copy)
        saliency = None
        if self.net.__name__ != 'LR':
            net_cpu.eval()
            Xd = X_cpu.requires_grad_(True)
            ys = net_cpu.forward(Xd)
            ys.backward(torch.ones_like(ys))
            saliency = abs(Xd.grad.detach().numpy())
            logging.info(f"*** saliency shape: {saliency.shape}")

        return rescl, saliency


# the square layer
class Square(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(in_features*2, out_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features,))

    def forward(self, x):
        xx = torch.square(x)
        return torch.matmul(torch.concat((xx, x), 1),
                            self.weight.detach()) + self.bias.detach()


# the linear regression
class LR(torch.nn.Module):
    def __init__(self, nInput, **kwargs):
        super(LR, self).__init__(**kwargs)
        self.__name__ = 'LR'
        self.output = torch.nn.Linear(nInput, 1)

    def forward(self, x):
        return self.output(x)


# the multi-layer Perceptron
class MLP2L(torch.nn.Module):
    def __init__(self, nInput, nFeature=2, **kwargs):
        super(MLP2L, self).__init__(**kwargs)
        self.__name__ = 'MLP2L'
        nHidden = int((nInput + nFeature)/2)
        actfunc = torch.nn.ELU
        self.input = torch.nn.Linear(nInput, nHidden)
        self.actI = actfunc()
        self.feature = torch.nn.Linear(nHidden, nFeature)
        self.actF = actfunc()
        self.output = torch.nn.Linear(nFeature, 1)

    def forward(self, x):
        outI = self.actI(self.input(x))
        outF = self.actF(self.feature(outI + x))
        return self.output(outF)


class MLP3L(torch.nn.Module):
    def __init__(self, nInput, nFeature=20, **kwargs):
        super(MLP3L, self).__init__(**kwargs)
        self.__name__ = 'MLP3L'
        nHidden = int((nInput + nFeature)/2)
        actfunc = torch.nn.ELU
        self.input = torch.nn.Linear(nInput, nHidden)
        self.actI = actfunc()
        self.hidden = torch.nn.Linear(nHidden, nHidden)
        self.actH = actfunc()
        self.feature = torch.nn.Linear(nHidden, nFeature)
        self.actF = actfunc()
        self.output = torch.nn.Linear(nFeature, 1)

    def forward(self, x):
        outI = self.actI(self.input(x))
        outH = self.actH(self.hidden(outI))
        outF = self.actF(self.feature(outH))
        return self.output(outF)
