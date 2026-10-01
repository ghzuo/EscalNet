#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2026
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2026-09-17 Thursday 18:38:41
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2026-09-30 Wednesday 15:51:10
'''

import pickle
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import basicTools as bt
import outplot as op


def parse_args():
    p = argparse.ArgumentParser(
        description="Get the clustering/transition of the rescaled "
                    "trajectory (rescl) from the pickle file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O
    p.add_argument("-f", "--datafile", required=True,
                   help="input pickle file from get_representation.py")
    p.add_argument("-o", "--output", default=None,
                   help="output directory")

    # smooth
    p.add_argument("--step", type=int, default=100,
                   help="step for segment sampling")
    p.add_argument("--width", type=int, default=200,
                   help="window width for segment sampling")

    # clustering
    p.add_argument("-n", "--ncls", type=int, default=34,
                   help="number of clusters for KMeans")
    p.add_argument("--n_init", type=int, default=50,
                   help="number of initializations for KMeans")

    # markov tests
    p.add_argument("--t_lag", type=int, default=1,
                   help="lag time (in units of segments) for the markov "
                        "tests")
    p.add_argument("--z", type=float, default=1.96,
                   help="z-value for the survival probability test")
    p.add_argument("--ndx", type=int, default=0,
                   help="index of the state for the survival probability test")
    p.add_argument("--lag_max", type=int, default=50,
                   help="max lag for the survival probability test")
    p.add_argument("--lag_step", type=int, default=1,
                   help="step size for the survival probability test")
    return p.parse_args()


def read_traj(datafile):
    """Read the rescaled trajectory (rescl) from the pickle file."""
    with open(datafile, 'rb') as f:
        obj = pickle.load(f)
    return obj["rescl"]


# get markov matrix
def markovMatrix(cl, lag=1):
    nStat = len(np.unique(cl))
    mat = np.zeros((nStat, nStat))
    np.add.at(mat, (cl[:-lag], cl[lag:]), 1)
    mat = (mat + mat.T)/2
    rowSum = mat.sum(axis=1, keepdims=True)
    return (mat/rowSum).T


def stationaryItem(mm):
    ex, ev = np.linalg.eig(mm)
    v = ev[:, np.argmax(ex)]
    return v/v.sum()


def histogram(cl):
    _, counts = np.unique(cl, return_counts=True)
    prob = counts / len(cl)
    return prob


def traj_stat_survival(cl, lags, ndx=0, z=1.96):
    cl = np.asarray(cl)
    n = len(cl)
    tmax = lags[-1]
    if tmax >= n:
        raise ValueError("tmax must be smaller than the length of cl")
    nlen = n - tmax
    start_mask = cl[:nlen] == ndx
    denom = start_mask.sum()
    probs = np.zeros(len(lags), dtype=[('value', float), ('error', float)])
    for i, t in enumerate(lags):
        end_mask = cl[t: t + nlen] == ndx
        p = (start_mask & end_mask).sum() / denom
        probs[i]['value'] = p
        probs[i]['error'] = z * np.sqrt(p * (1 - p) / denom)
    return probs


def markov_state_survival(mat, lags, ndx=0):
    mat = np.asarray(mat, dtype=float)
    prob = np.zeros_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        if lag == 0:
            prob[i] = 1.0
        else:
            mm = np.linalg.matrix_power(mat, lag)
            prob[i] = mm[ndx, ndx]
    return prob


def main():
    args = parse_args()
    seg2vec = bt.segment.mean

    rescl = read_traj(args.datafile)
    Xys = seg2vec(rescl['X'], args.step, args.width)
    cl = KMeans(n_clusters=args.ncls, n_init=args.n_init).fit_predict(Xys)

    # transition matrix and markov tests
    T = markovMatrix(cl, args.t_lag)
    logging.info(f"The transition matrix (t_lag={args.t_lag}) is:\n{T}")

    # get survival probability
    lags = np.arange(0, args.lag_max, args.lag_step)
    ptraj = traj_stat_survival(cl, lags, args.ndx, args.z)
    # get markov state survival probability
    pmarkov = markov_state_survival(T, args.ndx, lags)

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    op.cktest_plot(ax, lags, ptraj, pmarkov)
    plt.savefig(args.output, bbox_inches='tight')


if __name__ == "__main__":
    main()
