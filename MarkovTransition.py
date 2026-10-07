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
@Last Modified Time: 2026-10-05 Monday 15:22:43
'''

import pickle
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os

from EscalTK import basicTools as bt
from EscalTK import MarkovState as ms

def parse_args():
    p = argparse.ArgumentParser(
        description="Get the clustering/transition of the rescaled "
                    "trajectory (rescl) from the pickle file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O
    p.add_argument("-f", "--datafile", required=True,
                   help="input pickle file")
    p.add_argument("-o", "--output", default=None,
                   help="output file")

    # smooth
    p.add_argument("--step", type=int, default=100,
                   help="step for segment sampling")
    p.add_argument("--width", type=int, default=200,
                   help="window width for segment sampling")

    # clustering
    p.add_argument("-n", "--ncls", type=int, default=3,
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
    p.add_argument("--lag_max", type=int, default=100,
                   help="max lag for the survival probability test")
    p.add_argument("--lag_step", type=int, default=3,
                   help="step size for the survival probability test")

    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress info-level logging output")

    # configure logging
    args = p.parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s")
    if args.output is None:
        args.output = f"{os.path.basename(args.datafile).split('.')[0]}-cktest.pdf"
    elif (args.output.endswith("/")):
        args.output += f"{os.path.basename(args.datafile).split('.')[0]}-cktest.pdf"

    return args


def read_traj(datafile):
    """Read the rescaled trajectory (rescl) from the pickle file."""
    with open(datafile, 'rb') as f:
        obj = pickle.load(f)
    return obj["rescl"], 0.5/obj["efft"].ufreq


def sortClusterID(clx):
    """Sort the cluster ID as they first appear."""
    istat = pd.DataFrame(np.unique(clx, return_index=True)
                         ).T.sort_values(by=[1])
    mx = pd.Series(index=istat[0], data=np.arange(0, len(istat)))
    return np.array(list(map(lambda e: mx[e], clx)))



def cktest_plot(ax, time, ptraj, pmarkov):
    """Plot the cktest results."""
    pm, = ax.plot(time, pmarkov, color='r', lw=5)
    pt = ax.errorbar(time, ptraj['value'], yerr=ptraj['error'], fmt='s',
                     capsize=5, color='b', ecolor='b', markersize=8)
    ax.set_xlabel(r"Time($\mu$s)", fontsize=30, weight="bold")
    ax.set_ylabel("Probability", fontsize=30, weight="bold")
    ax.legend([pt, pm], ["Observed trajectory",
              "Markov Prediction"], frameon=False)


def main():
    args = parse_args()
    seg2vec = bt.segment.mean

    rescl, tTotal = read_traj(args.datafile)
    Xys = seg2vec(rescl['X'], args.step, args.width)
    uTime = tTotal/len(Xys)
    cl = KMeans(n_clusters=args.ncls, n_init=args.n_init).fit_predict(Xys)
    cl = sortClusterID(cl)

    # transition matrix and markov tests
    T = ms.markovMatrix(cl, args.t_lag)
    states = [f"S{x}" for x in range(0,len(T))]
    df = pd.DataFrame(T,  index=states, 
                      columns=pd.MultiIndex.from_tuples(
                      [("Markov Matrix", x) for x in states]))
    df[('Distribution','Stationary')] = ms.stationaryItem(T)
    df[('Distribution','Counting')] = ms.histogram(cl)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    logging.info(f"\n*** The Markov Statistics for (t_lag={args.t_lag}) is:\n{df}")

    # get survival probability
    lags = np.arange(0, args.lag_max, args.lag_step)
    time = lags * uTime
    ptraj = ms.traj_stat_survival(cl, lags, args.ndx, args.z)
    # get markov state survival probability
    pmarkov = ms.markov_state_survival(T, lags, args.ndx)

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cktest_plot(ax, time, ptraj, pmarkov)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"\n*** The cktest results are saved to {args.output}")


if __name__ == "__main__":
    main()
