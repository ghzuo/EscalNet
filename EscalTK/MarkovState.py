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
@Last Modified Time: 2026-10-26 Monday 09:44:25
'''

import numpy as np
import pandas as pd


# get markov matrix
def markovMatrix(cl, lag=1):
    nStat = len(np.unique(cl))
    mat = np.zeros((nStat, nStat))
    np.add.at(mat, (cl[:-lag], cl[lag:]), 1)
    mat = (mat + mat.T)/2
    rowSum = mat.sum(axis=1, keepdims=True)
    return (mat/rowSum).T


def stationaryItem(mm, tol=1e-5):
    ex, ev = np.linalg.eig(mm)
    perron_idx = None
    for i, v in enumerate(ex):
        if np.isclose(v.real, 1.0, atol=tol) and np.isclose(
                v.imag, 0.0, atol=tol):
            perron_idx = i
            break
    if perron_idx is None:
        print("No Perronvalue found.")
        return None
    v = ev[:, perron_idx].real
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
        probs[i]['error'] = z * np.sqrt(i*p * (1 - p) / denom)
    return probs


def markov_state_survival(mat, nFrame, utime, ndx=0):
    mat = np.asarray(mat, dtype=float)
    prob = np.zeros([2, nFrame])
    for i in range(0, nFrame):
        prob[0, i] = i * utime
        if i == 0:
            prob[1, i] = 1.0
        else:
            mm = np.linalg.matrix_power(mat, i)
            prob[1, i] = mm[ndx, ndx]
    return prob


def cktest_plot(ax, time, ptraj, pmarkov):
    """Plot the cktest results."""
    pm, = ax.plot(pmarkov[0], pmarkov[1], color='r', lw=5)
    pt = ax.errorbar(time, ptraj['value'], yerr=ptraj['error'], fmt='s',
                     capsize=5, color='b', ecolor='b', markersize=8)
    ax.set_xlabel(r"Time($\mu$s)", fontsize=30, weight="bold")
    ax.set_ylabel("Probability", fontsize=30, weight="bold")
    ax.legend([pt, pm], ["Observed trajectory",
                         "Markov Prediction"], frameon=False)


def multi_cktest_plot(ax, time, ptraj, pmlist):
    """Plot the cktest results."""
    legend = []
    pmline = []
    for nm, val in pmlist.items():
        pm, = ax.plot(val[0], val[1], lw=3)
        legend.append(nm)
        pmline.append(pm)
    pt = ax.errorbar(time, ptraj['value'], yerr=ptraj['error'], fmt='s',
                     capsize=5, color='b', ecolor='b', markersize=8)
    ax.set_xlabel(r"Time($\mu$s)")
    ax.set_ylabel("Probability")
    ax.legend([pt, *pmline], ["Observed trajectory", *legend], frameon=False)


def sortClusterID(clx):
    """Sort the cluster ID as they first appear."""
    istat = pd.DataFrame(np.unique(clx, return_index=True)
                         ).T.sort_values(by=[1])
    mx = pd.Series(index=istat[0], data=np.arange(0, len(istat)))
    return np.array(list(map(lambda e: mx[e], clx)))
