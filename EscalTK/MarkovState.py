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

import numpy as np


# get markov matrix
def markovMatrix(cl, lag=1):
    nStat = len(np.unique(cl))
    mat = np.zeros((nStat, nStat))
    np.add.at(mat, (cl[:-lag], cl[lag:]), 1)
    mat = (mat + mat.T)/2
    rowSum = mat.sum(axis=1, keepdims=True)
    return (mat/rowSum).T


def stationaryItem(mm):
    ex, ev = np.linalg.eigh(mm)
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

