#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:25:17
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2024-04-10 16:59:32
'''


import numpy as np


# for distance matrix
def expm(data, lam=0.01):
    mat = []
    for i in data:
        mat.extend(np.exp(- lam*np.linalg.norm(i - j)) for j in data)
    return np.array(mat).reshape(len(data), len(data))


def manhattan(data):
    mat = []
    for i in data:
        mat.extend(np.sum([abs(vi-vj) for vi, vj in zip(i, j)]) for j in data)
    return np.array(mat).reshape(len(data), len(data))


def euclidean(data):
    mat = [np.linalg.norm(data-x, axis=1) for x in data]
    return np.array(mat)


def norm_rows(Mx):
    norm = np.linalg.norm(Mx, axis=1)
    norm = norm.reshape([len(norm), 1])
    Mx = Mx/norm
    return Mx


def cosin(data, plus=False):
    if plus:
        data = np.concatenate([data, np.ones(data.shape)], axis=1)
    data = norm_rows(data)
    mat = np.dot(data, data.T)
    mat = (mat + 1)/2
    return mat
