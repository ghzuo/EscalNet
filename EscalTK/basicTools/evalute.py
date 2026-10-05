#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:34:28
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2022-10-14 08:44:17
'''


from sklearn import metrics
import numpy as np


# set the testing score function
def entropy(count):
    p = count/np.sum(count)
    return sum(x*np.log(x) for x in p)


def state_length_coef(cl):
    ncl = [0]
    prev = cl[0]
    for i in cl:
        if (i == prev):
            ncl[-1] += 1
        else:
            ncl.append(1)
            prev = i
    return entropy(ncl)/np.log(1/len(ncl))


def cluster(X, cl, uscore=0):
    if (np.max(cl) <= 0):
        return uscore

    # the coeff based on state length
    coef = state_length_coef(cl)

    # the sklearn score
    # score4cls = metrics.calinski_harabasz_score(X, cl)
    # score4cls = -metrics.davies_bouldin_score(X, cl)
    score4cls = metrics.silhouette_score(X, cl)
    return coef*score4cls
