#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:30:05
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2024-04-11 19:22:42
'''

import numpy as np


def Lmat(Smat):
    Dmat = np.diag(np.sum(Smat, axis=0))
    return Dmat - Smat


def NLmat_rw(Smat):
    Dmat = np.diag(np.sum(Smat, axis=0))
    Lmat = Dmat - Smat
    return np.linalg.inv(Dmat) * Lmat


def NLmat_sym(Smat):
    Dmat = np.diag(np.sum(Smat, axis=0))
    Lmat = Dmat - Smat
    iDmat = np.linalg.inv(Dmat)
    iDmatSqrt = np.sqrt(iDmat)
    return iDmatSqrt * Lmat * iDmatSqrt
