#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:36:20
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-07-24 20:31:59
'''

import numpy as np
import pandas as pd


def dih2X(file, quiet=False):
    # read the file
    dih = pd.DataFrame(np.loadtxt(file, comments=["@", "#"],
                                  dtype={'names': ('Phi', 'Psi', 'Residue'),
                                         'formats': ('f', 'f', 'S7')}))
    if not quiet:
        print("The shape of input dihedral is: ", dih.shape)

    # set the dih
    nres = dih['Residue'].nunique()
    dih['frame'] = dih.index // nres
    dih['index'] = dih.index % nres
    dih['Psi'] = dih['Psi']*np.pi/180
    dih['Phi'] = dih['Phi']*np.pi/180

    # set the dataframe of X
    X = dih[["frame", "index"]]
    X['PhiCos'] = np.cos(dih['Phi'])
    X['PhiSin'] = np.sin(dih['Phi'])
    X['PsiCos'] = np.cos(dih['Psi'])
    X['PsiSin'] = np.sin(dih['Psi'])
    return X.melt(id_vars=["frame", "index"],
                  value_vars=["PhiCos", "PhiSin", "PsiCos", "PsiSin"]
                  ).pivot_table(index="frame", columns=["index", "variable"])


def dihCombind(dihs):
    if dihs.ndim == 1:
        return [np.sqrt(dihs[i] * dihs[i] + dihs[i + 1] * dihs[i + 1])
                for i in range(0, len(dihs), 2)]
    else:
        return np.stack([np.sqrt(dihs[:, i] * dihs[:, i]
                                 + dihs[:, i + 1] * dihs[:, i + 1])
                         for i in range(0, dihs.shape[1], 2)])


def dist2X(file, quiet=False):
    dist = pd.DataFrame(np.loadtxt(file, comments=["@", "#"]))
    if not quiet:
        print("The shape of input distance is: ", dist.shape)
    return dist.loc[:, 1:]
