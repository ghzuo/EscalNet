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
@Last Modified Time: 2026-10-01 Thursday 11:17:07
'''

import logging
import numpy as np


def dih2X(file):
    """Read dihedral file and return feature array of
    shape (nframes, nres*4) with per-residue columns
    [PhiCos, PhiSin, PsiCos, PsiSin]."""
    # read the file
    dih = np.loadtxt(file, comments=["@", "#"], usecols=(0, 1))
    nres = np.unique(np.loadtxt(file, comments=["@", "#"],
                                usecols=2, dtype=str)).size
    logging.info(f"The shape of input dihedral is: {dih.shape}")

    # degree -> radian, then reshape to (nframes, nres, 2)
    dih = dih * np.pi / 180
    dih = dih.reshape(-1, nres, 2)  # columns: Phi, Psi

    # per-residue features: [PhiCos, PhiSin, PsiCos, PsiSin]
    X = np.stack([np.cos(dih[:, :, 0]), np.sin(dih[:, :, 0]),
                  np.cos(dih[:, :, 1]), np.sin(dih[:, :, 1])], axis=2)
    return X.reshape(dih.shape[0], nres * 4)


def dihCombind(dihs):
    if dihs.ndim == 1:
        return [np.sqrt(dihs[i] * dihs[i] + dihs[i + 1] * dihs[i + 1])
                for i in range(0, len(dihs), 2)]
    else:
        return np.stack([np.sqrt(dihs[:, i] * dihs[:, i]
                                 + dihs[:, i + 1] * dihs[:, i + 1])
                         for i in range(0, dihs.shape[1], 2)])


def dist2X(file):
    dist = np.loadtxt(file, comments=["@", "#"])
    logging.info(f"The shape of input distance is: {dist.shape}")
    return dist[:, 1:]


def rms(file):
    rms = np.loadtxt(file, comments=["@", "#"],
                     dtype={'names': ('time', 'rms'),
                            'formats': ('f', 'f')})
    logging.info(f"The shape of input rms is: {rms.shape}")
    return rms


def energy(file):
    energy = np.loadtxt(file, comments=["@", "#"],
                        dtype={'names': ('time', 'energy'),
                               'formats': ('f', 'f')})
    logging.info(f"The shape of input energy is: {energy.shape}")
    return energy
