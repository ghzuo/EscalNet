#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2023
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2023-05-20 11:06:05
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2026-10-01 Thursday 14:08:23
'''

import logging
import numpy as np


class LinearRegress:
    def __init__(self, fft):
        self.fft = fft
        self.X = np.c_[np.ones(len(fft.Xx)), fft.Xx]

    def q_multi_cor(self, X, Y):
        t = np.var(Y)
        q = np.linalg.lstsq(X, Y, rcond=None)[1][0]/len(Y)
        return np.sqrt(1 - q/t), t, q

    def score(self, qmclist, kw=0):
        # type the kappa
        for item in qmclist:
            kstart = max(0, int(item[0])-kw) if kw > 0 else 0
            item[1], t, q = self.q_multi_cor(
                self.X, self.fft.multi_exiFFT(item[0], kstart))
            logging.info(f"\t{item[0]:.2f}\t{item[1]:.4f}\t{t:.4f}\t{q:.4f}")

        # the result
        qmc = {'list': qmclist}
        imax = np.argmax(qmclist[:, 1])
        qmc['KappaMax'], qmc['qmcMax'] = qmclist[imax]
        return qmc

    def scale(self, kappa, kw=0):
        kstart = np.max(0, kappa-kw) if kw > 0 else 0
        res = np.linalg.lstsq(
            self.X, self.fft.multi_exiFFT(kappa, kstart))
        result = {'A': res[0][1:]}
        if self.fft.X0 is list:
            result['X'] = [x*result['A'] for x in self.fft.X0]
            result['Ee'] = [np.dot(x, result['A']) for x in self.fft.X0]
        else:
            result['X'] = self.fft.X0*result['A']
            result['Ee'] = np.dot(self.fft.X0, result['A'])
        return result
