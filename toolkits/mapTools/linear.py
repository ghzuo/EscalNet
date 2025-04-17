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
@Last Modified Time: 2023-06-04 16:33:09
'''

import numpy as np


class LinearRegress:
    def __init__(self, fft, info=0):
        self.fft = fft
        self.info = info

    def q_multi_cor(self, X, Y):
        t = np.var(Y)
        q = np.linalg.lstsq(X, Y, rcond=None)[1][0]/len(Y)
        return np.sqrt(1 - q/t), t, q

    def score(self, kmax=0, kmin=2, nk=100, kw=0):
        # type the kappa
        kmax = len(self.fft.F[0]) if kmax < kmin else min(
            [kmax, len(self.fft.F[0])])
        kmin = max(kmin, 2)
        ksep = int((kmax - kmin)/nk)+1
        klist = np.arange(kmin, kmax, ksep)
        qmc = np.array([klist, np.zeros(len(klist))]).T
        for item in qmc:
            kstart = max(0, int(item[0])-kw) if kw > 0 else 0
            item[1], t, q = self.q_multi_cor(
                self.fft.Xx, self.fft.multi_exiFFT(item[0], kstart))
            if (self.info > 0):
                print(item[0], item[1], t, q, sep="\t")

        # the result
        result = {'list': qmc}
        imax = np.argmax(qmc[:, 1])
        result['KappaMax'], result['qmcMax'] = qmc[imax]
        return result

    def scale(self, kappa, kw=0):
        kstart = np.max(0, kappa-kw) if kw > 0 else 0
        res = np.linalg.lstsq(
            self.fft.Xx, self.fft.multi_exiFFT(kappa, kstart))
        result = {'A': res[0][1:]}
        if self.fft.X is list:
            result['X'] = [x*result['A'] for x in self.fft.X]
            result['Ee'] = [np.dot(x, result['A']) for x in self.fft.X]
        else:
            result['X'] = self.fft.X*result['A']
            result['Ee'] = np.dot(self.fft.X, result['A'])
        return result
