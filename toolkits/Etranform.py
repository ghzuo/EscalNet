#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-08-21 22:32:17
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-06-02 09:26:35
'''


import numpy as np


class Efft:
    def __init__(self, X, E, base=True):
        self.X = X
        if all([type(X) is list, type(E) is list]):
            Xx = np.concatenate(X)
            self.F = [self.exFFT(td) for td in E]
        else:
            Xx = X
            self.F = [self.exFFT(E)]
        self.Xx = np.c_[np.ones(len(Xx)), Xx] if base else Xx

    def exFFT(self, tdata):
        exdata = np.append(tdata, tdata[-1::-1])  # even extension
        return np.fft.rfft(exdata)

    def exiFFT(self, fdata, kappa, kstart=0):
        kappa = int(kappa)
        kstart = int(kstart)
        noutput = len(fdata) - 1
        exdata = np.fft.irfft(fdata[kstart:kappa], 2*noutput)
        return exdata[:noutput]

    def Ef(self, kappa, kstart=0):
        if type(self.F) is list:
            return [self.exiFFT(f, kappa, kstart) for f in self.F]
        else:
            return self.exiFFT(self.F, kappa)

    def multi_exiFFT(self, kappa, kstart=0):
        return np.concatenate([self.exiFFT(f, kappa, kstart) for f in self.F])
