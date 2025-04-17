#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2022
Wenzhou Institute, University of Chinese Academy of Sciences.
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2022-07-03 16:18:21
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2023-07-24 20:12:30
'''


# Basic set of Python Data Analysis
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import seaborn as sns
import numpy as np
import pandas as pd


def implot(mat, ax=None):
    if ax is None:
        ax = plt.subplot()
    im = ax.imshow(mat, interpolation='nearest', cmap=cm.jet)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)


def feplot(mat, ax=None):
    if ax is None:
        ax = plt.subplot()
    im = ax.contourf(mat, cmap=cm.jet)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)


def distr_plot(feature):
    xd = feature[:, 0]
    yd = feature[:, 1]
    plt.figure(12, figsize=(16, 7))
    ax = plt.subplot(121)
    ax.scatter(xd, yd, s=5)
    ax = plt.subplot(122)
    hist = np.histogram2d(xd, yd, 20)
    fe = -np.log(hist[0]/len(feature))
    cutoff = np.percentile(fe, 80)
    fe[fe > cutoff] = np.nan
    xl = (hist[1][1:] + hist[1][:-1])/2
    yl = (hist[2][1:] + hist[2][:-1])/2
    im = plt.contourf(xl, yl, fe.T, levels=6, cmap=cm.rainbow)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(im, cax=cax)


def coefplot(ys, ax=None):
    if ax is None:
        ax = plt.subplot()
    xs = np.arange(1, len(ys)+1)
    ax.plot(xs, ys)
    for x, y in zip(xs, ys):
        ax.plot(x, y, 'o', ms=20, lw=2, alpha=0.7, mfc='yellow')
        ax.text(x, y, '%d' % (int(x)), fontsize=12,
                horizontalalignment='center', verticalalignment='center')


def saliplot(ys, ax=None):
    if ax is None:
        ax = plt.subplot()
    _ = ax.boxplot(ys, patch_artist=True, showfliers=False, showmeans=True,
                   medianprops={'lw': 2, 'color': 'orange'},
                   meanprops={'marker': 'o', 'mec': 'yellow',
                              'mfc': 'yellow', 'markersize': 10})
    ax.xaxis.set_tick_params(rotation=45)


def e3traj(E0, Ef, Ey):
    scale = (np.sqrt((np.var(E0)/np.var(Ey)))//10)*10
    plt.plot((E0-np.mean(E0))/scale, 'c-',
             label="Raw Data (/{:.0f})".format(scale))
    plt.plot(Ey-np.mean(Ey), "r-", label="Effective Energy")
    Efmean = np.mean(Ef)
    plt.plot(Ef[0]-Efmean, "b-", lw=5, label="iFFT Energy")
    xbeg = len(Ef[0])
    for ef in Ef[1:]:
        xend = xbeg + len(ef)
        plt.plot(range(xbeg, xend), ef-Efmean, "b-", lw=5)
        xbeg = xend


def efplot(E0, Ef, Ey):
    plt.figure(22, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.25)
    plt.subplot(121)
    scale = (np.sqrt((np.var(E0)/np.var(Ey)))//10)*10
    plt.plot((E0-np.mean(E0))/scale, 'c-',
             label="Raw Data (/{:.0f})".format(scale))
    plt.plot(Ey-np.mean(Ey), "r-", label="Effective Energy")
    Efmean = np.mean(Ef)
    plt.plot(Ef[0]-Efmean, "b-", lw=5, label="iFFT Energy")
    xbeg = len(Ef[0])
    for ef in Ef[1:]:
        xend = xbeg + len(ef)
        plt.plot(range(xbeg, xend), ef-Efmean, "b-", lw=5)
        xbeg = xend
    plt.legend()

    plt.subplot(222)
    sns.histplot(Ey, kde=True, color='red')
    plt.subplot(224)
    sns.histplot(E0, kde=True, color='cyan')


def clplot(score, cl, lab):
    # sort the clusters
    a = pd.DataFrame(np.unique(cl, return_index=True)).T.sort_values(by=[1])
    mx = pd.Series(index=a[0], data=np.arange(0, len(a))+1)
    clx = list(map(lambda e: mx[e], cl))

    # plot data
    plt.figure(22, figsize=(16, 7))
    ax = plt.subplot(121)
    score.plot(marker="o", ax=ax)

    plt.subplot(222)
    plt.plot(clx, "ro-", ms=8, label=lab)
    plt.legend(handlelength=0, markerscale=0)

    plt.subplot(224)
    (pd.value_counts(clx, sort=False)/len(clx)).plot.bar(rot=0)
