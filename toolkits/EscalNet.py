#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
Copyright (c) 2026
See the accompanying Manual for the contributors and the way to
cite this work. Comments and suggestions welcome. Please contact
Dr. Guanghong Zuo <ghzuo@ucas.ac.cn>

@Author: Dr. Guanghong Zuo
@Date: 2026-09-11 Friday 21:12:42
@Last Modified By: Dr. Guanghong Zuo
@Last Modified Time: 2026-10-01 Thursday 22:57:06
'''

import pickle
import argparse
import logging
import os
import numpy as np
from mapTools import encoder as mt
from Etranform import Efft as et
import basicTools as bt
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Obtain the EscalNet rescale representation and "
                    "save it to a Python pickle file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O
    p.add_argument("-f", "--fname", default="Chignolin_375",
                   help="trajectory file prefix (e.g. Chignolin_375 -> "
                        "{fname}-rama.xvg, "
                        "{fname}-potential.xvg, ...)")
    p.add_argument("-o", "--output", default=None,
                   help="output pickle path (default: "
                        "{fname}-{feature}-*.pkl)")

    # feature / resampling
    p.add_argument("--feature", default="dih", choices=["dih", "dist"],
                   help="input feature: dih (ramachandran) or dist "
                        "(Ca-Ca distance)")
    p.add_argument("--skip", type=int, default=1,
                   help="frame stride for resampling")
    p.add_argument("--imax", type=int, default=sys.maxsize,
                   help="max frame index to read (default: use all frames)")

    # network
    p.add_argument("--nFeature", type=int, default=32,
                   help="feature size of the network")
    p.add_argument("--nLayers", type=int, default=3,
                   help="number of hidden layers in the network")
    p.add_argument("--nHidden", type=int, default=None,
                   help="hidden layer size of the network")

    # training
    p.add_argument("--num_epochs", type=int, default=5,
                   help="number of epochs during training")
    p.add_argument("--kfold", type=int, default=5,
                   help="number of folds for cross-validation")
    p.add_argument("--learning_rate", type=float, default=0.01,
                   help="learning rate for EncoderNet")
    p.add_argument("--device", type=str, default=None,
                   help="device for EncoderNet")
    p.add_argument("--decorr", type=float, default=0.01,
                   help="correlation penalty for between features")
    p.add_argument("--balance", type=float, default=0.01,
                   help="unbalance penalty for feature")

    # options for data loading
    p.add_argument("--batch_size", type=int, default=128,
                   help="minibatch size for EncoderNet")
    p.add_argument("--shuffle", action="store_false",
                   help="do not shuffle the dataset")
    p.add_argument('--num_workers', type=int, default=4,
                   help='Number of workers for data loading (default: 4)')
    p.add_argument('--pin_memory', action='store_false',
                   help='Disable pin memory for faster GPU transfer')
    p.add_argument('--prefetch_factor', type=int, default=2,
                   help='Prefetch factor for data loading (default: 2)')
    p.add_argument('--persistent_workers', action='store_false',
                   help='Disable persistent workers for multiple epochs')

    # kappa
    p.add_argument("--score_kmax", type=int, default=40,
                   help="max kappa for score scan")
    p.add_argument("--score_kmin", type=int, default=2,
                   help="min kappa for score scan")
    p.add_argument("--score_kstep", type=int, default=None,
                   help="kappa step for score scan")
    p.add_argument("--kappa", type=int, default=None,
                   help="kappa for scale X")
    p.add_argument("--cache", default=None,
                   help="cache directory for the model")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress info-level logging output")

    args = p.parse_args()

    # set output path
    if args.output is None:
        args.output = f"{os.path.basename(args.fname)}-{args.feature}-MLP{args.nLayers}L{args.nFeature}F"
    elif (args.output.endswith("/")):
        args.output += f"{os.path.basename(args.fname)}-{args.feature}-MLP{args.nLayers}L{args.nFeature}F"
    # set model cache path
    if args.cache is None:
        args.cache = os.path.join(os.path.dirname(args.output), "models",
                                  os.path.basename(args.output))
    elif (args.cache.endswith("/")):
        args.cache += os.path.basename(os.path.basename(args.output))
    # set score_kstep
    if args.score_kstep is None:
        args.score_kstep = (args.score_kmax - args.score_kmin) // 64 + 1
    # configure logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s")

    return args


def prepare_data(args):
    """Prepare the data for the network."""
    # read feature
    if args.feature == "dih":
        X0 = bt.feature.dih2X(f"{args.fname}-rama.xvg")
    else:
        X0 = bt.feature.dist2X(f"{args.fname}-CaDist.xvg")
    # read energy
    eng = bt.feature.energy(f"{args.fname}-potential.xvg")
    # read rms
    rms = bt.feature.rms(f"{args.fname}-CaRMS.xvg")

    # align the feature and energy
    imax = min(args.imax, len(eng), len(X0), len(rms))
    X0 = X0[1:imax:args.skip]
    eng = eng[1:imax:args.skip]
    rms = rms[1:imax:args.skip]

    # base frequency in MHz (1000000 for MHz and us, 0.5 for even extension)
    tTotal = eng[-1][0] / 1000000.0
    ufreq = 0.5 / tTotal
    E0 = eng['energy']
    logging.info(f"The total time: {tTotal:.4f} us")
    logging.info(f"The base frequence: {ufreq:.4f} MHz")

    # build Efft
    efft = et(X0, E0, ufreq)

    return efft, rms


def main(args):
    logging.info(f"The analysis will do for: {args.fname}")
    logging.info(f"Output path: {args.output}.pkl")
    logging.info(f"Model cache path: {args.cache}-K<kappa>.pt\n")

    # --- setup data ---
    datafile = f"{args.output}.pkl"
    if os.path.exists(datafile):
        logging.info(f"\n=== Load data from file: {datafile} ===")
        data = pickle.load(open(datafile, "rb"))
        efft = data["efft"]
        data["args"] = args
    else:
        efft, rms = prepare_data(args)
        data = {"rms": rms,
                "efft": efft,
                "args": args
                }

    # --- setup the network ---
    nInput = efft.Xx.shape[1]
    net = mt.MLPxL(nInput, nFeature=args.nFeature,
                   nHidden=args.nHidden, nLayers=args.nLayers)
    data["net"] = net.__name__

    # --- setup model and train ---
    # set performance optimization arguments
    kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'prefetch_factor': args.prefetch_factor,
        'persistent_workers': args.persistent_workers,
        'batch_size': args.batch_size
    }

    xmap = mt.EncoderNet(net, efft, lr=args.learning_rate,
                         cachePref=args.cache, shuffle=args.shuffle,
                         num_epochs=args.num_epochs, kfold=args.kfold,
                         decorr=args.decorr, balance=args.balance,
                         device=args.device, **kwargs)
    xmap.info()

    # --- score, scale and saliency ---
    if args.kappa is None:
        # score scan
        klist = np.arange(args.score_kmin, args.score_kmax +
                          1, args.score_kstep, dtype=int)
        qmclist = np.array([klist, np.full(len(klist), np.nan, dtype=float)]).T
        if "qmc" in data:
            lookup = dict(zip(data["qmc"]["list"][:, 0],
                          data["qmc"]["list"][:, 1]))
            new_col = np.array([lookup.get(k, np.nan) for k in qmclist[:, 0]])
            qmclist[:, 1] = new_col
        data["qmc"] = xmap.score(qmclist)
        args.kappa = data["qmc"]["KappaMax"]
        # get/save model parameters
        xmap.oneKappa(args.kappa)
    else:
        # force to retrain and get model parameters
        xmap.oneKappa(args.kappa, retrain=True)
    data['rescl'], data['saliency'] = xmap.scale()

    # --- save result to the pkl file ---
    with open(datafile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(
        f"\n=== The result has been written to: {args.output}.pkl ===")


if __name__ == "__main__":
    args = parse_args()
    main(args)
