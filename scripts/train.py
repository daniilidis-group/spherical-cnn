#!/usr/bin/env python3
""" Main training script """

import os
import logging
import pickle
import glob
import socket

import numpy as np
import tensorflow as tf

# argh!
import sys
from os.path import abspath, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from spherical_cnn import params
from spherical_cnn import models
from spherical_cnn import datasets


def init_logger(args):
    # create logger
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    # create file handler
    logdir = os.path.expanduser(args.logdir)
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, 'logging.log')
    fh = logging.FileHandler(logfile)
    # create console handler
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s:%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main():
    args = params.parse()

    logger = init_logger(args)
    logger.info('Running on {}'.format(socket.gethostname()))

    if not args.test_only:
        logger.info(args)
        logger.info('Loading dataset from {}...'.format(args.dset))
        dset = datasets.load(args)

        with open(os.path.join(args.logdir, 'flags.pkl'), 'wb') as fout:
            pickle.dump(args, fout)

        logger.info('Loading model {}. Logdir {}'.format(args.model, args.logdir))
        if os.path.isdir(args.logdir):
            if args.from_scratch:
                logger.info('Training from scratch; removing existing data in {}'.format(args.logdir))
                for f in (glob.glob(args.logdir + '/*ckpt*') +
                          glob.glob(args.logdir + '/events.out*') +
                          glob.glob(args.logdir + '/checkpoint') +
                          glob.glob(args.logdir + '/graph.pbtxt')):
                    os.remove(f)
            else:
                logger.info('Continuing from checkpoint in {}'.format(args.logdir))

        dsetarg = {'dset': dset}
        net = models.get_model(args.model, args, **dsetarg)

        logger.info('Start training...')
        trainfun = models.Dataset_train
        train, valid, train_time, _ = trainfun(net, dset, args)

    else:
        dset_dir = args.dset_dir
        with open(os.path.join(args.logdir, 'flags.pkl'), 'rb') as fin:
            args = pickle.load(fin)
        args = params.parse(args.__dict__)
        args.dset_dir = dset_dir

        train, valid, train_time = 0, 0, 0
        logger.info(args)

    logger.info('Start testing...')
    tf.reset_default_graph()
    # need to reload tf dataset because graph was reset
    dset = datasets.load(args)

    dsetarg = {'dset': dset}
    net = models.get_model(args.model, args, **dsetarg)

    test_final, conf_final = models.Dataset_test(net, dset, args, 'final.ckpt')

    logger.info('|{}|{}|{}|{}|{}|'
                .format('model', 'train', 'val',
                        'test', 'train time'))
    logger.info('|{}|{:.4f}|{:.4f}|{:.4f}|{:.2f}|'
                .format(args.run_id, train, valid, test_final,
                        train_time))


if __name__ == '__main__':
    main()
