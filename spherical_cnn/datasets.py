import os
import glob

import tensorflow as tf


def from_cached_tfrecords(args):
    """ Use tf.Dataset, but feeding it using a placeholder w/ the whole dataset. """
    # this may seem a bit weird
    # we take tfrecords but load them into placeholders during training
    # we found that it loaded faster this way when this was first implemented
    # letting tf.Dataset loading all simulataneously is conceptually better

    res, nch = args.input_res, args.nchannels

    x = tf.placeholder(args.dtype, (None, res, res, nch))
    y = tf.placeholder('int64', (None))

    dataset = tf.contrib.data.Dataset.from_tensor_slices((x, y))

    # inputs are complex numbers
    # magnitude is ray length
    # phase is angle between ray and normal
    # we found that it is best to treat them independently, though
    dataset = dataset.map(lambda x, y: (tf.concat([tf.abs(x),
                                                   tf.imag(x/(tf.cast(tf.abs(x), 'complex64') +1e-8))],
                                                  axis=-1), y))

    # we use same batch sizes for train/val/test
    dataset = dataset.batch(args.train_bsize)
    iterator = dataset.make_initializable_iterator()

    fnames = {}
    for t in ['train', 'test', 'val']:
        fnames[t] = glob.glob(args.dset_dir + '/{}*.tfrecord'.format(t))

    out = {'x': x, 'y': y, 'fnames': fnames}
    print('loading dataset; number of tfrecords: {}'
          .format({k: len(v) for k, v in out['fnames'].items()}))

    return iterator, out


def load(args):
    dset_fun = globals().get(os.path.splitext(args.dset)[0])
    dset = dset_fun(args)

    return dset
