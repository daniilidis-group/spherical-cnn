""" Custom layers. """


import tensorflow as tf

import numpy as np

from . import util
from . import spherical
from . import tfnp_compatibility as tfnp


def sphconv(inputs, filters, use_bias=True, n_filter_params=0,
            *args, **kwargs):
    shapei = tfnp.shape(inputs)
    spectral_input = True if len(shapei) == 5 else False
    nchan = shapei[-1]
    n = shapei[2]
    if spectral_input:
        n *= 2

    with tf.variable_scope(None, default_name='sphconv'):
        # we desire that variance to stay constant after every layer
        # factor n // 2 is because we sum n // 2 coefficients
        # factor nchan is because we sum nchan channels
        # factor 2pi is the 'gain' of integrating over SO(3)
        # the factor 2 takes into account non-zero mean
        # (see He et al 2015 "Delving deep into rectifiers")
        std = 2./(2 * np.pi * np.sqrt((n // 2) * (nchan)))

        if n_filter_params == 0:
            weights = tf.get_variable('W',
                                      trainable=True,
                                      initializer=tf.truncated_normal([nchan, n // 2, filters],
                                                                      stddev=std),
                                      regularizer=tf.contrib.layers.l2_regularizer(1.))
            ker = weights[:, np.newaxis, :, np.newaxis, :]
        else:
            nw_in = n_filter_params
            if nw_in > n // 2:
                nw_in = n // 2
            weights = tf.get_variable('W',
                                   trainable=True,
                                   initializer=tf.truncated_normal([nchan,
                                                                    nw_in,
                                                                    filters],
                                                                   stddev=std),
                                   regularizer=tf.contrib.layers.l2_regularizer(1.))
            xw_in = np.linspace(0, 1, nw_in)
            xw_out = np.linspace(0, 1, n // 2)
            id_out = np.searchsorted(xw_in, xw_out)
            subws = []
            for i, x in zip(id_out, xw_out):
                # linear interpolation
                # HACK! we assume the first indices match so i-1 when i==0 cancels out
                subws.append(weights[:, i-1, :] +
                             (weights[:, i, :] - weights[:, i-1, :]) *
                             (x-xw_in[i-1]) /
                             ((xw_in[i]-xw_in[i-1])))
            ker = tf.stack(subws, axis=1)[:, np.newaxis, :, np.newaxis, :]

        if use_bias:
            bias = tf.get_variable('b',
                                   trainable=True,
                                   initializer=tf.zeros([1, 1, 1, filters], dtype=tf.float32))
        else:
            bias = tf.zeros([1, 1, 1, filters], dtype=tf.float32)

        conv = spherical.sph_conv_batch(inputs, ker, *args, **kwargs)
        conv = conv + bias

        for k, v in {'W': weights, 'b': bias, 'activations': conv}.items():
            tf.summary.histogram(k, v)
        # avg
        tf.summary.scalar('norm_activation', tf.reduce_mean(tf.norm(conv, axis=(1, 2)) / n))

    return conv


def block(params, fun, is_training=None, *args, **kwargs):
    """ Block consisting of weight layer + batch norm + nonlinearity"""
    use_bias = not params.batch_norm
    curr = fun(*args, **kwargs, use_bias=use_bias)
    if params.batch_norm:
        curr = tf.layers.batch_normalization(curr,
                                             # doesn't make sense to learn scale when using ReLU
                                             scale=False,
                                             training=is_training,
                                             renorm=params.batch_renorm)
        for v in tf.get_variable_scope().trainable_variables():
            if 'batch_normalization' in v.name:
                tf.summary.histogram(v.name, v)

    return nonlin(params)(curr)


def nonlin(params):
    return getattr(tf.nn, params.nonlin, globals().get(params.nonlin))


def identity(inputs):
    return inputs


def prelu(inputs):
    """ From: https://stackoverflow.com/a/40264459 """
    alphas = tf.Variable(0.1 * tf.ones(inputs.get_shape()[-1]),
                         trainable=True,
                         dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


def area_weights(x, invert=False):
    """ Apply weight each cell according to its area; useful for averaging/avg pooling. """
    n = tfnp.shape(x)[1]
    phi, theta = util.sph_sample(n)
    phi += np.diff(phi)[0]/2
    # this is proportional to the cell area, not exactly the area
    # this is the same as using |cos\phi_1 - cos\phi_2|
    if invert:
        x /= np.sin(phi)[np.newaxis, np.newaxis, :, np.newaxis]
    else:
        x *= np.sin(phi)[np.newaxis, np.newaxis, :, np.newaxis]

    return x
