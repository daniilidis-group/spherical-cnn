"""
Some functions are designed to work with either numpy arrays or tensorflow tensors.
These wrappers ensure compatibility.
"""
# this is a questionable design decision...

import tensorflow as tf
import numpy as np

from .util import safe_cast


def istf(x):
    """ Check if argument is a tensorflow structure (or list of). """
    if isinstance(x, list):
        x = x[0]
    return (isinstance(x, tf.Tensor) or
            isinstance(x, tf.Variable))


def shape(x):
    """ Return shape of either tensorflow Tensor or numpy array. """
    return (tuple(x.get_shape().as_list()) if istf(x)
            else x.shape)


def sum(*args, **kwargs):
    """ Return np.sum or tf.reduce_sum according to input. """
    return fun(['reduce_sum', 'sum'], *args, **kwargs)


def fun(fun_name, *args, **kwargs):
    """ Return np or tf version of function according to input.

    Args:
        fun_name (list or str): if str, return tf.fun_name or np.fun_name
                                if list, return tf.fun_name[0] or np.fun_name[1]
    """
    if isinstance(fun_name, list):
        f1, f2 = fun_name
    else:
        f1, f2 = fun_name, fun_name
    return (getattr(tf, f1)(*args, **kwargs) if any([istf(a) for a in args])
            else getattr(np, f2)(*args, **kwargs))


def dot(x, y, *args, **kwargs):
    """ Return np.tensordot or tf.tensordot according to input. """
    if istf(x) and not istf(y):
        y = tf.constant(y)
    if istf(y) and not istf(x):
        x = tf.constant(x)
    if istf(x) and istf(y):
        x, y = safe_cast(x, y)
        return tf.tensordot(x, y, *args, **kwargs)
    else:
        return np.tensordot(x, y, *args, **kwargs)


def concat(*args, axis=0):
    """ Return np.concatenate or tf.concat according to input. """
    # this is a bit fragile, why should y take type of x and not the other way around?
    return (tf.concat([*args], axis=axis) if istf(args[0])
            else np.concatenate(args, axis=axis))


def fft(x, *args, **kwargs):
    """ Return np.fft.fft or tf.fft according to input. """
    return (tf.fft(x, *args, **kwargs) if istf(x)
            else np.fft.fft(x, *args, **kwargs))


def conj(*args, **kwargs):
    return fun('conj', *args, **kwargs)


def transpose(*args, **kwargs):
    return fun('transpose', *args, **kwargs)


def reshape(*args, **kwargs):
    return fun('reshape', *args, **kwargs)


def real_or_imag(x, part):
    """ Return real or imaginary part of either tensorflow Tensor or numpy array. """
    if istf(x):
        fun = getattr(tf, part)
        if x.dtype.is_complex:
            return fun(x)
        else:
            nbits = x.dtype.size*8
            return fun(tf.cast(x, 'complex{}'.format(nbits*2)))
    else:
        fun = getattr(np, part)
        return fun(x)


def real(x):
    return real_or_imag(x, 'real')


def imag(x):
    return real_or_imag(x, 'imag')
