import sys
import warnings

import tensorflow as tf
import numpy as np

try:
    import pyshtools
except:
    pass

from spherical_cnn import spherical
from spherical_cnn import util
from spherical_cnn.util import tf_config


def test_sph_harm_lm():
    n = 10
    phi, theta = np.meshgrid(*util.sph_sample(n))

    # let's check the 2nd degree harmonics
    l, m = 2, -2
    Y = spherical.sph_harm_lm(l, m, n)
    Y_ref = 1/4*np.sqrt(15/2/np.pi)*(np.sin(phi))**2 * np.exp(-2j * theta)
    assert np.allclose(Y, Y_ref)

    l, m = 2, 1
    Y = spherical.sph_harm_lm(l, m, n)
    Y_ref = -1/2*np.sqrt(15/2/np.pi)*np.sin(phi)*np.cos(phi) * np.exp(1j * theta)
    assert np.allclose(Y, Y_ref)


def test_sph_harm_isft():
    """ Tests the SFT by doing ISFT -> SFT -> ISFT. """
    for n in [8, 16]:
        # sft for real signals: c_{-m} = (-1)^m Re(c_m) + (-1)^{m+1} Im(c_m)
        coeffs = [np.zeros(2*l+1, dtype=np.complex) for l in range(n // 2)]

        coeffs[0][0], coeffs[1][1], coeffs[2][2] = [np.random.rand() for _ in range(3)]

        c = np.random.rand() + 1j*np.random.rand()
        coeffs[1][0] = c
        coeffs[1][2] = -np.real(c) + 1j*np.imag(c)

        c = np.random.rand() + 1j*np.random.rand()
        coeffs[2][0] = c
        coeffs[2][4] = np.real(c) - 1j*np.imag(c)

        c = np.random.rand() + 1j*np.random.rand()
        coeffs[2][1] = c
        coeffs[2][3] = -np.real(c) + 1j*np.imag(c)

        f = spherical.sph_harm_inverse(coeffs)

        coeffs_ = spherical.sph_harm_transform(f)
        f_ = spherical.sph_harm_inverse(coeffs_)

        for c1, c2 in zip(coeffs, coeffs_):
            assert np.allclose(c1, c2)
        assert np.allclose(f, f_)


def test_sph_harm_sft():
    """ Tests the SFT by doing SFT -> ISFT -> SFT -> ISFT """
    f = np.random.rand(16, 16)
    coeffs = spherical.sph_harm_transform(f)
    # we are constraining the bandwidth to n/2 here, so f1 != f
    f1 = spherical.sph_harm_inverse(coeffs)
    coeffs1 = spherical.sph_harm_transform(f1)
    f2 = spherical.sph_harm_inverse(coeffs1)
    # now both f1 and f2 have constrained bandwidths, so they must be equal
    assert np.allclose(f1, f2)


def test_sph_harm_batch_and_inverse():
    """ Test batch form of spherical harmonics transform and inverse """
    f = np.random.rand(10, 32, 32, 3)
    c1 = spherical.sph_harm_transform_batch(f)
    r1 = spherical.sph_harm_inverse_batch(c1)
    c2 = spherical.sph_harm_transform_batch(r1)
    r2 = spherical.sph_harm_inverse_batch(c2)

    assert np.allclose(r1, r2)


def test_sph_harm_batch_real_complex():
    """ Test batch form of spherical harmonics transform and inverse """
    f = np.random.rand(10, 32, 32, 3)
    h_complex = spherical.sph_harm_to_shtools(spherical.sph_harm_all(32))
    h_real = spherical.sph_harm_to_shtools(spherical.sph_harm_all(32, real=True))

    c1 = spherical.sph_harm_transform_batch(f, harmonics=h_complex)
    c2 = spherical.sph_harm_transform_batch(f.astype('complex'), harmonics=h_complex)
    c3 = spherical.sph_harm_transform_batch(f, harmonics=h_real)
    c4 = spherical.sph_harm_transform_batch(f.astype('complex'), harmonics=h_real)

    assert np.allclose(c1, c2)
    assert np.allclose(c2[:, [0]], c3)
    assert np.allclose(c3, c4)


def test_sph_harm_batch():
    """ Test batch form of spherical harmonics transform and inverse """
    f = np.random.rand(1, 32, 32, 1)
    # computing harmonics on the fly
    c1 = spherical.sph_harm_transform_batch(f)[0, ..., 0]
    c2 = spherical.sph_harm_to_shtools(spherical.sph_harm_transform(f[0, ..., 0]))
    assert np.allclose(c1, c2)

    # caching harmonics
    for real in [False, True]:
        h = spherical.sph_harm_all(32, real=real)
        c1 = spherical.sph_harm_transform_batch(f,
                                                harmonics=spherical.sph_harm_to_shtools(h))[0, ..., 0]
        c2 = spherical.sph_harm_to_shtools(spherical.sph_harm_transform(f[0, ..., 0],
                                                                        harmonics=h))

        assert np.allclose(c1, c2)


def test_sph_harm_batch_harmonics_input():
    """ Test batch form of spherical harmonics transform and inverse """
    for real in [False, True]:
        harmonics = spherical.sph_harm_to_shtools(spherical.sph_harm_all(32, real=real))

        f = np.random.rand(10, 32, 32, 3)
        c1 = spherical.sph_harm_transform_batch(f, harmonics=harmonics)
        r1 = spherical.sph_harm_inverse_batch(c1, harmonics=harmonics)
        c2 = spherical.sph_harm_transform_batch(r1, harmonics=harmonics)
        r2 = spherical.sph_harm_inverse_batch(c2, harmonics=harmonics)

        assert np.allclose(r1, r2)


def test_sph_harm_tf():
    """ Test spherical harmonics expansion/inversion with tensorflow Tensors. """
    n = 32
    f = np.random.rand(n, n)
    ref = spherical.sph_harm_inverse(spherical.sph_harm_transform(f))

    inp = tf.placeholder('complex128', shape=[n, n])
    coeffs = spherical.sph_harm_transform(inp)
    recons = spherical.sph_harm_inverse(coeffs)

    with tf.Session(config=tf_config()).as_default() as sess:
        c, r = sess.run([coeffs, recons],
                        feed_dict={inp: f})

    assert np.allclose(r, ref)
    for x1, x2 in zip(c, spherical.sph_harm_transform(f)):
        assert np.allclose(x1, x2)


def test_sph_harm_tf_harmonics_input():
    """ Test spherical harmonics inputs as tensorflow Variables. """
    n = 32
    f = np.random.rand(n, n)

    for real in [False, True]:
        dtype = tf.complex64
        harmonics = [[tf.Variable(hh.astype('complex64')) for hh in h]
                     for h in spherical.sph_harm_all(n, real=real)]

        inp = tf.placeholder(dtype, shape=[n, n])
        c1 = spherical.sph_harm_transform(inp, harmonics=harmonics)
        r1 = spherical.sph_harm_inverse(c1, harmonics=harmonics)
        c2 = spherical.sph_harm_transform(inp, harmonics=harmonics)
        r2 = spherical.sph_harm_inverse(c2, harmonics=harmonics)

        with tf.Session(config=tf_config()).as_default() as sess:
            sess.run(tf.global_variables_initializer())
            c1v, c2v, r1v, r2v = sess.run([c1, c2, r1, r2],
                                          feed_dict={inp: f})

        for x1, x2 in zip(c1v, c2v):
            assert np.allclose(x1, x2)
        for x1, x2 in zip(r1v, r2v):
            assert np.allclose(x1, x2)


def test_sph_harm_shtools():
    """ Compare our sph harmonics expansion with pyshtools. """
    if 'pyshtools' not in sys.modules:
        warnings.warn('pyshtools not available; skipping test_sph_harm_shtools')
        return

    n = 32
    f = np.random.rand(n, n)
    # lowpass
    f = spherical.sph_harm_inverse(spherical.sph_harm_transform(f))
    
    c_mine = spherical.sph_harm_transform(f)
    c_pysh = pyshtools.SHGrid.from_array(f.T).expand(csphase=-1, normalization='ortho')

    c1 = spherical.sph_harm_to_shtools(c_mine)
    c2 = c_pysh.coeffs

    # there seems to be a bug on the coefficient of highes degree l, order -l in pyshtools
    # we don't test that value
    c1[1][(n // 2) - 1][-1] = c2[1][(n // 2) -1][-1] = 0
    assert np.allclose(c1, c2)


def test_sph_conv():
    """ Test spherical convolution and rotation commutativity.

    sph_conv and sphrot_shtools are exercised here.
    """
    if 'pyshtools' not in sys.modules:
        warnings.warn('pyshtools not available; skipping test_sph_conv')
        return

    n = 32
    f = np.random.rand(n, n)
    # lowpass
    f = spherical.sph_harm_inverse(spherical.sph_harm_transform(f))

    g = np.zeros_like(f)
    g[:, :5] = np.random.rand(5)
    g /= g.sum()

    ang = np.random.rand(3)*2*np.pi

    # check if (pi f * g) == pi(f * g)
    rot_conv = util.sphrot_shtools(spherical.sph_conv(f, g), ang)
    conv_rot = spherical.sph_conv(util.sphrot_shtools(f, ang), g)

    assert not np.allclose(rot_conv, f)
    assert not np.allclose(rot_conv, g)
    assert np.allclose(rot_conv, conv_rot)


def test_sph_conv_tf():
    """ Test spherical convolution with tensorflow Tensors. """
    n = 32
    kernel_size = 5
    f = np.random.rand(32, 32)

    # double precision harmonics are returned by default, so that's how we test
    inp = tf.placeholder('complex128', shape=[n, n])
    weights = tf.Variable(tf.cast(tf.truncated_normal([kernel_size]), 'complex128'))
    ker = tf.concat([tf.tile(weights[np.newaxis, :], [32, 1]),
                     tf.zeros((n,27), dtype='complex128')], axis=1)
    conv = spherical.sph_conv(inp, ker)

    with tf.Session(config=tf_config()).as_default() as sess:
        sess.run(tf.global_variables_initializer())
        conv_v, ker_v = sess.run([conv, ker],
                               feed_dict={inp: f})

    assert np.allclose(conv_v, spherical.sph_conv(f, ker_v))


def test_sph_conv_batch():
    """ Test batch spherical convolution """
    f = np.random.rand(2, 32, 32, 3).astype('float32') + 0j
    g = np.random.rand(3, 32, 32, 6).astype('float32') + 0j
    res = spherical.sph_conv_batch(f, g)

    # compare with regular convolution
    for i in range(f.shape[0]):
        for j in range(g.shape[-1]):
            conv = np.zeros_like(f[0, ..., 0]).astype('complex')
            for k in range(f.shape[-1]):
                conv += spherical.sph_conv(f[i, ..., k], g[k, ..., j])
            assert np.allclose(conv, res[i, ..., j])

    # test using tf Tensors
    tfconv = spherical.sph_conv_batch(tf.constant(f), tf.constant(g))
    with tf.Session(config=tf_config()).as_default() as sess:
        restf = sess.run(tfconv)

    assert np.allclose(res, restf)


def test_sph_conv_real_complex():
    """ Compare spherical convolution assuming real inputs and not. """
    f = np.random.rand(2, 32, 32, 3).astype('float32')
    g = np.random.rand(3, 32, 32, 6).astype('float32')

    h_complex = spherical.sph_harm_to_shtools(spherical.sph_harm_all(32))
    h_real = spherical.sph_harm_to_shtools(spherical.sph_harm_all(32, real=True))

    res_complex = spherical.sph_conv_batch(f, g, harmonics_or_legendre=h_complex).real
    res_real = spherical.sph_conv_batch(f, g, harmonics_or_legendre=h_real)
    assert np.allclose(res_complex, res_real)

    # test using tf Tensors
    tfconv = spherical.sph_conv_batch(tf.constant(f), tf.constant(g),
                                      harmonics_or_legendre=tf.constant(h_real))
    with tf.Session(config=tf_config()).as_default() as sess:
        restf = sess.run(tfconv)

    assert np.allclose(res_real, restf)


def test_sph_conv_batch_spectral_filters():
    f = np.random.rand(2, 32, 32, 3).astype('float32') + 0j
    g = np.random.rand(3, 32, 32, 6).astype('float32') + 0j
    ref = spherical.sph_conv_batch(f, g)

    cg = spherical.sph_harm_transform_batch(g, m0_only=True)
    test = spherical.sph_conv_batch(f, cg)

    assert np.allclose(ref, test)


def test_sph_conv_batch_spectral_input():
    f = np.random.rand(2, 32, 32, 3).astype('float32') + 0j
    g = np.random.rand(3, 32, 32, 6).astype('float32') + 0j
    ref = spherical.sph_conv_batch(f, g)

    cf = spherical.sph_harm_transform_batch(f, m0_only=False)
    test = spherical.sph_conv_batch(cf, g)
    assert np.allclose(ref, test)

    cg = spherical.sph_harm_transform_batch(g, m0_only=True)
    test = spherical.sph_conv_batch(cf, cg)
    assert np.allclose(ref, test)
