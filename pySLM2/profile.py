import tensorflow as tf
import numpy as np
from scipy.special import hermite

DTYPE = tf.float32


class FunctionProfile(object):
    @tf.function
    def _func(self, x, y):
        raise NotImplementedError

    def __call__(self, x, y):
        x = tf.constant(x, dtype=DTYPE)
        y = tf.constant(y, dtype=DTYPE)
        return np.array(self._func(x, y))

    def __add__(self, other):
        func_profile = FunctionProfile()
        if isinstance(other, FunctionProfile):
            func_profile._func = tf.function(func=lambda x, y: self._func(x, y) + other._func(x, y))
        else: # TODO: check availabe types
            func_profile._func = tf.function(func=lambda x, y: self._func(x, y) + other)
        return func_profile

    def __mul__(self, other):
        func_profile = FunctionProfile()
        if isinstance(other, FunctionProfile):
            func_profile._func = tf.function(func=lambda x, y: self._func(x, y) * other._func(x, y))
        else:
            # TODO: check availabe types
            func_profile._func = tf.function(func=lambda x, y: self._func(x, y) + other)
        return func_profile

    def rotate(self, theta):
        # TODO: verify the theta direction (CW or CCW for positive theta?)
        theta = tf.constant(theta, dtype=DTYPE)
        func_profile = FunctionProfile()
        func_profile._func = tf.function(func=lambda x, y: self._func(
            x=tf.cos(theta) * x - tf.sin(theta) * y,
            y=tf.sin(theta) * x + tf.cos(theta) * y
        ))
        return func_profile

    def shift(self, dx, dy):
        dx = tf.constant(dx, dtype=DTYPE)
        dy = tf.constant(dy, dtype=DTYPE)

        func_profile = FunctionProfile()
        func_profile._func = tf.function(func=lambda x, y: self._func(
            x=x-dx,
            y=y-dy
        ))
        return func_profile


class HermiteGaussian(FunctionProfile):
    def __init__(self, x0, y0, a, w, n=0, m=0):
        self._x0 = tf.Variable(x0, dtype=DTYPE)
        self._y0 = tf.Variable(y0, dtype=DTYPE)
        self._a = tf.Variable(a, dtype=DTYPE)
        self._w = tf.Variable(w, dtype=DTYPE)
        self._n = n # read only for now
        self._m = m # read only for now

        self._hermite_n_coef = [tf.constant(c, dtype=DTYPE) for c in hermite(self._n).coef]
        self._hermite_m_coef = [tf.constant(c, dtype=DTYPE) for c in hermite(self._m).coef]

    @tf.function
    def _func(self, x, y):
        x_norm = (x - self._x0) / self._w
        y_norm = (y - self._y0) / self._w
        h_n = tf.math.polyval(coeffs=self._hermite_n_coef, x=tf.sqrt(2.0) * x_norm)
        h_m = tf.math.polyval(coeffs=self._hermite_n_coef, x=tf.sqrt(2.0) * y_norm)
        return self._a * h_n * h_m * tf.exp(-(x_norm**2 + y_norm**2))

    @property
    def x0(self):
        return float(self._x0)

    @x0.setter
    def x0(self, value):
        self._x0.assign(value)

    @property
    def y0(self):
        return float(self._y0)

    @y0.setter
    def y0(self, value):
        self._y0.assign(value)

    @property
    def a(self):
        return float(self._a)

    @a.setter
    def a(self, value):
        self._a.assign(value)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m





