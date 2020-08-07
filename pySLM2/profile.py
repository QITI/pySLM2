import tensorflow as tf
import numpy as np
from scipy.special import hermite, factorial
from ._backend import BACKEND
import math


class FunctionProfile(object):
    @tf.function
    def _func(self, x, y):
        raise NotImplementedError

    def __call__(self, x, y):
        x = tf.constant(x, dtype=BACKEND.dtype)
        y = tf.constant(y, dtype=BACKEND.dtype)
        return np.array(self._func(x, y))

    def __add__(self, other):
        func_profile = FunctionProfile()
        if isinstance(other, FunctionProfile):
            func_profile._func = tf.function(func=lambda x, y: self._func(x, y) + other._func(x, y))
        else:  # TODO: check availabe types
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
        theta = tf.constant(theta, dtype=BACKEND.dtype)
        func_profile = FunctionProfile()
        func_profile._func = tf.function(func=lambda x, y: self._func(
            x=tf.cos(theta) * x - tf.sin(theta) * y,
            y=tf.sin(theta) * x + tf.cos(theta) * y
        ))
        return func_profile

    def shift(self, dx, dy):
        dx = tf.constant(dx, dtype=BACKEND.dtype)
        dy = tf.constant(dy, dtype=BACKEND.dtype)

        func_profile = FunctionProfile()
        func_profile._func = tf.function(func=lambda x, y: self._func(
            x=x - dx,
            y=y - dy
        ))
        return func_profile


class HermiteGaussian(FunctionProfile):
    def __init__(self, x0, y0, a, w, n=0, m=0):
        self._x0 = tf.Variable(x0, dtype=BACKEND.dtype)
        self._y0 = tf.Variable(y0, dtype=BACKEND.dtype)
        self._a = tf.Variable(a, dtype=BACKEND.dtype)
        self._w = tf.Variable(w, dtype=BACKEND.dtype)
        self._n = n  # read only for now
        self._m = m  # read only for now

        self._hermite_n_coef = [tf.constant(c, dtype=BACKEND.dtype) for c in hermite(self._n).coef]
        self._hermite_m_coef = [tf.constant(c, dtype=BACKEND.dtype) for c in hermite(self._m).coef]

    @tf.function
    def _func(self, x, y):
        x_norm = (x - self._x0) / self._w
        y_norm = (y - self._y0) / self._w
        h_n = tf.math.polyval(coeffs=self._hermite_n_coef, x=math.sqrt(2) * x_norm)
        h_m = tf.math.polyval(coeffs=self._hermite_n_coef, x=math.sqrt(2) * y_norm)
        return self._a * h_n * h_m * tf.exp(-(x_norm ** 2 + y_norm ** 2))

    @property
    def x0(self):
        return self._x0.value()

    @x0.setter
    def x0(self, value):
        self._x0.assign(value)

    @property
    def y0(self):
        return self._y0.value()

    @y0.setter
    def y0(self, value):
        self._y0.assign(value)

    @property
    def a(self):
        return self._a.value()

    @a.setter
    def a(self, value):
        self._a.assign(value)

    @property
    def w(self):
        return self._w.value()

    @w.setter
    def w(self, value):
        self._w.assign(value)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m


class Zernike(FunctionProfile):
    def __init__(self, a, radius, n=0, m=0, normalize=True, extrapolate=False):
        if not n >= m:
            raise ValueError("Zernike index m must be >= index n")
        if (n - m) % 2 != 0:
            print("Radial polynomial is zero for these inputs: m={}, n={} " +
                  "(are you sure you wanted this Zernike?)".format(m, n))

        self._n = n
        self._m = m
        self._radius = tf.constant(radius, dtype=BACKEND.dtype)
        self._a = tf.Variable(a, dtype=BACKEND.dtype)

        self._coef = [0.0 for _ in range(n + 1)]

        self._normalization = (math.sqrt(n+1) if m==0 else math.sqrt(2*n+2)) if normalize else None
        self._extraoikate = extrapolate

        m_abs = np.abs(m)
        for k in range(int((n - m_abs) / 2) + 1):
            self._coef[n - 2 * k] = (-1) ** k * factorial(n - k) / (
                    factorial(k) * factorial((n + m_abs) / 2. - k) * factorial((n - m_abs) / 2. - k))

        self._coef = [tf.constant(c, dtype=BACKEND.dtype) for c in self._coef]

    @tf.function
    def _func(self, x, y):
        r = tf.sqrt(x ** 2 + y ** 2)
        rho = r / self._radius
        phi = tf.math.atan2(y, x)
        R = tf.math.polyval(coeffs=self._coef, x=rho)

        if self._m == 0:
            Z_unnomalized = self._a * R
        elif self._m > 0:
            Z_unnomalized = self._a * R * tf.cos(self._m * phi)
        else:
            Z_unnomalized = self._a * R * tf.sin(self._m * phi)

        Z = self._normalization * Z_unnomalized if self.is_normalized() else Z_unnomalized

        if not self._extraoikate:
            Z = tf.where(rho > 1.0, tf.zeros_like(Z, dtype=BACKEND.dtype), Z)

        return Z


    @property
    def a(self):
        return self._a.value()

    @a.setter
    def a(self, value):
        self._a.assign(value)

    @property
    def radius(self):
        return float(self._radius)

    def is_normalized(self):
        return False if self._normalization is None else True

    @property
    def normalization(self):
        return self._normalization

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m