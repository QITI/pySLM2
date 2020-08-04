import tensorflow as tf
from .profile import FunctionProfile
from ._backend import BACKEND

__all__ = ["RectWindow"]


class RectWindow(FunctionProfile):
    def __init__(self, x0, y0, wx, wy):
        self._x0 = tf.Variable(x0, dtype=BACKEND.dtype)
        self._y0 = tf.Variable(y0, dtype=BACKEND.dtype)
        self._wx = tf.Variable(wx, dtype=BACKEND.dtype)
        self._wy = tf.Variable(wx, dtype=BACKEND.dtype)

    @tf.function
    def _func(self, x, y):
        return tf.cast((tf.abs(x - self._x0) < (self._wx / 2)), BACKEND.dtype) * \
                tf.cast((tf.abs(y - self._y0) < (self._wy / 2)), BACKEND.dtype)

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
    def wx(self):
        return self._wx.value()

    @wx.setter
    def wx(self, value):
        self._wx.assign(value)

    @property
    def wy(self):
        return self._wy.value()

    @wy.setter
    def wy(self, value):
        self._wy.assign(value)