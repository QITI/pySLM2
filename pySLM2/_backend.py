import tensorflow as tf


class _BackEnd(object):
    FFT_BACKEND_TENSORFLOW = 0
    FFT_BACKEND_NUMPY = 1

    TENSOR_32BITS = tf.float32
    TENSOR_64BITS = tf.float64

    def __init__(self):
        self._dtype = tf.float32
        self._fft_backend = tf.Variable(self.FFT_BACKEND_NUMPY)

        if tf.config.list_logical_devices("GPU"):
            self._fft_backend.assign(self.FFT_BACKEND_TENSORFLOW)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if not (value == tf.float32 or value == tf.float64):
            raise ValueError()
        else:
            self._dtype = value

    @property
    def dtype_complex(self):
        _dtype_complex = tf.complex64 if self._dtype == tf.float32 else tf.complex128
        return _dtype_complex

    @property
    def fft_backend(self):
        return self._fft_backend.value()

    @fft_backend.setter
    def fft_backend(self, value):
        self._fft_backend.assign(value)


BACKEND = _BackEnd()
