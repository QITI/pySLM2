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

        self._initialized = False

    @property
    def dtype(self):
        self._initialized = True
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        if self._initialized:
            # TODO: Not sure which error to raise
            # TODO: More precise description.
            raise AssertionError("Configure the backend at the beginning of your code!")

        if not (value == tf.float32 or value == tf.float64):
            raise ValueError()
        else:
            self._dtype = value

    @property
    def dtype_complex(self):
        self._initialized = True
        _dtype_complex = tf.complex64 if self._dtype == tf.float32 else tf.complex128
        return _dtype_complex

    @property
    def dtype_int(self):
        self._initialized = True
        _dtype_int = tf.int32 if self._dtype == tf.float32 else tf.int64
        return _dtype_int

    @property
    def fft_backend(self):
        return self._fft_backend.value()

    @fft_backend.setter
    def fft_backend(self, value):
        self._fft_backend.assign(value)


BACKEND = _BackEnd()
