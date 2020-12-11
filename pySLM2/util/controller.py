try:
    import ALP4
except:
    ALP4 = None

__all__ = ["DMDControllerBase", "ALPController", "LuxbeamController"]


class DMDControllerBase(object):
    def __init__(self, invert=False):
        self._initialized = False
        self.invert = invert
        super(DMDControllerBase, self).__init__()

    def initialize(self):
        self._initialized = True

    def close(self):
        pass

    def load_single(self, image):
        raise NotImplementedError

    def load_sequence(self):
        raise NotImplementedError

    def fire_software_trigger(self):
        raise NotImplementedError

    @property
    def initialized(self):
        return self._initialized


def _check_initialization(function):
    def wrapper(*args, **kwargs):
        controller = args[0]
        assert isinstance(controller, DMDControllerBase)
        if not controller.initialized:
            raise Exception("Controller not initialized.")
        return function(*args, **kwargs)

    return wrapper


class ALPController(DMDControllerBase):
    def __init__(self, invert=False, version='4.3'):
        if ALP4 is None:
            raise ModuleNotFoundError("ALP4lib module is unavailable.")

        self.alp = ALP4.ALP4(version=version, invert=invert)

        super(ALPController, self).__init__(invert=invert)

    def initialize(self):
        self.alp.Initialize()
        super(ALPController, self).initialize()

    def close(self):
        self.alp.Free()

    @_check_initialization
    def load_single(self, image):
        # Allocate the onboard memory for the image sequence
        self.alp.SeqAlloc(nbImg=1, bitDepth=1)

        # Remove the reset time between frames.
        self.alp.SeqControl(ALP4.ALP_BIN_MODE, ALP4.ALP_BIN_UNINTERRUPTED)

        # Send the image sequence as a 1D list/array/numpy array
        self.alp.SeqPut(imgData=image)

        self.alp.SetTiming()

        self.alp.Run()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LuxbeamController(DMDControllerBase):
    def __init__(self):
        super(LuxbeamController, self).__init__()
