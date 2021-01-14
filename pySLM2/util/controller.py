try:
    import ALP4
except:
    ALP4 = None

try:
    import Luxbeam
except:
    Luxbeam = None

__all__ = ["DMDControllerBase", "ALPController", "LuxbeamController"]


class DMDControllerBase(object):
    """Base class for the DMD controller."""
    def __init__(self, invert=False):
        self._initialized = False
        self.invert = invert
        super(DMDControllerBase, self).__init__()

    def initialize(self):
        self._initialized = True

    def close(self):
        pass

    def load_single(self, dmd_state):
        """

        Parameters
        ----------
        dmd_state: numpy.ndarray
        """
        raise NotImplementedError

    def load_sequence(self, images):
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
    """This class implements the control function for interacting the controller from Vialux.

    Parameters
    ----------
    invert: bool
        If true, invert the on/off mirrors in the hologram.
    version: str
        Version of the ALP library fom Vialux.

    See Also
    --------
    ALP4.ALP4

    """
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
    def load_single(self, dmd_state):
        """load and display a single binary image on the DMD.

        Parameters
        ----------
        dmd_state: numpy.ndarray
            The dtype must be bool and have the same dimension as the DMD.
        """
        # Allocate the onboard memory for the image sequence
        self.alp.SeqAlloc(nbImg=1, bitDepth=1)

        # Remove the reset time between frames.
        self.alp.SeqControl(ALP4.ALP_BIN_MODE, ALP4.ALP_BIN_UNINTERRUPTED)

        # Send the image sequence as a 1D list/array/numpy array
        self.alp.SeqPut(imgData=dmd_state)

        self.alp.SetTiming()

        self.alp.Run()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LuxbeamController(DMDControllerBase):
    """This class implements the control function for interacting the controller from Visitech.

    Parameters
    ----------
    ip: str
        IP address of the controller.
    invert: bool
        If true, invert the on/off mirrors in the hologram.
    timeout: None or float
        Timeout of the network socket.

    See Also
    --------
    Luxbeam.Luxbeam

    """
    def __init__(self, ip, invert=False, timeout=None):
        if Luxbeam is None:
            raise ModuleNotFoundError("Luxbeam module is unavailable.")

        self.luxbeam = Luxbeam.Luxbeam(ip, inverse=invert, timeout=timeout)

        super(LuxbeamController, self).__init__(invert=invert)

    def initialize(self):
        super(LuxbeamController, self).initialize()

    def close(self):
        pass

    @_check_initialization
    def load_single(self, dmd_state):
        """load and display a single binary image on the DMD.

        Parameters
        ----------
        dmd_state: numpy.ndarray
            The dtype must be bool and have the same dimension as the DMD.
        """
        # TODO Load the sequencer
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RUN, Luxbeam.ENABLE)
        self.luxbeam.load_image(0, dmd_state)
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RUN, Luxbeam.DISABLE)

    @_check_initialization
    def fire_software_trigger(self):
        self.luxbeam.set_software_sync(1)
        self.luxbeam.set_software_sync(0)









