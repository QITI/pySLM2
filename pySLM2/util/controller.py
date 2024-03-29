try:
    import ALP4
except:
    ALP4 = None

try:
    import Luxbeam
except:
    Luxbeam = None

import inspect
from .sample import number_image
import numpy as np

from typing import List

__all__ = ["DMDControllerBase", "ALPController", "LuxbeamController"]


def _check_initialization(function):
    def wrapper(*args, **kwargs):
        controller = args[0]
        assert isinstance(controller, DMDControllerBase)
        if not controller.is_initialized():
            raise Exception("Controller not initialized.")
        return function(*args, **kwargs)
    wrapper.__signature__ = inspect.signature(function)
    return wrapper


class DMDControllerBase(object):
    """Base class for the DMD controller."""
    def __init__(self, invert=False):
        self._initialized = False
        self.invert = invert
        super(DMDControllerBase, self).__init__()

    def initialize(self):
        self._initialized = True

    def is_initialized(self):
        return self._initialized

    def close(self):
        pass

    @_check_initialization
    def load_single(self, dmd_state):
        """

        Parameters
        ----------
        dmd_state: numpy.ndarray
        """
        raise NotImplementedError

    @_check_initialization
    def load_multiple(self, dmd_states: List[np.ndarray]):
        raise NotImplementedError

    @_check_initialization
    def load_sequence(self, images):
        raise NotImplementedError

    @property
    @_check_initialization
    def Nx(self) -> int:
        raise NotImplementedError

    @property
    @_check_initialization
    def Ny(self) -> int:
        raise NotImplementedError

    def fire_software_trigger(self):
        raise NotImplementedError

    @_check_initialization
    def number_image(self, i):
        return number_image(i, self.Nx, self.Ny)


class ALPController(DMDControllerBase):
    MAX_UINT8 = 2**8 - 1 
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

        self.alp = ALP4.ALP4(version=version)
        super(ALPController, self).__init__(invert=invert)

    def initialize(self):
        self.alp.Initialize()
        if self.invert:
            self.alp.ProjControl(ALP4.ALP_PROJ_INVERSION, ALP4.ALP_ENABLE)
        super(ALPController, self).initialize()

    def close(self):
        self.alp.Halt()
        if self.alp._lastDDRseq is not None:
            self.alp.FreeSeq()
        self.alp.Free()

    @property
    @_check_initialization
    def Nx(self) -> int:
        """Number of pixels in x direction (width)."""
        return self.alp.DevInquire(ALP4.ALP_DEV_DISPLAY_WIDTH)

    @property
    @_check_initialization
    def Ny(self) -> int:
        """Number of pixels in y direction (height)."""
        return self.alp.DevInquire(ALP4.ALP_DEV_DISPLAY_HEIGHT)

    @_check_initialization
    def load_single(self, dmd_state):
        """load and display a single binary image on the DMD until program is closed, after which the DMD gets reset.

        Parameters
        ----------
        dmd_state: numpy.ndarray
            The dtype must be bool and have the same dimension as the DMD.
        """
        # Prepare the dmd in available state
        self.alp.Halt()
        
        # Allocate the onboard memory for the image sequence
        self.alp.SeqAlloc(nbImg=1, bitDepth=1)

        # Remove the reset time between frames.
        self.alp.SeqControl(ALP4.ALP_BIN_MODE, ALP4.ALP_BIN_UNINTERRUPTED)

        # Send the image sequence as a 1D list/array/numpy array
        self.alp.SeqPut(imgData=dmd_state * self.MAX_UINT8)

        self.alp.SetTiming()

        self.alp.Run()


    @_check_initialization
    def load_multiple(self, dmd_states: List[np.ndarray]):
        """load and display a list of binary images on the DMD on the rising edge of the trigger signal.
        All pictures are displayed once for the MIN_PICTURE_TIME.
        

        Parameters
        ----------
        dmd_states: List[numpy.ndarray]
            The dtype must be bool and have the same dimension as the DMD.
        """
        # Prepare the dmd in available state
        self.alp.Halt()

        # Set to slave mode
        self.alp.ProjControl(ALP4.ALP_PROJ_MODE, ALP4.ALP_SLAVE)

        # Set the trigger edge to be rising
        self.alp.DevControl(ALP4.ALP_TRIGGER_EDGE, ALP4.ALP_EDGE_RISING)

        # Allocate the onboard memory for the image sequence
        self.alp.SeqAlloc(nbImg=len(dmd_states), bitDepth=1)

        # Remove the reset time between frames.
        self.alp.SeqControl(ALP4.ALP_BIN_MODE, ALP4.ALP_BIN_UNINTERRUPTED)

        # Send the image sequence as a 1D list/array/numpy array
        self.alp.SeqPut(imgData=np.concatenate(dmd_states)*self.MAX_UINT8)
        
        # Set the timing
        self.alp.SetTiming(pictureTime=self.alp.SeqInquire(ALP4.ALP_MIN_PICTURE_TIME))

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
        seq = Luxbeam.LuxbeamSequencer()
        # ======= Sequencer ============
        reg0 = seq.assign_var_reg(regno=0) # reg0 is the total number of images
        for _ in seq.jump_loop_iter():
            seq.load_global(0, 400) # Load the first image to the DMD memory
            for _, inum in seq.range_loop_iter(0, reg0):
                seq.reset_global(40) # Set the DMD mirror with the image in the memory
                seq.load_global(inum + 1, 400) # Load the next image to the DMD memory
                seq.trig(Luxbeam.TRIG_MODE_POSITIVE_EDGE,
                         Luxbeam.TRIG_SOURCE_SOFTWARE +
                         Luxbeam.TRIG_SOURCE_ELECTRICAL +
                         Luxbeam.TRIG_SOURCE_OPTICAL,
                         0) # Wait for the trigger
        # ======= Sequencer ============

        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RUN, Luxbeam.DISABLE)
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RESET, Luxbeam.ENABLE)
        self.luxbeam.load_sequence(seq.dumps())
        self.luxbeam.set_sequencer_reg(reg_no=0, reg_val=1)
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RESET, Luxbeam.DISABLE)

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
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RUN, Luxbeam.DISABLE)

        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RESET, Luxbeam.ENABLE)
        self.luxbeam.set_sequencer_reg(reg_no=0, reg_val=1)
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RESET, Luxbeam.DISABLE)

        self.luxbeam.load_image(0, dmd_state)
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RUN, Luxbeam.ENABLE)

    @_check_initialization
    def load_multiple(self, dmd_states: List[np.ndarray]):
        """load and display a list of binary images on the DMD.

        Parameters
        ----------
        dmd_states: List[numpy.ndarray]
            The dtype must be bool and have the same dimension as the DMD.
        """
        
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RUN, Luxbeam.DISABLE)

        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RESET, Luxbeam.ENABLE)
        N = len(dmd_states)
        self.luxbeam.set_sequencer_reg(reg_no=0, reg_val=N)
        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RESET, Luxbeam.DISABLE)

        for i, dmd_state in enumerate(dmd_states):
            self.luxbeam.load_image(i, dmd_state)

        self.luxbeam.set_sequencer_state(Luxbeam.SEQ_CMD_RUN, Luxbeam.ENABLE)


    @property
    @_check_initialization
    def Nx(self) -> int:
        return self.luxbeam.cols

    @property
    @_check_initialization
    def Ny(self) -> int:
        return self.luxbeam.rows

    @_check_initialization
    def fire_software_trigger(self):
        self.luxbeam.set_software_sync(1)
        self.luxbeam.set_software_sync(0)
