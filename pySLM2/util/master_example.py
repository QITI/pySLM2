import numpy as np
from ALP4 import *
import time
from sample import *

one = number_image(1, 2560,1600) *(2**8-1)
two = number_image(2, 2560,1600) *(2**8-1)

# one = write_image("Q")*(2**8-1)
# two = write_image("I")*(2**8-1)
# three = write_image("T")*(2**8-1)
# four = write_image("I")*(2**8-1)
# five = write_image("QITI")*(2**8-1)



# Load the Vialux .dll
DMD = ALP4(version = '4.3')
# Initialize the device
DMD.Initialize()
DMD.ProjControl(ALP_PROJ_MODE, ALP_MASTER)

# Binary amplitude image (0 or 1)
bitDepth = 1    
# imgBlack = np.zeros([DMD.nSizeY,DMD.nSizeX])
# imgWhite = np.ones([DMD.nSizeY,DMD.nSizeX])*(2**8-1)
imgSeq  = np.concatenate([one.ravel(),two.ravel()])

#imgSeq  = np.concatenate([four.ravel(),five.ravel()])


# Allocate the onboard memory for the image sequence
DMD.SeqAlloc(nbImg = 2, bitDepth = bitDepth)
# Send the image sequence as a 1D list/array/numpy array
DMD.SeqPut(imgData = imgSeq)
# Set image rate to 50 Hz
DMD.SetTiming(pictureTime = 2000000)
#DMD.SeqControl(ALP_SEQ_REPEAT, 3)

# Run the sequence in an infinite loop
# DMD.Run()
DMD.Run(loop=True)


input("Press any key to continue")

# time.sleep(10)

# Stop the sequence display
DMD.Halt()
# Free the sequence from the onboard memory
DMD.FreeSeq()
# De-allocate the device
DMD.Free()