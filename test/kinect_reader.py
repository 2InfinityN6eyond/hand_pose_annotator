
import sys
import numpy as np
import cv2
import os

from multiprocessing import Process, Pipe, shared_memory

PROJ_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYKINECT_AZURE_PATH = os.path.join(PROJ_ROOT_PATH, "pyKinectAzure")
sys.path.insert(0, PYKINECT_AZURE_PATH)

import pykinect_azure as pykinect

K4A_DEPTH_MODE_OFF = 0
K4A_DEPTH_MODE_NFOV_2X2BINNED = 1
K4A_DEPTH_MODE_NFOV_UNBINNED = 2
K4A_DEPTH_MODE_WFOV_2X2BINNED = 3
K4A_DEPTH_MODE_WFOV_UNBINNED = 4
K4A_DEPTH_MODE_PASSIVE_IR = 5

K4A_COLOR_RESOLUTION_OFF = 0
K4A_COLOR_RESOLUTION_720P = 1
K4A_COLOR_RESOLUTION_1080P = 2
K4A_COLOR_RESOLUTION_1440P = 3
K4A_COLOR_RESOLUTION_1536P = 4
K4A_COLOR_RESOLUTION_2160P = 5
K4A_COLOR_RESOLUTION_3072P = 6

class KinectReader(Process) :
    def __init__(
        self,
        shm_name
    ) :
        super(KinectReader, self).__init__()
        self.shm_name = shm_name

    def run(
        self
    ) :
        pass

    