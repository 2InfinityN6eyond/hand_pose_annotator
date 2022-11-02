import sys
import numpy as np
import cv2
import os

import multiprocessing
from multiprocessing import Process, sharedctypes, Pipe

PROJ_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYKINECT_AZURE_PATH = os.path.join(PROJ_ROOT_PATH, "pyKinectAzure")
sys.path.insert(0, PYKINECT_AZURE_PATH)
import pykinect_azure as pykinect

from kinect_configs import KinectConfigs

class SharedFrameProcessor(Process) :
    """
    shared_array
    transform ir, depth image to color image and preprocess
    body joints
    """

    def __init__(
        self,
        kinect_cfg : KinectConfigs
    ) :        
        self.color_buffer
        self.depth_buffer
        self.ir_buffer

        self.color_array
        self.depth_transformed_array
        self.ir_transformed_array

    def run(self) :
        pass

    