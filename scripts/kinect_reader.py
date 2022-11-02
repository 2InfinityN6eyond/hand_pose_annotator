import sys
import numpy as np
import cv2
import os

from multiprocessing import Process, Pipe, shared_memory

PROJ_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYKINECT_AZURE_PATH = os.path.join(PROJ_ROOT_PATH, "pyKinectAzure")
sys.path.insert(0, PYKINECT_AZURE_PATH)

import pykinect_azure as pykinect

from kinect_configs import KinectConfigs

class KinectReader(Process) :
    def __init__(
        self,
        COLOR_MODE,
        DEPTH_MODE,
        color_shm_names,
        ir_shm_names,
        depth_shm_names
    ) :
        super(KinectReader, self).__init__()
        self.color_shm_names = color_shm_names
        self.ir_shm_names = ir_shm_names
        self.depth_shm_names = depth_shm_names
        self.device_config = pykinect.default_configuration
        self.device_config.color_resolution = COLOR_MODE
        self.device_config.depth_mode = DEPTH_MODE    

    def get_this_step_idx(self) :
        shm_len = len(self.color_shm_names)
        curr_idx = 0
        while True :
            yield curr_idx
            curr_idx += 1
            if curr_idx == shm_len :
                curr_idx = 0

    def run(
        self
    ) :
        self.color_shms = list(map(
            lambda shm_name : shared_memory.SharedMemory(shm_name),
            self.color_shm_names
        ))
        self.ir_shms = list(map(
            lambda shm_name : shared_memory.SharedMemory(shm_name),
            self.ir_shm_names
        ))
        self.depth_shms = list(map(
            lambda shm_name : shared_memory.SharedMemory(shm_name),
            self.depth_shm_names
        ))

        self.

        while True :
            pass
            
        list(map(
            lambda shm : shm.close(),
            self.shms
        ))