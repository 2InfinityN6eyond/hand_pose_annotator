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
        reader_2_main:Pipe,
        kinect_cfg : KinectConfigs,
        img_list_size = 10,
    ) :
        super(KinectReader, self).__init__()
        self.reader_2_main = reader_2_main
        self.kinect_cfg = kinect_cfg

        # initialize shared memory for color image
        self.color_shms = list(map(
            lambda i : shared_memory.SharedMemory(
                create = True,
                size = kinect_cfg.color_bufsize
            ),
            range(img_list_size)
        ))
        self.ir_shms = list(map(
            lambda i : shared_memory.SharedMemory(
                create = True,
                size = kinect_cfg.ir_bufsize
            ),
            range(img_list_size)
        ))
        self.depth_shms = list(map(
            lambda i : shared_memory.SharedMemory(
                create = True,
                size = kinect_cfg.depth_bufsize
            ),
            range(img_list_size)
        ))
        
        self.color_shm_names = list(map(
            lambda shm : shm.name,
            self.color_shms
        ))
        self.ir_shm_names = list(map(
            lambda shm : shm.name,
            self.ir_shms
        ))
        self.depth_shm_names = list(map(
            lambda shm : shm.name,
            self.depth_shms
        ))

        self.device_config = pykinect.default_configuration
        self.device_config.color_resolution = kinect_cfg.color_option
        self.device_config.depth_mode = kinect_cfg.depth_option
        self.device_config.camera_fps = kinect_cfg.fps_option

        pykinect.initialize_libraries()
        
    def run(
        self
    ) :

        # idx looper.
        idx_emmiter = self.get_this_step_idx()
        
        # start azure kinct

        #device.start(self.device_config)
        pykinect.initialize_libraries()
        device = pykinect.start_device(config=self.device_config)
        
        flags = [False, False, False]
        while True :
            curr_idx = next(idx_emmiter)
            
            capture = device.update()
            flags[0], color_image = capture.get_color_image()
            flags[1], ir_image    = capture.get_ir_image()
            flags[2], depth_image = capture.get_depth_image()

            if all(flags) == False :
                print(flags)
                continue

            color_array = np.ndarray(
                self.kinect_cfg.color_shape,
                dtype = self.kinect_cfg.color_dtype,
                buffer = self.color_shms[curr_idx].buf
            )
            color_array[:] = color_image[:]

            self.reader_2_main.send(curr_idx)

            if 1 == 2 :
                break
        
        list(map(
            lambda shm : shm.close(),
            self.color_shms
        ))
        list(map(
            lambda shm : shm.close(),
            self.ir_shms
        ))
        list(map(
            lambda shm : shm.close(),
            self.depth_shms
        ))

    def get_this_step_idx(self) :
        """
        iterate 0, 1, 2 ... len(self.color_shm_names) -1 forever
        """
        shm_len = len(self.color_shm_names)
        curr_idx = 0
        while True :
            yield curr_idx
            curr_idx += 1
            if curr_idx == shm_len :
                curr_idx = 0


