import sys
import numpy as np
import cv2
import os

import ctypes
import multiprocessing
from multiprocessing import Process, sharedctypes, Pipe, shared_memory

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
        my_idx,
        kinect_cfg : KinectConfigs,
        to_proc_pipe : Pipe,
        from_proc_pipe : Pipe,
    ) :
        super(SharedFrameProcessor, self).__init__()

        self.my_idx = my_idx
        self.kinect_cfg = kinect_cfg
        self.to_proc_pipe = to_proc_pipe
        self.from_proc_pipe = from_proc_pipe

        self.color_shbuffer = sharedctypes.RawArray(
            ctypes.c_ubyte,
            kinect_cfg.color_bufsize
        )
        self.depth_shbuffer = sharedctypes.RawArray(
            ctypes.c_ubyte,
            kinect_cfg.depth_bufsize
        )
        self.ir_shbuffer = sharedctypes.RawArray(
            ctypes.c_ubyte,
            kinect_cfg.ir_bufsize
        )

        self.color_sha = shared_memory.SharedMemory(
            create = True,
            size = kinect_cfg.color_bufsize
        )
        self.color_sha_name = self.color_sha.name
        self.depth_transformed_sha = shared_memory.SharedMemory(
            create = True,
            size = kinect_cfg.depth_bufsize
        )
        self.depth_transformed_sha_name = self.depth_transformed_sha.name
        self.ir_transformed_sha = shared_memory.SharedMemory(
            create = True,
            size = kinect_cfg.ir_bufsize
        )
        self.ir_transformed_sha_name = self.ir_transformed_sha.name

    def run(self) :
        # initalize array   from shared memory
        color_ndarray = np.ndarray(
            self.kinect_cfg.color_shape,
            dtype = self.kinect_cfg.color_dtype,
            buffer = self.color_sha.buf
        )
        ir_ndarray = np.ndarray(
            self.kinect_cfg.ir_shape,
            self.kinect_cfg.ir_dtype,
            self.ir_transformed_sha.buf
        )
        depth_ndarray = np.ndarray(
            self.kinect_cfg.depth_shape,
            self.kinect_cfg.depth_dtype,
            self.depth_transformed_sha.buf
        )

        # initialize library
        pykinect.pykinect.initialize_libraries()

        while True :
            color_bufsize = self.from_proc_pipe.recv()
            color_ndarray[:] = cv2.imdecode(
                np.frombuffer(
                    np.ctypeslib.as_array(
                        self.color_shbuffer,
                        shape=(color_bufsize,)
                    ),
                    dtype = np.uint8
                ),
                -1
            )[:]


            cv2.imshow("color", color_ndarray)
            if cv2.waitKey(1) == ord('q'):  
                break            

            '''
            ir_image = np.frombuffer(
                np.ctypeslib.as_array(
                    buffer_pointer,
                    shape=(image_size,)
                ),
                dtype="<u2"
            ).copy().reshape(self.kinect_cfg.ir_shape)

            depth_image = np.frombuffer(
                np.ctypeslib.as_array(
                    buffer_pointer,
                    shape=(image_size,)
                ),
                dtype="<u2"
            ).copy().reshape(self.kinect_cfg.depth_shape)

            '''