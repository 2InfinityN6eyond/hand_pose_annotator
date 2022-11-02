import time
import sys
import cv2
import os
import numpy as np

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

import pykinect_azure as pykinect

import ctypes
import multiprocessing
from multiprocessing import Process, shared_memory, sharedctypes, Pipe

class KinectReader(Process) :
    def __init__(
        self,
        to_main:Pipe,
        num_shm_sets = 10
    ) :
        super(KinectReader, self).__init__()
        self.to_main = to_main
        self.num_shm_sets = num_shm_sets
        self.device_config = pykinect.default_configuration
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
        self.sha_list = list(map(
            lambda x : sharedctypes.RawArray(
                ctypes.c_ubyte,
                3840 * 2160 * 3
            ),
            range(self.num_shm_sets)
        ))
        
    def run(self) :
        idx_emmiter = self.get_this_step_idx()

        pykinect.initialize_libraries()
        device = pykinect.start_device(config=self.device_config)
        while True:
            ____start = time.time()

            capture = device.update()
            color_image_object = capture.get_color_image_object()
            
            if color_image_object is None or color_image_object.get_size() is None :
                continue

            nbytes = color_image_object.get_size()

            curr_idx = next(idx_emmiter)
            ctypes.memmove(
                self.sha_list[curr_idx],
                color_image_object.get_buffer(),
                nbytes
            )
            self.to_main.send(
                [curr_idx, nbytes]
            )
            ____end = time.time()
            print(____end - ____start)


    def get_this_step_idx(self) :
        """
        iterate 0, 1, 2 ... len(self.color_shm_names) -1 forever
        """
        shm_len = self.num_shm_sets
        curr_idx = 0
        while True :
            yield curr_idx
            curr_idx += 1
            if curr_idx == shm_len :
                curr_idx = 0

if __name__ == "__main__":

    num_sha_set = 5
    windows_names = list(map(str, range(num_sha_set)))
    '''
    cv_windows = list(map(
        lambda name : cv2.namedWindow(name, cv2.WINDOW_NORMAL),
        windows_names
    ))
    '''

    main_2_reader, reader_2_main = Pipe()
    reader = KinectReader(
        reader_2_main,
        num_shm_sets = num_sha_set
    )
    sha_list = reader.sha_list

    reader.start()


    while True :
        idx, size = main_2_reader.recv()
        if idx == 0 :
            buffer_array = np.frombuffer(
                np.ctypeslib.as_array(sha_list[idx], shape=(size))
            )
            color_image = cv2.imdecode(
                np.frombuffer(buffer_array, dtype=np.uint8),
                -1
            )
            #print(type(buffer_array), buffer_array.shape, type(color_image), color_image.shape)

            cv2.imshow("1", color_image)

        if cv2.waitKey(1) == ord('q'):  
            break
    list(map(
        lambda child_process : child_process.terminate(),
        multiprocessing.active_children()
    ))
    reader.join()