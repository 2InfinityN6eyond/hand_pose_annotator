import cv2
import numpy as np

from multiprocessing import Process, Pipe, shared_memory

from kinect_configs import KinectConfigs
from kinect_reader  import KinectReader

if __name__ == "__main__" :
    kinect_cfg = KinectConfigs(
        color_option = KinectConfigs.COLOR_RESOLUTION_2160P,
        depth_option = KinectConfigs.DEPTH_MODE_NFOV_UNBINNED,
        fps_option = KinectConfigs.K4A_FRAMES_PER_SECOND_30
    )

    from_reader, to_main = Pipe()

    kinect_reader = KinectReader(
        reader_2_main = to_main,
        kinect_cfg = kinect_cfg,
        img_list_size = 2
    )
    color_shm_names = kinect_reader.color_shm_names
    depth_shm_names = kinect_reader.depth_shm_names
    ir_shm_names    = kinect_reader.ir_shm_names

    color_shms = list(map(
        lambda shm_name : shared_memory.SharedMemory(name = shm_name),
        color_shm_names
    ))

    print(color_shm_names)
    print(depth_shm_names)
    print(ir_shm_names)

    kinect_reader.start()

    while True :
        curr_idx = from_reader.recv()
        cv2.imshow(
            "color",
            np.ndarray(
                kinect_cfg.color_shape,
                dtype = np.uint8,
                buffer = color_shms[curr_idx].buf
            )
        )
        if cv2.waitKey(1) == ord('q'): 
            break

    print("started")

    kinect_reader.join()