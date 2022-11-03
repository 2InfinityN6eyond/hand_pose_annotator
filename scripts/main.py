import cv2
import numpy as np

from multiprocessing import Process, Pipe, shared_memory

from kinect_configs import KinectConfigs
from frame_processor import SharedFrameProcessor
from kinect_reader  import KinectReader

if __name__ == "__main__" :

    num_frame_processor = 10

    kinect_cfg = KinectConfigs(
        color_option = KinectConfigs.COLOR_RESOLUTION_2160P,
        depth_option = KinectConfigs.DEPTH_MODE_NFOV_UNBINNED,
        fps_option = KinectConfigs.K4A_FRAMES_PER_SECOND_30
    )

    shared_frame_processors = list(map(
        lambda x : SharedFrameProcessor(x, kinect_cfg, *Pipe()),
        range(num_frame_processor)
    ))
    shared_ctype_arrays_color = list(map(lambda x : x.color_shbuffer, shared_frame_processors))
    shared_ctype_arrays_depth = list(map(lambda x : x.depth_shbuffer, shared_frame_processors))
    shared_ctype_arrays_ir = list(map(lambda x : x.ir_shbuffer, shared_frame_processors))
    to_shared_frame_processor_pipes = list(map(lambda x : x.to_proc_pipe, shared_frame_processors))

    list(map(lambda x : x.start(), shared_frame_processors))

    kinect_reader = KinectReader(
        kinect_cfg = kinect_cfg,
        shared_ctype_arrays_color = shared_ctype_arrays_color,
        shared_ctype_arrays_ir = shared_ctype_arrays_ir,
        shared_ctype_arrays_depth = shared_ctype_arrays_depth,
        to_processor_pipes = to_shared_frame_processor_pipes
    )
    kinect_reader.start()

    while True :
        
        if cv2.waitKey(1) == ord('q'): 
            break

    print("started")

    kinect_reader.join()

    list(map(lambda x : x.join(), shared_frame_processors))