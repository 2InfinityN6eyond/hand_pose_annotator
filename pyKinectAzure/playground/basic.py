import time
import sys
import cv2
import os

from multiprocessing import sharedctypes

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

import pykinect_azure as pykinect

COLOR_MODE = pykinect.K4A_COLOR_RESOLUTION_2160P
DEPTH_MODE = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED

if __name__ == "__main__":
    pykinect.initialize_libraries()

    device_config = pykinect.default_configuration
    device_config.color_resolution = COLOR_MODE
    device_config.depth_mode       = DEPTH_MODE
    device = pykinect.start_device(config=device_config)
    
    while True:
        capture = device.update()
        color_object = capture.get_color_image_object()
        depth_object = capture.get_depth_image_object()
        ir_object    = capture.get_ir_image_object()

        if not all([color_object, depth_object, ir_object]) :
            continue

        print(
            color_object.get_size(),
            depth_object.get_size(),
            ir_object.get_size(),
            "  ",
            type(color_object.get_buffer()),
            type(depth_object.get_buffer()),
            type(ir_object.get_buffer())
        )


        '''
        print("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4}".format(
            color_object._handle.contents
            
            #___end_img_read - ___start_img_read,
            #___end_vis - ___start_vis,
            #___end_vis - ___start_img_read            # whole cost
        ))
        '''