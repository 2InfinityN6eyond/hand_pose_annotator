import time
import sys
import numpy as np
import cv2

import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

import pykinect_azure as pykinect

DEPTH_MODE = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
COLOR_MODE = pykinect.K4A_COLOR_RESOLUTION_2160P

DEPTH_TRANSFORMED_NAME = "depth transformed"
OVERLAYED_NAME = "overlayed"

if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = COLOR_MODE
    device_config.depth_mode = DEPTH_MODE
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

    # Start device
    device = pykinect.Device(0)
    device.start(device_config)

    calibration_handle = device.get_calibration(
        DEPTH_MODE,
        COLOR_MODE
    )._handle

    print(type(calibration_handle))
    transformation = pykinect.Transformation(
        calibration_handle
    )

    cv2.namedWindow(DEPTH_TRANSFORMED_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(OVERLAYED_NAME, cv2.WINDOW_NORMAL)
    
    flags = [False, False, False]    
    while True:

        ___start_img_read = time.time()
        capture = device.update()
        flags[0], color_image = capture.get_color_image()
        ir_object = capture.get_ir_image_object()
        flags[1], ir_image = ir_object.to_numpy()
        depth_object = capture.get_depth_image_object()
        flags[2], depth_image = depth_object.to_numpy()
        ___end_img_read = time.time()

        ___start_custom_init = time.time()
        ir_custom16_image_object = pykinect.Image.create_custom_from_ir(
            ir_object
        )
        ret, custom_image = ir_custom16_image_object.to_numpy()
        ___end_custom_init = time.time()

        ___start_transform = time.time()

        ir_transformed_object, depth_transformed_object = transformation.ir_depth_image_to_color_camera_custom(
            depth_object,
            ir_custom16_image_object
        )
        ret, ir_transformed_image = ir_transformed_object.to_numpy()
        ret, depth_transformed_image = depth_transformed_object.to_numpy()
        
        ___end_transform = time.time()

        ___start_clip = time.time()

        #ir_transformed_clipped = (ir_transformed_image / ir_transformed_image.max() * 255).astype(np.uint8)
        ir_transformed_clipped = (ir_transformed_image).astype(np.uint8)

        ___end_clip = time.time()

        ___start_vis = time.time()
        
        cv2.resizeWindow(
            DEPTH_TRANSFORMED_NAME,
            ir_transformed_image.shape[1] // 2,
            ir_transformed_image.shape[0] // 2
        )
        cv2.imshow(DEPTH_TRANSFORMED_NAME, ir_transformed_clipped)
    
        ir_ggg_image = np.stack(
            [ir_transformed_clipped] * 3,
            axis = 2
        )

        cv2.resizeWindow(
            OVERLAYED_NAME,
            ir_transformed_image.shape[1] // 2,
            ir_transformed_image.shape[0] // 2
        )
        cv2.imshow(
            OVERLAYED_NAME,
            (color_image * 0.5 + ir_ggg_image * 0.5).astype(np.uint8)
        )

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break
        
        ___end_vis = time.time()

        print("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.5f}".format(
            ___end_img_read - ___start_img_read,
            ___end_custom_init - ___start_custom_init,
            ___end_transform - ___start_transform,
            ___end_clip - ___start_clip,
            ___end_vis - ___start_vis,
            ___end_vis - ___start_img_read            # whole cost
        ))