import time
import sys
import cv2
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

import pykinect_azure as pykinect

if __name__ == "__main__":
    pykinect.initialize_libraries()
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
    device = pykinect.start_device(config=device_config)

    cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
    while True:
        ret = False

        ___start_img_read = time.time()
        
        _____start_update = time.time()
        capture = device.update()
        _____end_update = time.time()

        _____start_get_image = time.time()
        #ret, color_image = capture.get_color_image()
        _____end_get_image = time.time()

        _____start_get_object = time.time()
        color_object = capture.get_color_image_object()
        _____end_get_object = time.time()



        _____start_convert = time.time()
        ret, color_image = color_object.to_numpy()
        _____end_convert = time.time()


        ___end_img_read = time.time()

        if not ret :
            continue


        ___start_vis = time.time()

        # Plot the image
        cv2.imshow("Color Image",color_image)
        
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break

        ___end_vis = time.time()

        
        print("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4}".format(
            
            _____end_update - _____start_update,
            _____end_get_image - _____start_get_image,
            _____end_get_object - _____start_get_object,

            _____end_convert - _____start_convert,
            
            color_object._handle.contents
            
            #___end_img_read - ___start_img_read,
            #___end_vis - ___start_vis,
            #___end_vis - ___start_img_read            # whole cost
        ))