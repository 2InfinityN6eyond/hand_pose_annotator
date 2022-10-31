import sys
import cv2

import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

import pykinect_azure as pykinect

DEPTH_MODE = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
COLOR_MODE = pykinect.K4A_COLOR_RESOLUTION_2160P

if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = COLOR_MODE
    device_config.depth_mode = DEPTH_MODE
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    calibration_handle = device.get_calibration(
        DEPTH_MODE,
        COLOR_MODE
    )._handle
    print(type(calibration_handle))
    transformation = pykinect.Transformation(
        calibration_handle
    )

    


    cv2.namedWindow('Transformed Color Depth Image',cv2.WINDOW_NORMAL)
    while True:
        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret, color_image = capture.get_color_image()
        ir_image_object = capture.get_ir_image_object()
        ir_image = ir_image_object.to_numpy()

        if not ret:
            continue
        
        ir_transformed = transformation.depth_image_to_color_camera(
            ir_image_object
        )

        print(ir_transformed)


        """
        # Get the colored depth
        ret, transformed_colored_depth_image = capture.get_transformed_colored_depth_image()

        # Combine both images
        combined_image = cv2.addWeighted(
            color_image[:,:,:3],
            0.7,
            transformed_colored_depth_image,
            0.3,
            0
        )
        """
    
        # Overlay body segmentation on depth image
        #cv2.imshow('Transformed Color Depth Image',combined_image)
        
        #print(ir_transformed.shape)

        #cv2.imshow("ir", ir_transformed)
        #cv2.imshow("color", color_image)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'): 
            break