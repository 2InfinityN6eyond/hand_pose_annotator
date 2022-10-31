import sys
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)
print(sys.path)

import cv2
from pykinect_azure.k4abt.body2d import Body2d
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


IR_WINDOW_NAME = "infrared"
DEPTH_WINDOW_NAME = "depth"
COLOR_WINDOW_NAME = "color"

COLOR_RESOLUTION = pykinect.K4A_COLOR_RESOLUTION_2160P
DEPTH_MODE = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED


if __name__ == "__main__":
    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = COLOR_RESOLUTION
    device_config.depth_mode = DEPTH_MODE
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

    device = pykinect.start_device(config=device_config)
    
    bodyTracker = pykinect.start_body_tracker()

    calibration = device.get_calibration(
        pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED,
        pykinect.K4A_COLOR_RESOLUTION_1440P
    )

    cv2.namedWindow( 
        DEPTH_WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )
    cv2.namedWindow(
        COLOR_WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )
    cv2.namedWindow(
        'segmentation',
        cv2.WINDOW_NORMAL
    )
    cv2.namedWindow(
        IR_WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    image_read_succeed = [False] * 6

    while True:
        capture = device.update()
        body_frame = bodyTracker.update()

        image_read_succeed[0], color_image       = capture.get_color_image()
        image_read_succeed[1], depth_color_image = capture.get_colored_depth_image()
        image_read_succeed[2], body_image_color  = body_frame.get_segmentation_image()
        image_read_succeed[3], ir_image          = capture.get_ir_image()
        image_read_succeed[4], depth_image       = capture.get_depth_image()
        image_read_succeed[5], transformed_color_image = capture.get_transformed_color_image()

        if False in image_read_succeed :
            print(image_read_succeed)
            continue
        
        #print(depth_image.shape, depth_image.dtype, "|", ir_image.shape, ir_image.dtype)
        print(transformed_color_image.shape)

        cv2.resizeWindow(IR_WINDOW_NAME, ir_image.shape[1], ir_image.shape[0])
        cv2.imshow(IR_WINDOW_NAME, ir_image / 256)# * 10)

        cv2.resizeWindow(COLOR_WINDOW_NAME, color_image.shape[1], color_image.shape[0])
        cv2.imshow(COLOR_WINDOW_NAME, color_image)

        cv2.resizeWindow(DEPTH_WINDOW_NAME, depth_image.shape[1], depth_image.shape[0])
        cv2.imshow(DEPTH_WINDOW_NAME, depth_image / 256) # / 256)

        cv2.resizeWindow("segmentation", transformed_color_image.shape[1], transformed_color_image.shape[0])
        cv2.imshow("segmentation", transformed_color_image) # / 256)

        if cv2.waitKey(1) == ord('q'):  
            break

        continue


        combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
        combined_image = body_frame.draw_bodies(combined_image)
        
        body_joints = None





        try :
            if body_frame.get_num_bodies() > 0 :
                body_joints = body_frame.get_body2d()
                body_handle = body_frame.get_body().handle()
                
                body_joints = Body2d.create(
                    body_handle=body_handle,
                    calibration=calibration,
                    bodyIdx=0,
                    dest_camera= pykinect.K4A_CALIBRATION_TYPE_COLOR
                )
                print(body_joints.joints[7].get_coordinates())

        except Exception as e : 
            print(e)
            print('-------')
            continue

        try :            
            left_hand_left_end = min(
                body_joints.joints[7].get_coordinates()[0],
                body_joints.joints[9].get_coordinates()[0]
            )
            left_hand_right_end = max(
                body_joints.joints[7].get_coordinates()[0],
                body_joints.joints[9].get_coordinates()[0]
            )
            left_hand_top_end = min(
                body_joints.joints[7].get_coordinates()[1],
                body_joints.joints[9].get_coordinates()[1]
            )
            left_hand_bottom_end = max(
                body_joints.joints[7].get_coordinates()[1],
                body_joints.joints[9].get_coordinates()[1]
            )

            bbox_shape = max(left_hand_right_end - left_hand_left_end, left_hand_bottom_end - left_hand_top_end)

            left_hand_center_x = int((left_hand_right_end + left_hand_left_end) / 2)
            left_hand_center_y = int((left_hand_top_end + left_hand_bottom_end) / 2)

            left_hand_left_end   = max(left_hand_center_x - bbox_shape, 0)
            left_hand_right_end  = min(left_hand_center_x + bbox_shape, color_image.shape[1])
            left_hand_top_end    = max(left_hand_center_y - bbox_shape, 0)
            left_hand_bottom_end = min(left_hand_center_y + bbox_shape, color_image.shape[0])
            
            hand_cropped_image = color_image[left_hand_top_end:left_hand_bottom_end, left_hand_left_end:left_hand_right_end].copy()

            cv2.line(
                color_image,
                body_joints.joints[7].get_coordinates(),
                body_joints.joints[9].get_coordinates(),
                (255,255),
                10,
                cv2.LINE_8
            )

            cv2.rectangle(
                color_image,
                (left_hand_left_end, left_hand_top_end),
                (left_hand_right_end, left_hand_bottom_end),
                (255,255,255),
                3,
                cv2.LINE_8
            )        


            image = cv2.cvtColor(hand_cropped_image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )


            cv2.imshow('segmentation', image)
            if cv2.waitKey(1) == ord('q'):  
                break

        except Exception as e :
            print(e)

        cv2.imshow('Depth image with skeleton',combined_image)
        if cv2.waitKey(1) == ord('q'):  
            break

        cv2.imshow('COLOR', color_image)        
        if cv2.waitKey(1) == ord('q'):  
            break
