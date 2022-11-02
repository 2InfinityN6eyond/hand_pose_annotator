import sys
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)
print(sys.path)

import numpy as np
import cv2
from pykinect_azure.k4abt.body2d import Body2d
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DEPTH_MODE = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
COLOR_MODE = pykinect.K4A_COLOR_RESOLUTION_2160P

DEPTH_TRANSFORMED_NAME = "depth transformed"
OVERLAYED_NAME = "overlayed"

if __name__ == "__main__":

    pykinect.initialize_libraries(track_body=True)

    device_config = pykinect.default_configuration
    device_config.color_resolution = COLOR_MODE
    device_config.depth_mode = DEPTH_MODE
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

    device = pykinect.start_device(config=device_config)
    bodyTracker = pykinect.start_body_tracker()

    calibration = device.get_calibration(
        DEPTH_MODE,
        COLOR_MODE
    )

    cv2.namedWindow(
        'Depth image with skeleton',
        cv2.WINDOW_NORMAL
    )
    cv2.namedWindow(
        'COLOR',
        cv2.WINDOW_NORMAL
    )
    cv2.namedWindow(
        'segmentation',
        cv2.WINDOW_NORMAL
    )

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands: 

        while True:
            capture = device.update()
            body_frame = bodyTracker.update()

            ret, color_image = capture.get_color_image()
            
            depth_object = capture.get_depth_image_object()
            ir_object    = capture.get_ir_image_object()

            if not ret:
                continue
                        
            body_joints = None
            
            try :
                if body_frame.get_num_bodies() > 0 :
                    #body_joints = body_frame.get_body2d()
                    body_handle = body_frame.get_body().handle()
                    
                    body_joints = Body2d.create(
                        body_handle=body_handle,
                        calibration=calibration,
                        bodyIdx=0,
                        dest_camera = pykinect.K4A_CALIBRATION_TYPE_COLOR
                    )

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
            except Exception as e :
                print(e)

            if isinstance(color_image, np.ndarray) :
                cv2.imshow('COLOR', color_image)        
                if cv2.waitKey(1) == ord('q'):  
                    break