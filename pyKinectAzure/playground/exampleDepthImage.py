import sys
import cv2

import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

import pykinect_azure as pykinect

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries()

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_2160P
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_PASSIVE_IR
	device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

	# Start device
	device = pykinect.start_device(config=device_config)

	cv2.namedWindow('Depth Image',cv2.WINDOW_NORMAL)
	cv2.namedWindow('IR Image',cv2.WINDOW_NORMAL)

	while True:

		# Get capture
		capture = device.update()

		# Get the color depth image from the capture
		ret1, depth_image = capture.get_depth_image()
		if not ret1:
			continue
			
		ret, ir_image = capture.get_ir_image()
		if not ret:
			continue
		
		print(
			ir_image.dtype, ir_image.shape,
			ir_image.min(), ir_image.max(), ir_image.mean()
		)
		print(
			depth_image.dtype, depth_image.shape,
			depth_image.min(), depth_image.max(), depth_image.mean()
		)
		print()

		# Plot the image
		cv2.resizeWindow('Depth Image', depth_image.shape[1], depth_image.shape[0])
		cv2.imshow('Depth Image',depth_image / 256)
		
		cv2.resizeWindow('IR Image', ir_image.shape[1], ir_image.shape[0])
		cv2.imshow('IR Image', ir_image * 10)

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'):  
			break