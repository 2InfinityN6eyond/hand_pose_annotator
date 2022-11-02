import sys
import cv2
import os

PROJ_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYKINECT_AZURE_PATH = os.path.join(PROJ_ROOT_PATH, "pyKinectAzure")
sys.path.insert(0, PYKINECT_AZURE_PATH)

import pykinect_azure as pykinect

from multiprocessing import Process, shared_memory


class KinectReader(Process) :
	def __init__(self) :
		super(KinectReader, self).__init__()			
		pykinect.initialize_libraries()
		self.device_config = pykinect.default_configuration
		self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P

	def run(self) :
		device = pykinect.start_device(config=self.device_config)
		cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
		while True:
			capture = device.update()
			ret, color_image = capture.get_color_image()
			if not ret:
				continue
			cv2.imshow("Color Image",color_image)
			if cv2.waitKey(1) == ord('q'): 
				break

def readColorImage() :
	
	pykinect.initialize_libraries()

	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P

	device = pykinect.start_device(config=device_config)

	cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
	while True:

		capture = device.update()

		ret, color_image = capture.get_color_image()

		if not ret:
			continue
			
		# Plot the image
		cv2.imshow("Color Image",color_image)
		
		# Press q key to stop
		if cv2.waitKey(1) == ord('q'): 
			break



if __name__ == "__main__":
	p1 = Process(target = readColorImage)

	p1.start()

	p1.join()