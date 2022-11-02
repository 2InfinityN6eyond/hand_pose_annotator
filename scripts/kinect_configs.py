import numpy as np

class KinectConfigs :
    COLOR_RESOLUTION_OFF = 0
    COLOR_RESOLUTION_720P = 1
    COLOR_RESOLUTION_1080P = 2
    COLOR_RESOLUTION_1440P = 3
    COLOR_RESOLUTION_1536P = 4
    COLOR_RESOLUTION_2160P = 5
    COLOR_RESOLUTION_3072P = 6
    
    DEPTH_MODE_OFF = 0
    DEPTH_MODE_NFOV_2X2BINNED = 1
    DEPTH_MODE_NFOV_UNBINNED = 2
    DEPTH_MODE_WFOV_2X2BINNED = 3
    DEPTH_MODE_WFOV_UNBINNED = 4
    DEPTH_MODE_PASSIVE_IR = 5

    K4A_FRAMES_PER_SECOND_5 = 0
    K4A_FRAMES_PER_SECOND_15 = 1
    K4A_FRAMES_PER_SECOND_30 = 2


    color_widths = [0, 1280, 1920, 2560, 2048, 3840, 4096]
    color_heights = [0, 720, 1080, 1440, 1536, 2160, 3072]
    depth_widths =  [0, 320, 640, 512, 1024, 1024]
    depth_heights = [0, 288, 576, 512, 1024, 1024]

    def __init__(
        self,
        color_option,
        depth_option,
        fps_option
    ) :
        self.color_option = color_option
        self.depth_option = depth_option

        self.color_width  =  KinectConfigs.color_widths[color_option]
        self.color_height = KinectConfigs.color_heights[color_option]
        self.color_shape = (self.color_height, self.color_width, 3)        
        self.color_dtype = np.uint8
        self.color_bufsize = self.color_width * self.color_height * 3

        self.depth_width  =  KinectConfigs.depth_widths[depth_option]
        self.depth_height = KinectConfigs.depth_heights[depth_option]
        self.depth_shape = (self.depth_height, self.depth_width)
        self.depth_dtype = np.uint16
        self.depth_bufsize = self.depth_width * self.depth_height * 2

        self.ir_width  = KinectConfigs.depth_widths[depth_option]
        self.ir_height = KinectConfigs.depth_heights[depth_option]
        self.ir_shape = (self.ir_height, self.ir_width)
        self.ir_dtype = np.uint16
        self.ir_bufsize = self.ir_width * self.ir_height * 2

        self.fps_option = fps_option