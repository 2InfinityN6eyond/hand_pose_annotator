
import sys
import numpy as np
import cv2
import os

from multiprocessing import Process, Pipe, shared_memory

class ImageTransformer