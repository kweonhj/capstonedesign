#!/usr/bin/env python3
# -*- coding: utf-8 -*-`
"""
Created on Mon May 22 14:38:10 2023

@author: kweonhyuckjin
"""

import numpy as np
from ObjectDetectionController import ObjectDetectionController
import pyfirmata as pf

port1 = '/dev/cu.usbmodem1101' #mega1 usb port name(1:12)
ard1 = pf.ArduinoMega(port1)

# custom_weights_path = '/Users/kweonhyuckjin/miniforge3/envs/dysurf/capstonedesign-main/best1120.pt'
custom_weights_path = '/Users/kweonhyuckjin/Downloads/0530/last0530.pt'
camera_indices = [0, 1]
width = 640
height = 480
P = 5e-4
I = 0
D = 9e-5
setpoint_xy = np.array([0.5, 0.5])
setpoint_z = np.array([0.5])
desired_B = np.array([5e-3, 0, 0])

obj_detection = ObjectDetectionController(custom_weights_path, camera_indices, width, height, P, I, D, setpoint_xy, setpoint_z, ard1, desired_B)
obj_detection.run()







