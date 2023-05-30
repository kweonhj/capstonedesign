#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:23:17 2023

@author: kweonhyuckjin
"""

import numpy as np
from ArduinoSetting import ArduinoSetting 
import pyfirmata as pf

port1 = '/dev/cu.usbmodem1101' #mega1 usb port name(1:12)
ard = pf.ArduinoMega(port1)
ArduinoControl = ArduinoSetting(ard)
ArduinoControl.arduino_setting()
# Create a random vector of shape (6, 1) with values ranging from -1 to 1
#random_vector = np.random.uniform(low=-1, high=1, size=8)
#random_vector = [0,0,0,0,0,0,0,0]
#ArduinoControl.MotordriveControl(random_vector)
ArduinoControl.MotordriveControl([0., 0., 0., 1., 0., 0., 0., 0.1])
ArduinoControl.arduino_initalize()
