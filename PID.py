#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 21:01:43 2023

@author: kweonhyuckjin
"""

import time
import numpy as np

class PID:
    """PID Controller
    """

    def __init__(self, P, I, D, Setpoint, current_time=None):
        self.Kp = P
        self.Ki = I
        self.Kd = D 
        self.Setpoint = np.array(Setpoint)
        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = np.zeros_like(self.Setpoint)

        self.PTerm = np.zeros_like(self.Setpoint)
        self.ITerm = np.zeros_like(self.Setpoint)
        self.DTerm = np.zeros_like(self.Setpoint)
        self.last_error = np.zeros_like(self.Setpoint)

        # Windup Guard
        self.int_error = np.zeros_like(self.Setpoint)
        self.windup_guard = 20.0

        self.output = np.zeros_like(self.Setpoint)

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback"""
        error = self.SetPoint - np.array(feedback_value)

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time
            self.DTerm = np.zeros_like(self.Setpoint)
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
            return self.output

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain
        
    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sample time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

if __name__ == '__main__':
    PID()
