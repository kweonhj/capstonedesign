#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:34:54 2023

@author: kweonhyuckjin
"""

import numpy as np
from MagneticFieldCalculator import MagneticFieldCalculator

class CalculateActuationMatrix:
    
    def __init__(self, desired_B, position):
        self.criteria_currents = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        self.desired_B = desired_B # For determine the robot's orientation
        self.position = position # current position

    def calculate(self):

        calculator0 = MagneticFieldCalculator(self.criteria_currents)
        B0 = calculator0.calculate_magnetic_field(self.position)
        F0 = calculator0.calculate_magnetic_force(self.position)
        B0_matrix = np.array(calculator0.B_field).T
        F0_matrix = np.array(calculator0.F_field).T

        delta = 1e-6
        position2 = self.position + [delta, 0, 0]
        calculator2 = MagneticFieldCalculator(self.criteria_currents)
        B2 = calculator2.calculate_magnetic_field(position2)
        B2_matrix = np.array(calculator2.B_field).T
        gradient_Bx = (B2_matrix - B0_matrix) / delta

        position3 = self.position + [0, delta, 0]
        calculator3 = MagneticFieldCalculator(self.criteria_currents)
        B3 = calculator3.calculate_magnetic_field(position3)
        B3_matrix = np.array(calculator3.B_field).T
        gradient_By = (B3_matrix - B0_matrix) / delta

        position4 = self.position + [0, 0, delta]
        calculator4 = MagneticFieldCalculator(self.criteria_currents)
        B4 = calculator4.calculate_magnetic_field(position4)
        B4_matrix = np.array(calculator4.B_field).T
        gradient_Bz = (B4_matrix - B0_matrix) / delta

        m2_scal = 1.210 * 10**(-9) * 2/(4 * np.pi * 10**(-7))
        m2_moment = m2_scal * self.desired_B/np.linalg.norm(self.desired_B) 
        m2 = m2_moment
        
        F1 = np.dot(m2, gradient_Bx)
        F2 = np.dot(m2, gradient_By)
        F3 = np.dot(m2, gradient_Bz)
        A = np.zeros((6, 8))
        A[0:3, :] = B0_matrix
        A[3, :] = F1
        A[4, :] = F2
        A[5, :] = F3
        self.actuation_matrix = A
        
        return self.actuation_matrix
    
    def calculate_coil_currents(self):
        BFmatrix = np.array([self.total_field,
                    self.total_force]).reshape(6,)
        actuation_matrix = self.actuation_matrix
        inverse_actuation_matrix = np.linalg.pinv(actuation_matrix)
        self.coil_currents = np.dot(inverse_actuation_matrix, BFmatrix)
        
        return self.coil_currents




