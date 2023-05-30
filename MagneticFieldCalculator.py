#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 17:45:18 2023

@author: kweonhyuckjin
"""

import numpy as np

class MagneticFieldCalculator:
    
    def __init__(self, currents):
        
        self.coil_centroids = np.array([[-0.03, 0, 0.0288],
                        [0, 0.03, 0.0288],
                        [0.03, 0, 0.0288],
                        [0, -0.03, 0.0288],
                        [-0.0312, 0.0312, 0.0027],
                        [0.0312, 0.0312, 0.0027],
                        [0.0312, -0.0312, 0.0027],
                        [-0.0312, -0.0312, 0.0027]])
        self.magnetic_moments = np.array([[1.5836, -0.0051, -2.1707],
                        [-0.0051, -1.5836, -2.1707],
                        [-1.5836, 0.0051, -2.1707],
                        [0.0051, 1.5836, -2.1707],
                        [2.3077, -2.3103, -1.2661],
                        [-2.3103, -2.3077, -1.2661],
                        [-2.3077, 2.3103, -1.2661],
                        [2.3103, 2.3077, -1.2661]])
        
        self.currents = currents
        
    def calculate_magnetic_field(self, m2_position):
        
        mu0 = 4 * np.pi * 1e-7
        total_field = np.zeros(3)
        self.B_field = []
    
        for centroid, moment, current in zip(self.coil_centroids, self.magnetic_moments, self.currents):
            displacement = m2_position - centroid
            r_norm = np.linalg.norm(displacement)
            r_hat = displacement / r_norm
            moment = moment * current
            field = (mu0 / (4 * np.pi)) * (3 * r_hat * (np.dot(r_hat, moment)) - moment) / r_norm**3
            field = np.array(field).flatten()  # Flatten the field to 1D
            total_field += field
            self.B_field.append(field)
            self.total_field = total_field

        return self.total_field

    
    def calculate_magnetic_force(self, m2_position):
        # implementation of calculate_magnetic_force method
        self.F_field = [];
        mu0 = 4 * np.pi * 1e-7
        total_force = np.zeros(3)
        m2_scal = 1.210 * 10**(-9) * 2/(4 * np.pi * 10**(-7))
        m2_moment = m2_scal * self.total_field/np.linalg.norm(self.total_field) 
        self.m2_moment = m2_moment
        
        for centroid, moment, current in zip(self.coil_centroids, self.magnetic_moments, self.currents):
            displacement = m2_position - centroid
            r_norm = np.linalg.norm(displacement)
            r_hat = displacement / r_norm
            moment = moment * current
            m1_dot_r_hat = np.dot(moment, r_hat)
            m2_dot_r_hat = np.dot(m2_moment, r_hat)
            m1_dot_m2 = np.dot(moment, m2_moment)
            r_hat_dot_m1_dot_r_hat = np.dot(r_hat, m1_dot_r_hat)
            r_hat_dot_m2_dot_r_hat = np.dot(r_hat, m2_dot_r_hat)
            
            force = (3 * mu0 / (4 * np.pi * r_norm**4)) * (
                    moment * m2_dot_r_hat +
                    m2_moment * m1_dot_r_hat +
                    r_hat * m1_dot_m2 -
                    5 * r_hat * m1_dot_r_hat * m2_dot_r_hat
                )
            
            force = np.array(force)
            force.reshape(3,)
            total_force += force
            self.F_field.append(force)
            self.total_force = total_force

        return self.total_force
    
    
if __name__ == '__main__':
    calculator = MagneticFieldCalculator()
    
    
    
    
    
    
    
    
    
    