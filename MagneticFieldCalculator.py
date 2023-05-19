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
                        [-0.312, 0.0312, 0.0027],
                        [0.0312, 0.312, 0.0027],
                        [0.312, -0.0312, 0.0027],
                        [-0.0312, -0.312, 0.0027]])
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
        # implementation of calculate_magnetic_field method
        mu0 = 4 * np.pi * 1e-7
        total_field = np.zeros(3)
        self.B_field = [];
        for centroid, moment, current in zip(self.coil_centroids, self.magnetic_moments, self.currents):
            displacement = m2_position - centroid
            r_norm = np.linalg.norm(displacement)
            r_hat = displacement / r_norm
            moment = moment * current
            field = (mu0 / (4 * np.pi)) * (3 * r_hat * (np.dot(r_hat, moment)) - moment) / r_norm**3
            field = np.array(field)
            field.reshape(3,)
            total_field += field
            self.B_field.append(field)
            self.total_field = total_field
            
        print(np.array(self.B_field)[:,0])
        return self.total_field
    
    def calculate_magnetic_force(self, m2_position):
        # implementation of calculate_magnetic_force method
        
        mu0 = 4 * np.pi * 1e-7
        total_force = np.zeros(3)
        m2_scal = 1.210 * 10**(-9)/(4 * np.pi * 10**(-7))
        m2_moment = m2_scal * self.total_field/np.linalg.norm(self.total_field) 
        self.m2_moment = m2_moment
        
        for centroid, moment, current in zip(self.coil_centroids, self.magnetic_moments, self.currents):
            displacement = m2_position - centroid
            r_norm = np.linalg.norm(displacement)
            r_hat = displacement / r_norm
            m1_dot_r_hat = np.dot(moment, r_hat)
            m2_dot_r_hat = np.dot(m2_moment, r_hat)
            m1_dot_m2 = np.dot(moment, m2_moment)
            r_hat_dot_m1_dot_r_hat = np.dot(r_hat, m1_dot_r_hat)
            r_hat_dot_m2_dot_r_hat = np.dot(r_hat, m2_dot_r_hat)
            force = (3 * mu0 / (4 * np.pi * r_norm**4)) * (
                    moment * m2_dot_r_hat +
                    m2_moment * m1_dot_r_hat +
                    r_hat * m1_dot_m2 -
                    5 * r_hat * r_hat_dot_m1_dot_r_hat * r_hat_dot_m2_dot_r_hat
                )
            force = np.array(force)
            force.reshape(3,)
            total_force += force
            self.total_force = total_force

        return self.total_force
    
    def calculate_delta_B(self, m2_position):
        
        self.delta = 1e-8  # Small perturbation value
        self.B = np.array(self.B_field)
        B_perturbed = self.calculate_magnetic_field(m2_position + self.delta)
        self.B_delta = np.array(self.B_field)
        
    def calculate_partial_derivative(self, m2_position, variable):
        if variable == 'x':
            partial_B_delta = self.B_delta[:, 0]  # Extract the x-component of B_delta
            partial_B = self.B[:, 0]  # Extract the x-component of B
        elif variable == 'y':
            partial_B_delta = self.B_delta[:, 1]  # Extract the y-component of B_delta
            partial_B = self.B[:, 1]  # Extract the y-component of B
        elif variable == 'z':
            partial_B_delta = self.B_delta[:, 2]  # Extract the z-component of B_delta
            partial_B = self.B[:, 2]  # Extract the z-component of B
        else:
            raise ValueError("Invalid variable")
        
        self.partial_derivative = (partial_B_delta - partial_B) / self.delta
        
        return self.partial_derivative


    def calculate_actuation_matrix(self, m2_position):
        num_coils = self.magnetic_moments.shape[0]
        num_axes = 3
        self.calculate_delta_B(m2_position)
        self.actuation_matrix = np.zeros((num_axes + num_axes, num_coils))

        # Step 2: Compute partial derivatives of B with respect to spatial coordinates
        partial_B_partial_x = self.calculate_partial_derivative(m2_position, 'x')
        partial_B_partial_y = self.calculate_partial_derivative(m2_position, 'y')
        partial_B_partial_z = self.calculate_partial_derivative(m2_position, 'z')
    
        for i in range(num_coils):
            moment = self.magnetic_moments[i]
            
            # Step 3: Calculate force M^t
            force_x = np.dot(self.m2_moment[0], partial_B_partial_x[i])
            force_y = np.dot(self.m2_moment[1], partial_B_partial_y[i])
            force_z = np.dot(self.m2_moment[2], partial_B_partial_z[i])
    
            # Step 4: Construct actuation matrix
            
            self.actuation_matrix[0:num_axes, i] = moment
            self.actuation_matrix[num_axes:num_axes + num_axes, i] = [force_x, force_y, force_z]
    
        return self.actuation_matrix


        
    def calculate_coil_currents(self):
        BFmatrix = np.array([self.total_field,
                    self.total_force]).reshape(6,)
        actuation_matrix = self.actuation_matrix
        inverse_actuation_matrix = np.linalg.pinv(actuation_matrix)
        self.coil_currents = np.dot(inverse_actuation_matrix, BFmatrix)
        
        return self.coil_currents

    
if __name__ == '__main__':
    calculator = MagneticFieldCalculator()
    
    
    
    
    
    
    
    
    
    