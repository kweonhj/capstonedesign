#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 12:54:35 2023

@author: kweonhyuckjin
"""

import numpy as np
from MagneticFieldCalculator import MagneticFieldCalculator


criteria_currents = np.array([1., 1., 1., 1., 1., 1., 1., 1.])

currents = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
position = np.array([0., 0., 0.])

calculator = MagneticFieldCalculator(currents)
B = calculator.calculate_magnetic_field(position)
F = calculator.calculate_magnetic_force(position)
B_matrix = calculator.B_field
B_matrix = np.array(B_matrix)
B_matrix = B_matrix.T
F_matrix = calculator.F_field
F_matrix = np.array(F_matrix)
F_matrix = F_matrix.T
print('B:')
print(B)
# print('F:')
# print(F)

calculator0 = MagneticFieldCalculator(criteria_currents)
B0 = calculator0.calculate_magnetic_field(position)
F0 = calculator0.calculate_magnetic_force(position)
B0_matrix = calculator0.B_field
B0_matrix = np.array(B0_matrix)
B0_matrix = B0_matrix.T
F0_matrix = calculator0.F_field
F0_matrix = np.array(F0_matrix)
F0_matrix = F0_matrix.T


delta = 1e-6
position2 = position + [delta, 0, 0]
calculator2 = MagneticFieldCalculator(criteria_currents)
B2 = calculator2.calculate_magnetic_field(position2)
B2_matrix = calculator2.B_field
B2_matrix = np.array(B2_matrix)
B2_matrix = B2_matrix.T
# print(B2)
# print(B2_matrix)

gradient_Bx = (B2_matrix - B0_matrix) / delta

position3 = position + [0, delta, 0]
calculator3 = MagneticFieldCalculator(criteria_currents)
B3 = calculator3.calculate_magnetic_field(position3)
B3_matrix = calculator3.B_field
B3_matrix = np.array(B3_matrix)
B3_matrix = B3_matrix.T
# print(B3)
# print(B3_matrix)

gradient_By = (B3_matrix - B0_matrix) / delta

position4 = position + [0, 0, delta]
calculator4 = MagneticFieldCalculator(criteria_currents)
B4 = calculator4.calculate_magnetic_field(position4)
B4_matrix = calculator4.B_field
B4_matrix = np.array(B4_matrix)
B4_matrix = B4_matrix.T
# print(B4)
# print(B4_matrix)

gradient_Bz = (B4_matrix - B0_matrix) / delta

m2 = calculator.m2_moment

F1 = np.dot(m2, gradient_Bx)
F2 = np.dot(m2, gradient_By)
F3 = np.dot(m2, gradient_Bz)
A = np.zeros((6,8))
A[0:3, :] = B0_matrix
A[3, :] = F1
A[4, :] = F2
A[5, :] = F3
A0 = np.zeros((6,8))
A0[0:3, :] = B0_matrix
A0[3:6, :] = F0_matrix

# print('act * I :')
# print(np.dot(A, currents))
# print('act0 * I :')
# print(np.dot(A0, currents))





























