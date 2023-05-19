#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:01:22 2023

@author: kweonhyuckjin
"""

import numpy as np
from MagneticFieldCalculator import MagneticFieldCalculator

currents = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
m2_position = np.array([0,0,0])
# Create an instance of MagneticFieldCalculator
magnetic_field_calculator = MagneticFieldCalculator(currents)

# Perform calculations
B = magnetic_field_calculator.calculate_magnetic_field(m2_position)
F = magnetic_field_calculator.calculate_magnetic_force(m2_position)
# Assuming you have an instance of the MagneticFieldCalculator class called 'calculator'
actuation_matrix = magnetic_field_calculator.calculate_actuation_matrix(m2_position)
cal_current = magnetic_field_calculator.calculate_coil_currents()
# Print the actuation matrix
print("Actuation Matrix:")
print(actuation_matrix)
print('Current:')
print(cal_current)

