#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 21:32:59 2023

@author: kweonhyuckjin
"""
import pyfirmata as pf
import numpy as np

class ArduinoSetting:
    
    def __init__(self,ard):

        self.ard = ard
        
    def arduino_setting(self):
   
         # To avoid overflow between python and arduino
        it1 = pf.util.Iterator(self.ard)
        it1.start()
        
        # brake pin set with mega1 and mega2
        brak1 = self.ard.get_pin('d:22:o')
        brak1.write(0)
     
   
        for i in range(23, 31):
   
             self.ard.digital[i].mode = pf.OUTPUT
             self.ard.digital[i].write(0)
   
   
        for i in range(2, 10):
             self.ard.digital[i].mode = pf.PWM
             self.ard.digital[i].write(0)   
        
        return self.ard
             
    def arduino_initalize(self):
        
        for i in range(2, 12):
            
            self.ard.digital[i].write(0)
            self.ard.digital[i + 21].write(0)
            
    def  MotordriveControl(self, I):
        
        for i in range(23, 31):
            direction = np.zeros(8)

            if I[i-23] < 0:
                direction[i-23] = 1
            else:
                direction[i-23] = 0

            self.ard.digital[i].write(direction[i-23])


        for i in range(2,10):
            
            if abs(I[i-2]) > 1 :
                
                self.ard.digital[i].write(1)
                
            else:
                
                self.ard.digital[i].write(abs(I[i-2]))
                
        # print(I)
        
        