#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 03:38:03 2022

@author: kweonhyuckjin
"""
class arduinoset():
    
    def __init__(self,ard1):
        
        self.ard1 = ard1
        
    def arduino_set(self):
     
        import pyfirmata as pf
        from time import sleep
        import numpy as np
        

         # To avoid overflow between python and arduino
        it1 = pf.util.Iterator(self.ard1)
        it1.start()
        
        # brake pin set with mega1 and mega2
        brak1 = self.ard1.get_pin('d:22:o')
        brak1.write(0)
     

        for i in range(23, 27):

             self.ard1.digital[i].mode = pf.OUTPUT
             self.ard1.digital[i].write(0)


        for i in range(2, 6):
             self.ard1.digital[i].mode = pf.PWM
             self.ard1.digital[i].write(0)    #0~11

if __name__ == '__main__':
    print('Insert current')
    arduino_set()
