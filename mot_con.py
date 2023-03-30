

import pyfirmata as pf
from time import sleep
import numpy as np

class motordrive:

    def __init__(self,I,ard):
        self.I = np.array(I)
        self.ard = ard
   
        
    def Mcurrent(self):

#direction pins random set

        for i in range(23, 27):
            direction = np.zeros(4)

            if self.I[i-23][0] < 0:
                direction[i-23] = 1
            else:
                direction[i-23] = 0

            self.ard.digital[i].write(direction[i-23])


        # for i in range(25, 27):
        #     direction = np.zeros(4)

        #     if self.I[i-23][0] < 0:
        #         direction[i-23] = 0
        #     else:
        #         direction[i-23] = 1

        #   self.ard.digital[i].write(direction[i-23])

#pwm pins set and the value set range from 0 to 1

        for i in range(2,6):
            
            if abs(self.I[i-2]) > 1 :
                self.ard.digital[i].write(1)
                
            else:
                self.ard.digital[i].write(abs(self.I[i-2][0]))
                

            
if __name__ == '__main__':
    print('Insert current')
    motordrive()
