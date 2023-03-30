# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:08:33 2022

@author: ehdqk
"""
import numpy as np

class capst:
      
    def __init__(self,rand_loc,b_vector,f_vector):
        self.rand_loc=np.array(rand_loc)
        self.b_vector=np.array(b_vector)
        self.f_vector=np.array(f_vector)
        
    def actuation(rand_loc,b_vector,f_vector): 
        perma = 4*np.pi*10**(-7); #유전율 값 
        
        coil_A =[-0.015,0.015]; coil_B = [0.02,-0.015];  coil_C = [0.055,0.015]; coil_D = [0.02,0.045]
        #coil의 중심위치를 지정
        maga = 0.;  magb= 0.045;  magc = 0.; magd = 0.045; # mage = 0.87; magf = 0.87; magg = 0.87; magh = 0.87
        Mag=[maga,magb,magc,magd] #,mage,magf,magg,magh]
        # 자기모멘트과 전류값을 입력(8개 전자석)
        
        vect_a = [1,0]; vect_b = [0,1]; vect_c = [1,0]; vect_d = [0,1];
        magnetic_vect=np.array([vect_a,vect_b,vect_c,vect_d]).T #,vect_e,vect_f,vect_g,vect_h]).T
        # n개 코일의 자기모멘트 벡터 방향을 나타냄(normal vector)
        
        coil = np.array([coil_A,coil_B,coil_C,coil_D]) # coil_E,coil_F,coil_G,coil_H]) # shape(4,2) n,3 n=4
        n=len(coil) #n=코일의 개수
        m=len(coil_A) # mxm배열 2차원 m=2 3차원 m=3
        
        bf_vector=np.concatenate((b_vector,f_vector),axis=0)
        unit_vect=[]
        mag_mom=[]
        r_vector=[]
        r_unit=[]
        mag_mom_R=(b_vector/np.linalg.norm(b_vector))*0.017 # 로봇의 자기모멘트 벡터 방향 (normal vector) , 직접계산해서 수치 넣기 0.XXX
        
        for i in range(n):
            unit_vect.append(magnetic_vect[:,i]/np.linalg.norm(magnetic_vect[:,i])) #전자석의 자기모멘트 의 단위벡터를 만들어줌
            mag_mom.append(unit_vect[i] * Mag[i]) #전자석의 자기모멘트값(방향*면적)
            r_vector.append(rand_loc-coil[i,:])
            r_unit.append(r_vector[i]/np.linalg.norm(r_vector[i]))
            
        unit_vect=np.array(unit_vect).T # 코일의 벡터 값 shape(3,n)
        mag_mom=np.array(mag_mom).T # 코일의 자기 모멘트 값 shape(3,n)
        r_vector = np.array(r_vector).T # 자석의 벡터 값 shape(3,n)
        r_unit = np.array(r_unit).T # 자석의 단위벡터 값 shape(3,n)
    
        B=np.zeros((m,m*n))
        F=np.zeros((m,m*n))
        MAG=np.zeros((m*n,n))
        
        for i in range(n):
            B[:,m*(i):m*i+m]=perma/(4*np.pi*(np.linalg.norm(r_vector[:,i]))**(3))*\
                                 (3*np.dot(r_unit[:,i],r_unit[:,i].T)-np.eye(m))
            #자기장값(magnetic moment를 안 곱한 값) 
            F[:,m*(i):m*i+m]=(3*perma/(4*np.pi*(np.linalg.norm(r_vector[:,i]))**(4))*
                              (np.dot(mag_mom_R,r_unit[:,i].reshape(m,1).T)+np.dot(r_unit[:,i].reshape(m,1),mag_mom_R.T)+
                               np.dot(r_unit[:,i].reshape(m,1).T,mag_mom_R)*(np.eye(m)-5*np.dot(r_unit[:,i],r_unit[:,i].T))))
            #힘값 (magnetic moment를 안 곱한 값)
            MAG[m*(i):m*i+m,i] = mag_mom[:,i]   #magnetic moment shape(3n,n)  
            
        BF_Matrix=np.concatenate((B,F),axis=0)
        act_M=np.dot(BF_Matrix,MAG)
        pseudo_M=np.linalg.pinv(act_M)
        # np.dot(np.linalg.inv(np.dot(act_M.T,act_M)),act_M.T)
        pseudo_M.shape
        Cur_vect=np.dot(pseudo_M,bf_vector)
        print(Cur_vect)
        return Cur_vect

if __name__=="__main__":
    capst.actuation()        




##처음부터 작은 값부터 올리기 kp, 혁진이형이랑 값 맞추기, 종서랑 calibration,
 