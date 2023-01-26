# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:29:16 2020

@author: 47227
"""
import numpy as np
import os

#import matplotlib.pyplot as plt

import random
import tifffile
import h5py

from MySim import SimIP
def simulation(alpha_1 = -9.5,
            beta_1 = -1.5,
            gamma_1 = 20.0,
            dgamma_1 = 40.0,
            scale_1 = 4000.0, 
              alpha_2 = 86.0,
            beta_2 = -3.0,
            gamma_2 = 80.0,
            dgamma_2 = 10.0,
            scale_2 = 2000.0):
     

            labels=[]
            I_110_save=[]
           
            mag_factor=400
            xcenter_shift=0
            ycenter_shift=0

            q_110=0.1925
            
            #################################110########################################
            I_tmp_110_ip=SimIP([1,1,0],alpha=alpha_1,beta=beta_1,gamma0=gamma_1 ,dgamma0=dgamma_1,
                        scale=scale_1*15,q_hkl=q_110,
                        mag_factor=mag_factor,xcenter_shift=xcenter_shift,ycenter_shift=ycenter_shift,gaussian_sigma=4)

            I_110_ip,chi_110_ip=I_tmp_110_ip.Delta_mu_Weight()
            

            I_tmp_110_op=SimIP([1,1,0],alpha=alpha_2,beta=beta_2,gamma0=gamma_2 ,dgamma0=dgamma_2,
                        scale=scale_2*15,q_hkl=q_110,
                        mag_factor=mag_factor,xcenter_shift=xcenter_shift,ycenter_shift=ycenter_shift,gaussian_sigma=4)

            I_110_op,chi_110_op=I_tmp_110_op.Delta_mu_Weight()


           
            I_110_save.append(I_110_ip+I_110_op)


            tmp=[]
            tmp.append(alpha_1)
            tmp.append(beta_1)
            tmp.append(gamma_1)
            tmp.append(dgamma_1) 
            tmp.append(scale_1)

            tmp.append(alpha_2)
            tmp.append(beta_2)
            tmp.append(gamma_2)
            tmp.append(dgamma_2) 
            tmp.append(scale_2)

            labels.append(tmp)
         
            return np.array(I_110_save), np.array(labels)