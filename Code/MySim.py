# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:03:11 2019

@author: dongzheng
"""

import numpy as np

from scipy import integrate

from scipy import interpolate



class SimIP():

    #int
    def __init__(self,hkl,alpha,beta,gamma0 ,dgamma0,scale,q_hkl,
                 mag_factor,xcenter_shift,ycenter_shift,gaussian_sigma):

        self.Alpha_1 = alpha
        self.Beta_1 = beta
        self.gamma0_1= gamma0
        self.dgamma0_1 = dgamma0
        self.scale_1 = scale
        self.mag_factor=mag_factor
        self.xcenter_shift=xcenter_shift
        self.ycenter_shift=ycenter_shift
        self.gaussian_sigma=gaussian_sigma
    
        self.Wavelength=0.8856
        self.q_ES=1#2*np.pi/self.Wavelength
        self.cell_a=4.74
        self.cell_b=10.32
        self.cell_c=18.86
    
        self.rAlpha_1 = np.deg2rad(self.Alpha_1)
        self.rBeta_1 = np.deg2rad(self.Beta_1)
        self.rdgamma0_1 = np.deg2rad(self.dgamma0_1)
        self.rgamma0_1= np.deg2rad(self.gamma0_1)
        self.q_hkl=q_hkl#np.sqrt( (self.hkl_h/self.cell_a)**2+(self.hkl_k/self.cell_b)**2+(self.hkl_l/self.cell_c)**2  )
        
        #self.chi_num=int(self.q_hkl*self.mag_factor*2*np.sqrt(2)*np.pi)
        self.chi_num=360#q_hkl*10000
        delta_chi=(2*np.pi)/self.chi_num
        
        self.chi=np.arange(0,2*np.pi,delta_chi)
    
        self.hkl=hkl
        self.hkl_h=hkl[0]
        self.hkl_k=hkl[1]
        self.hkl_l=hkl[2]

        
        
        qx_L=np.ones(len(self.chi))*(-self.q_hkl**2/(2*self.q_ES))
    
        tmp=self.q_hkl*np.sqrt(1-((self.q_hkl)/(2*self.q_ES))**2)
    
        qy_L=tmp*np.sin(self.chi)
        qz_L=tmp*np.cos(self.chi)
    
        Q_L= np.concatenate((np.array(qx_L)[:,np.newaxis],np.array(qy_L)[:,np.newaxis],np.array(qz_L)[:,np.newaxis]),axis=1)
    
    
        L2B = np.array([[np.cos(self.rAlpha_1),                     np.sin(self.rAlpha_1),0],
                     [-np.sin(self.rAlpha_1)*np.cos(self.rBeta_1),   np.cos(self.rAlpha_1)*np.cos(self.rBeta_1),  np.sin(self.rBeta_1)],
                     [ np.sin(self.rAlpha_1)*np.sin(self.rBeta_1),  -np.cos(self.rAlpha_1)*np.sin(self.rBeta_1),  np.cos(self.rBeta_1) ]])
    
    
    
    
        Q_B=np.dot(L2B,Q_L.T)
    
        self.qx_b=Q_B[0,:]
        self.qy_b=Q_B[1,:]
        self.qz_b=Q_B[2,:]
    
###############

    def lab(self):
        qx_L=np.ones(len(self.chi))*(-self.q_hkl**2/(2*self.q_ES))
    
        tmp=self.q_hkl*np.sqrt(1-((self.q_hkl)/(2*self.q_ES))**2)
    
        qy_L=tmp*np.sin(self.chi)
        qz_L=tmp*np.cos(self.chi)
        
        return qx_L,qy_L,qz_L
    
    def q_b(self):
        qx_b=self.qx_b
        qy_b=self.qy_b
        qz_b=self.qz_b
        
        return qx_b,qy_b,qz_b

###############


    def Delta_mu_Weight(self):
        
        w_mu=self.q_hkl/10
        
        if self.hkl_h !=0 and self.hkl_k != 0  and self.hkl_l ==0:
            
            mu_1=np.pi/2
        elif self.hkl_h ==0 and self.hkl_k ==0 and self.hkl_l !=0:
            mu_1=0
        elif self.hkl_h ==0 and self.hkl_k ==1 and self.hkl_l ==3:
            mu_1=30
        else:
            tmp=(self.hkl_h/self.cell_a)**2/( (self.hkl_k/self.cell_b)**2 + (self.hkl_l/self.cell_c)**2 )
            mu_1=np.arctan(np.sqrt(tmp))
        
        mu_1=np.pi/2
        tmp_I=[]
        for i in  range(0,len(self.chi)):
            
            chi_tmp=self.chi[i]
            qx_b_tmp=self.qx_b[i]
            qy_b_tmp=self.qy_b[i]
            qz_b_tmp=self.qz_b[i]
            
            
            def delta_mu_weight(rgamma_1):
                
                weight=np.exp(-(1/2)*((rgamma_1-self.rgamma0_1)/self.rdgamma0_1)**2)/(2*np.pi*self.rdgamma0_1* w_mu )
                
                '''
                #general
                tmp_mu=np.arctan( np.sqrt( (  self.q_hkl/(qx_b_tmp*np.cos(gamma_1) +qz_b_tmp* np.sin(gamma_1) ))**2-1)   )
                
                delta_mu_weight_without_w=np.exp(-(tmp_mu-mu_1)**2/(2*w_mu**2))
                '''
                #zhang
                
                #110
                if self.hkl==[1,1,0]:
                    delta_mu_weight_without_w=np.exp(-(1/2)*(( qx_b_tmp*np.cos(rgamma_1)-qz_b_tmp*np.sin(rgamma_1)  )/w_mu )**2)
                elif self.hkl==[0,0,2]:
                    delta_mu_weight_without_w=np.exp(-((qz_b_tmp**2)/(2*(w_mu**2))))/np.sqrt(self.q_hkl**2-qz_b_tmp**2)
                elif self.hkl==[0,1,3]:
                    mu_013 = 10.337
                    mu_013  = np.deg2rad(mu_013)
 
                    delta_mu_weight_without_w=\
                    np.exp(-(((qy_b_tmp*np.cos(rgamma_1)-qx_b_tmp*np.sin(rgamma_1))-self.q_hkl*np.cos(mu_013))**2/(w_mu**2)))+\
                    np.exp(-(((qy_b_tmp*np.cos(rgamma_1)-qx_b_tmp*np.sin(rgamma_1))+self.q_hkl*np.cos(mu_013))**2/(w_mu**2))) 
                    

                return weight*delta_mu_weight_without_w
            
            #####integrate from -pi/2  pi/2
            
            tmp, err = integrate.quad(delta_mu_weight, -np.pi/2 , np.pi/2)
            
            tmp_I.append(tmp)
            #print(tmp)
            self.I_1D=np.array(tmp_I)*self.scale_1 
        return  self.I_1D , self.chi
    
    def To_2D(self):
        self.I_2D=np.zeros((256,256))
        
        q=self.q_hkl*self.mag_factor
   
        x=[]
        y=[]
        z=[]

        
        for tmp in range(0,len(self.chi)):            
            x.append(q*np.cos(self.chi[tmp])+127+self.xcenter_shift)
            y.append(q*np.sin(self.chi[tmp])+127+self.ycenter_shift)
            z.append(self.I_1D[tmp])
            
        
        x_new=np.around(x).astype(int)
        y_new=np.around(y).astype(int)
        
        newfunc = interpolate.interp2d(x, y, z, kind='linear')
           
         
        for n_tmp in range(0,len(x_new)):
                #tmp=newfunc(i, j)
                #tmp=tmp[0]
                i=x_new[n_tmp]
                j=y_new[n_tmp]
                self.I_2D[i,j]=newfunc(i,j)[0]
                
                
                for di in range(-2,2):
                    for dj in range(-2,2):
                        ii=i-di
                        jj=j-dj
                        tmp=newfunc(ii, jj)[0]
                        
                        self.I_2D[ii,jj]=tmp 
                
                
        return self.I_2D
    
    
    
  
    def gaussian_filter(self):
        tmp=self.I_2D.shape
        g_sigma=self.gaussian_sigma#*int((mag_factor/400))
        g_size=int(2*g_sigma+1)
        
        
        g_core=np.zeros((g_size*2+1,g_size*2+1))
        for gi in range(-g_size,g_size+1):
            for gj  in range (-g_size,g_size+1):
        
                 g_core[gi+g_size,gj+g_size]=np.exp(-(gi**2+gj**2)/(2*g_sigma**2))/(2*np.pi*g_sigma**2)
        
        I_all_g2=np.zeros(tmp)
        
        for gx in range(g_size+1,tmp[0]-g_size-1):
            for gy in range(g_size+1,tmp[0]-g_size-1):
                s_data=self.I_2D[gx-g_size:gx+g_size+1,gy-g_size:gy+g_size+1]
                I_all_g2[gx,gy]=np.sum(np.multiply(s_data,g_core))
        
        
        return I_all_g2  
    



            




