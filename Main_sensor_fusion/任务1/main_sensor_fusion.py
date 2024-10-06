# -*- coding: utf-8 -*-
# ！/usr/bin/python3.9.7
# @Time   : 2022.4.18
# @Author : Yue Gaofeng
# @version: V2.0
# @Des    : NP_frame-based to learn optimal fusion and draw ROC curve. Muddle headed.

from time import *
import cal_fus
import numpy as np
import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    begin_time = time()                                  # Measure processing time of program
    N, epoch = 6, 10000                                  # Set sensor number and length of individual sequence
    M = 10                                               # Set length of measurement sequences
    Sigma = [1,1.1,1.2,1.3,1.4,1.5]                      # Noise variances of six sensors
    Yita=0.01                                            # Slice operation
    Pdset = np.linspace(0,1,int(1/Yita+1))               # False alarm probability（PF） constraint value
    Alpha = 0.3                                          # PF of local sensor 
    Chi_mtx = np.zeros([N,epoch,M])                      # Finish data initialization 

    Chi_mtx = cal_fus.gen(N,Chi_mtx,Sigma,epoch,M)       # Generate dataset
    print('Sequence fusion is working!!!')
    Z_mtx = cal_fus.fus_seq(Chi_mtx,Sigma,Pdset,M,\
                            epoch,N)                     # fusing sequences of same sensor
    Pd_single = cal_fus.cal(Z_mtx,Sigma,Pdset,N,epoch)   # Compute dection probability
    print('Mutiple-sonsor fusion is working!!!')
    pd_N,pf_N = cal_fus.fusion(Pdset,Alpha,Pd_single,N)  # N sensors' data fusion
    pd_N_2, pf_N_2 = cal_fus.fusion(Pdset,Alpha,\
                                Pd_single, int(N/2))     # N/2 sensors' data fusion
    plt.figure()                                         # Drawing ROC curve
    for i in range(N):                                   # Sensor 1-7 ROC curve
        colour = ''.join(random.sample('0123456789ABCDEF', 6))
        plot1 = plt.plot(Pdset,Pd_single[i,:],'#'+colour,linestyle='-.')        
    plot7 = plt.plot(pd_N_2,pf_N_2,'r','-')              # Three sensors fusion ROC curve'
    plot8 = plt.plot(pd_N,pf_N,'b','-')                  # Six sensors fusion ROC curve'
    plt.grid(linestyle='-.')
    plt.xlabel('False alarm probability', fontsize=14)
    plt.ylabel('Detection probability', fontsize=14)
    plt.title('ROC Curve Analysis', fontsize=14)
    plt.xlim([-0.02, 1.0])
    plt.ylim([-0.02, 1.05])
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0], fontsize=14)   # Set range of X-axis
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0], ['0.0','0.2',\
                '0.4','0.6','0.8','1.0'], fontsize=14)   # Set range of Y-axis
    plt.legend(['Sensor1','Sensor2','Sensor3','Sensor4','Sensor5',\
                   'Sensor6','Three sensors fusion','Six sensors fusion'],\
                       fontsize = 14, loc='best')
    end_time = time()
    run_time = end_time - begin_time
    print('Working is over! Running time:', run_time)
    #plt.savefig('Fig1.png',dpi = 600)                   # Save figure
    plt.show()
    









