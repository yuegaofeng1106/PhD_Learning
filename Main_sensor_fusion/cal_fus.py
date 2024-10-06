# -*- coding: utf-8 -*-
# ！/usr/bin/python3
# @Time   : 2022.3.18
# @Author : Yue Gaofeng
# @version: V2.0
# @Des    : Generate raw data and calculate detection probability of simple sensor
#         : Mutiple-sensor fusion. So hard!!!

import itertools
from math import sqrt
from re import A
from turtle import shape
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
# 比较大小函数

def gen(N, Chi_mtx, Sigma, epoch, M):                      # Initialization and generate data    
    for i in range(N): 
        for j in range(epoch):              
            A, B = 2, sqrt(Sigma[i])                      # A is amplitude and B is standard deviation
            Chi_mtx[i,j,:] = np.random.normal(A, B, M)    # Obtaining measurement matrix 
    return Chi_mtx

def cal(Z_mtx, Sigma, Pdset, N, epoch):               # Find the PD of mutiple-sensors given PF
    L = len(Pdset); Gamma = np.zeros([N,L])
    pd_single = np.zeros([N, L], dtype=float)         # Number of one (0-1 distribution)
    for i in tqdm(range(N)):
        for j in range(L):
            Gamma[i, j] = norm.ppf(1-Pdset[j], 0, sqrt(Sigma[i])) # Inverse of CDF and get PD given PF
            s = 0.0  
            for k in range(epoch):
                if Z_mtx[i,k] > Gamma[i,j]: s+=1.0   # Likelihood ratio judgment, caculative sum
                else: s+=0.0
                pd_single[i, j] = s/(k+1)            # Calculate                          
    return pd_single

def No_zero_value(value_list, logical_list):  # Saving non-zero elements but delete corresponding to 0
    L = len(value_list)
    back_list = []
    for i in range(0, L):
        if logical_list[i] == 1: back_list.append(value_list[i])  # Save where index is true
    return back_list

def fus_seq(Chi_mtx,Sigma,Pdset,M,epoch,N):     # fusing sequences of same sensor
    L = len(Pdset)
    Gamma = np.zeros([1,L])
    pd_m = np.zeros([N, epoch], dtype=float)

    for i in tqdm(range(N)):
        Chi_mtx = np.resize(Chi_mtx[i],(epoch,M))
        for j in range(L):
            Gamma[0, j] = norm.ppf(1-Pdset[j], 0, sqrt(Sigma[i])) # Inverse of CDF and get PD given PF
            for m in range(M):
                s = 0.0
                for k in range(epoch):
                    if Chi_mtx[k,m] > Gamma[0,j]: s+=1.0   # Likelihood ratio judgment, caculative sum
                    else: s+=0.0
                    pd_m[i,k] = s/epoch                   # Calculate
    return pd_m

def fusion(Pdset,Alpha,Pd_single,N):        # Diffrent sensor fusion
    L = len(Pdset)                          # Initialize local variables , L = 101
    Pro_if = np.ones([3,N], dtype=float)
    pd_tmp = np.ones([1,N], dtype=float)
    delta = np.ones([1,2**N], dtype=float)
    p01_newp01 = np.ones([4,2**N], dtype=float)
    
    Arrange = np.array([i for i in itertools.product([0, 1], repeat = N)])   # Arrange combinations
    for i in range(2**N):                                 # Deal various arrange
        for j in range(N):                                #To address N sensors
            No_zero = ((Pdset[0:L] == Alpha).astype(int)).tolist()  # Logical metrix 
            pd_tmp[0,j] = Pd_single[j, No_zero.index(max(No_zero))] # Get index of max value
            if  Arrange[i,j] == 0:                        # Conditional probability(Con_pro)
                Pro_if[0,j] = (1-pd_tmp[0,j])/(1-Alpha)   # Miss probability: beta/(1-alpha)
                Pro_if[1,j] =  1-Alpha                    # H0: Con_pro of single sensor likelihood ratio
                Pro_if[2,j] = 1-pd_tmp[0,j]               # H1: Con_pro of single sensor likelihood ratio
            else:
                Pro_if[0,j] = pd_tmp[0,j]/Alpha
                Pro_if[1,j] = Alpha
                Pro_if[2,j] = pd_tmp[0,j]

            delta[0,i]  = np.prod(Pro_if[0,:])            # Multi-sensor union likelihood ratio
            p01_newp01[0,i]  = np.prod(Pro_if[1,:])       # H0: Con_pro of Multi-sensor union likelihood ratio
            p01_newp01[1,i]  = np.prod(Pro_if[2,:])       # H1: Con_pro of Multi-sensor union likelihood ratio

    idx = delta.argsort()                                 # Sored from max to min to likelihood
    p01_newp01[2] = p01_newp01[0,idx[0,:]]                # Re-ordering
    p01_newp01[3] = p01_newp01[1,idx[0,:]]

    b2_c2 = np.zeros([2, 2**N], dtype=float)
    pf_pd = np.zeros([2,L], dtype=float)

    b2_c2[0,:] = np.cumsum(p01_newp01[2,:])             # Cumulative sum to form novel array 
    b2_c2[1,:] = np.cumsum(p01_newp01[3,:])             # Cumulative sum by element in row
    b2_new = [0]+list(b2_c2[0,:])+[1] 
    c2_new = [0]+list(b2_c2[1,:])+[1]

    for i in range(L):
        metric = ((b2_new <= Pdset[i]).astype(int))            # Logical Matrix
        back_list = No_zero_value(b2_new, metric)              # Delete the column corresponding to 0
        pf_pd[0,i], id = max(back_list), np.argmax(back_list)  # Returns index and maximum
        pf_pd[1,i] = c2_new[id]                                # Returns pd sorted by idex list
    
    return pf_pd[1,:].tolist(),pf_pd[0,:].tolist()





