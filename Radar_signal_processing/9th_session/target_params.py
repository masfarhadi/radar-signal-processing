# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:38:55 2019

@author: wagner2
"""
import numpy as np

system_parameters={
    # system parameters
    'fs': 1e6,        # IF sample frequency
    'B': 250e6,       # RF bandwidth
    'fc': 24.125e9,   # carrier frequency
    'T': 1e-3,        # chirp duration
    'NF_dB': 12,      # equivalent noise temperature
    'T0': 290,        # system temperature
    'R': 50,          # reference impedance
    'NA': 8,          # number of ULA channels
    'T_dB': -160,     # threshold in dBV
    }

def get_target_params(t):
    """
    Calculates targets parameters for a given point in time.
    
    Parameters
    ----------
    t : float
        Time in seconds
        
    Returns
    -------
    Tuple containing target parameters as numpy arrays.
    
    A0_arr : 0D-array 
        holding magnitudes in V
    r0_arr : 0D-array 
        holding ranges in m
    theta0_arr : 0D-array 
        holding angles of incidend in rad
    """
    
    r0_lst=[0.001, 0.1, 15]
    A0_lst_uV=[20, 18, 0.1]
    theta0_lst_deg=[-3,5,-35]
    
    # add a target moving radially
    r0_lst.append(12+1*t)
    A0_lst_uV.append(15**4/r0_lst[-1]**4)
    theta0_lst_deg.append(31)
    
    # add a target moving in a cirlce
    r0_lst.append(40)
    A0_lst_uV.append(2)
    theta0_lst_deg.append(-25+2*t)
    

    A0_arr=np.array(A0_lst_uV)*1e-6
    r0_arr=np.array(r0_lst)
    theta0_arr=np.array(theta0_lst_deg)*np.pi/180
    
    return (A0_arr, r0_arr, theta0_arr)