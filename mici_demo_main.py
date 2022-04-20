#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 08:05:14 2022

@author: cmccurley
"""


"""
***********************************************************************
    *  File:  
    *  Name:  Connor H. McCurley
    *  Date:  
    *  Desc:  
    *
    *
    *  Author: Connor H. McCurley
    *  University of Florida, Electrical and Computer Engineering
    *  Email Address: cmccurley@ufl.edu
    *  Latest Revision: 
    *  This product is Copyright (c) 2022 University of Florida
    *  All rights reserved
**********************************************************************
"""


"""
%=====================================================================
%================= Import Settings and Packages ======================
%=====================================================================
"""

######################################################################
######################### Import Packages ############################
######################################################################
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

## General packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

## Custom packages
from cm_MICI.learnCIMeasureParams import set_parameters
from cm_MICI.util.cm_mi_choquet_integral import MIChoquetIntegral

"""
%=====================================================================
%============================ Main ===================================
%=====================================================================
"""

if __name__== "__main__":
    
    print('================= Running Main =================\n')
    
    ######################################################################
    ######################### Set Parameters  ############################
    ######################################################################
    args = None
    Parameters = set_parameters(args)
    
    ######################################################################
    ############################ Load Data ###############################
    ######################################################################
    
    ## Load the demo data
    data = io.loadmat('./cm_MICI/demo_data_cl.mat')
    Bags = data['Bags'][0,:]
    Labels = data['Labels'][0,:]
    gtrue = data['gtrue'][0,:]
    X = data['X']
    
    ######################################################################
    ########################## Training Stage ############################
    ######################################################################
    
    chi_genmean = MIChoquetIntegral()
    
    # Training Stage: Learn measures given training bags and labels
#    measure_genmean, initialMeasure_genmean, Analysis_genmean = chi_genmean.train_chi_softmax(Bags, Labels, Parameters) ## generalized-mean model
    chi_genmean.train_chi_softmax(Bags, Labels, Parameters) ## generalized-mean model
    
#    ######################################################################
#    ########################## Testing Stage #############################
#    ######################################################################
#    
#    ## Testing Stage: Given the learned measures above, compute fusion results
#    Ytrue = computeci(Bags,gtrue);  %true label
#    Yestimate_genmean = computeci(Bags,measure_genmean);  % learned measure by MICI generalized-mean model
#
#
#    ######################################################################
#    ############################## Plots #################################
#    ######################################################################
#
#    ## Plot true and estimated labels
#    fig, ax = plt.subplots(1,2)
#    ax[0].scatter(X[:,0],X[:,1],c=Ytrue)
#    ax[0].set_title('True Labels')
#    ax[1].scatter(X[:,0],X[:,1],c=Yestimate_genmean)
#    ax[1].set_title('Fusion result: MICI generalized-mean')
    
    
    
    
    