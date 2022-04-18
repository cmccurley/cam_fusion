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

from cm_MICI import [set_parameters, learnCIMeasure_softmax, 
                     computeCI, learnCIMeasure_softmax, evalFitness_softmax]

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
    Bags = data['Bags']
    Labels = data['Labels']
    gtrue = data['gtrue']
    X = data['X']
    
    ######################################################################
    ########################## Training Stage ############################
    ######################################################################
    
    ## Training Stage: Learn measures given training bags and labels
    measure_genmean, initialMeasure_genmean, Analysis_genmean = learnCIMeasure_softmax(Bags, Labels, Parameters) ## generalized-mean model
#    
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
#    plt.figure(101)
#    subplot(1,2,1);scatter(X(:,1),X(:,2),[],Ytrue);title('True Labels');
#    subplot(1,2,2);scatter(X(:,1),X(:,2),[],Yestimate_genmean);title('Fusion result: MICI noisy-or')
    
    
    
    
    