#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:15:34 2022

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  set_mici_parameters.py
    *  Name:  Connor H. McCurley
    *  Date:  04/29/2022
    *  Desc:  Implementation of the Multiple Instance Choquet Integral defined
    *         by X. Du and A. Zare, "Multiple Instance Choquet Integral Classifier
    *         Fusion and Regression for Remote Sensing Applications," in IEEE
    *         Trans. on Geoscience and Remote Sensing (TGRS), vol 57, no 5, 2019.
    *
    *         This code defines the parameters for the MICI.
    *
    *
    *  Author: Connor H. McCurley
    *  University of Florida, Electrical and Computer Engineering
    *  Email Address: cmccurley@ufl.edu
    *  Latest Revision: 04/29/2022
    *  This product is Copyright (c) 2022 University of Florida
    *  All rights reserved
**********************************************************************
"""
    

import argparse

#######################################################################
########################## Define Parameters ##########################
#######################################################################
def set_parameters(args):
    """
    ******************************************************************
        *
        *  Func:    set_parameters()
        *
        *  Parameters - Argparser - The parser contains the following fields:
        *               These parameters are user defined (can be modified)
        *
        *           1. nPop: Size of population
        *           2. sigma: Sigma of Gaussians in fitness function
        *           3. nIterations: Number of iterations
        *           4. eta:Percentage of time to make small-scale mutation
        *           5. sampleVar: Variance around sample mean
        *           6. mean: mean of ci in fitness function. Is always set to 1 if the positive label is "1".
        *           7. analysis: if ="1", record all intermediate results
        *           8. p: the power coefficient for the generalized-mean function. Empirically, setting p(1) to a large postive number (e.g., 10) and p(2) to a large negative number (e.g., -10) works well.
        *           9. use_parallel: Flag for using parallel processing in measure sampling
        *
    ******************************************************************
    """  
    
    ######################################################################
    ############################ Define Parser ###########################
    ######################################################################
    
    parser = argparse.ArgumentParser(description='Parameters for the Multiple Instance Choquet Integral.')
    
    ######################################################################
    ######################### Input Parameters ###########################
    ######################################################################
    
    parser.add_argument('--nPop', help='Size of population, preferably even numbers.', default=100, type=int)
    parser.add_argument('--sigma', help='Sigma of Gaussians in fitness function.', default=0.1, type=float)
    parser.add_argument('--maxIterations', help='Max number of iteration.', default=5000, type=int)
    parser.add_argument('--fitnessThresh', help='Stopping criteria: when fitness change is less than the threshold.', default=0.0001, type=float)
    parser.add_argument('--eta', help='Percentage of time to make small-scale mutation.', default=0.8, type=float)
    parser.add_argument('--sampleVar', help='Variance around sample mean.', default=0.1, type=float)
    parser.add_argument('--mean', help='Mean of ci in fitness function. Is always set to 1 if the positive label is "1".', default=1.0, type=float)
    parser.add_argument('--analysis', action='store_true', help='If ="1", record all intermediate results.', default=False)
    parser.add_argument('--p', help='p power value for the softmax. p(1) should->Inf, p(2) should->-Inf.', nargs='+', default=[10,-10])
    parser.add_argument('--use_parallel', action='store_true', help='If ="1", use parallel processing for population sampling.', default=True)

    return parser.parse_args(args)



                  

