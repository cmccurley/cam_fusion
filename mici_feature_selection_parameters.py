#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:39:40 2022

@author: cmccurley
"""


"""
***********************************************************************
    *  File:  
    *  Name:  Connor H. McCurley
    *  Date:  
    *  Desc:  
    *
    *  Syntax: parameters = atr_parameters()
    *
    *  Outputs: parameters: dictionary holding parameters for atrMain.py
    *
    *
    *  Author: Connor H. McCurley
    *  University of Florida, Electrical and Computer Engineering
    *  Email Address: cmccurley@ufl.edu
    *  Latest Revision: 
    *  This product is Copyright (c) 2020 University of Florida
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
        *  Desc:    Class definition for the configuration object.  
        *           This object defines that parameters for the MIL U-Net
        *           script.
        *
    ******************************************************************
    """  
    
    ######################################################################
    ############################ Define Parser ###########################
    ######################################################################
    
    parser = argparse.ArgumentParser(description='Training script for bag-level classifier.')
    
    parser.add_argument('--run_mode', help='Mode to train, test, or compute CAMS. (cam)', default='test', type=str)
    
    parser.add_argument('--experiment_name', help='Mode to train, test, or compute CAMS. (cam)', default='_verification_neg_source_in_pos_bag_mici', type=str)
    
    ######################################################################
    ######################### Input Parameters ###########################
    ######################################################################
    
    parser.add_argument('--DATASET', help='Dataset selection.', default='mnist', type=str)
    parser.add_argument('--DATABASENAME', help='Relative path to the training data list.', default='', type=str)
 
    parser.add_argument('--target_classes', help='List of target classes.', nargs='+', default=['aeroplane'])
    parser.add_argument('--bg_classes', help='List of background classes.', nargs='+', default=['cat'])
    parser.add_argument('--layers', help='Layers to compute CAM at.', nargs='+', default=[4,9,16,23,30])
#    parser.add_argument('--CAM_SEG_THRESH', help='Hard threshold to convert CAM to segmentation decision.', nargs='+', default=[0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9])
    
    parser.add_argument('--NUM_SOURCES', help='Input batch size for training.', default=3, type=int)
    parser.add_argument('--INCLUDE_INVERTED', action='store_true', help='If true, include inverted feature maps in selection.', default=False)
    
    ######################################################################
    ##################### Training Hyper-parameters ######################
    ######################################################################
    
    ## Starting parameters
#    parser.add_argument('--starting_parameters', help='Initial parameters when training from scratch. (0 uses pre-training on ImageNet.)', default='0', type=str)
    parser.add_argument('--starting_parameters', help='Initial parameters when training from scratch. (0 uses pre-training on ImageNet.)', default='/model_eoe_80.pth', type=str)
    parser.add_argument('--model', help='Neural network model', default='vgg16', type=str)
    
#    ## Hyperparameters
#    parser.add_argument('--NUM_CLASSES', help='Number of classes in the data set.', default=2, type=int)
    parser.add_argument('--BATCH_SIZE', help='Input batch size for training.', default=1, type=int)
#    parser.add_argument('--EPOCHS', help='Number of epochs to train.', default=100, type=int)
#    parser.add_argument('--LR', help='Learning rate.', default=0.001, type=float)
#    
#    ## Updating  
#    parser.add_argument('--update_on_epoch', help='Number of epochs to update loss script.', default=5, type=int)
    parser.add_argument('--parameter_path', help='Where to save/load weights after each epoch of training', default='/trainTemp.pth', type=str)
#    
    ## PyTorch/GPU parameters
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA training.', default=False)
    parser.add_argument('--cuda', help='Enable CUDA training.', default=True)
    
    ######################################################################
    ######################### Output Parameters ##########################
    ######################################################################
    
    ## Output parameters
    parser.add_argument('--loss_file', help='Where to save training and validation loss updates.', default='/loss.txt', type=str)
    parser.add_argument('--outf', help='Where to save output files.', default='./mnist/vgg16/output', type=str)
    parser.add_argument('--NUMWORKERS', help='', default=0, type=int)
    
    
    parser.add_argument('--TMPDATAPATH', help='', default='', type=str)
    
    ######################################################################
    ########################## MICI Parameters ###########################
    ######################################################################
    
    ## MICI parameters
    parser.add_argument('--nPop', help='Size of population, preferably even numbers.', default=100, type=int)
    parser.add_argument('--sigma', help='Sigma of Gaussians in fitness function.', default=0.1, type=float)
    parser.add_argument('--maxIterations', help='Max number of iteration.', default=300, type=int)
    parser.add_argument('--fitnessThresh', help='Stopping criteria: when fitness change is less than the threshold.', default=0.0001, type=float)
    parser.add_argument('--eta', help='Percentage of time to make small-scale mutation.', default=0.5, type=float)
    parser.add_argument('--sampleVar', help='Variance around sample mean.', default=0.1, type=float)
    parser.add_argument('--mean', help='Mean of ci in fitness function. Is always set to 1 if the positive label is "1".', default=1.0, type=float)
    parser.add_argument('--analysis', action='store_true', help='If ="1", record all intermediate results.', default=False)
    parser.add_argument('--p', help='p power value for the softmax. p(1) should->Inf, p(2) should->-Inf.', nargs='+', default=[10,-10])
    parser.add_argument('--use_parallel', action='store_true', help='If ="1", use parallel processing for population sampling.', default=True)
    
    ## Parameters for binary fuzzy measure sampling
    parser.add_argument('--U', help='How many times we are willing to sample a new measure before deeming the search exhaustive.', default=500, type=int)
    parser.add_argument('--Q', help='How many times the best fitness value can remain unchanged before stopping.', default=3, type=int)
    
    return parser.parse_args(args)
    
    
