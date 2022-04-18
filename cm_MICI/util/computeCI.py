#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:01:22 2022

@author: cmccurley
"""


"""
***********************************************************************
    *  File:  cnn_bag_classifier_parameters.py
    *  Name:  Connor H. McCurley
    *  Date:  2020-03-08
    *  Desc:  Handles all parameters for bag-level classifier based on ResNet18
    *
    *  Syntax: parameters = atr_parameters()
    *
    *  Outputs: parameters: dictionary holding parameters for atrMain.py
    *
    *
    *  Author: Connor H. McCurley
    *  University of Florida, Electrical and Computer Engineering
    *  Email Address: cmccurley@ufl.edu
    *  Latest Revision: March 8, 2022
    *  This product is Copyright (c) 2020 University of Florida
    *  All rights reserved
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

## Default packages
import argparse
import numpy as np
from tqdm import tqdm

## Pytorch packages
import torch
import torch.nn as nn

######################################################################
##################### Function Definitions ###########################
######################################################################

        
def print_status(epoch, epoch_loss, train_loader, valid_loader, model, device, logfilename):
    """
    ******************************************************************
        *  Func:    print_status()
        *  Desc:    Updates the terminal with training progress and writes to loss text file.
        *  Inputs:  Epoch #, train loss, valid loss, desired output file, time taken to complete epoch
        *  Outputs: -----
    ******************************************************************
    """
    ############################# Training #############################
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    print('Computing training loss...')
    with torch.no_grad():
        for data in tqdm(train_loader):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    training_accuracy = (100 * correct / total)
    
    ############################# Validation #############################
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    print('Computing validation loss...')
    with torch.no_grad():
        for data in tqdm(valid_loader):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    validation_accuracy = (100 * correct / total)
    
    print('Training accuracy: %d %%' % training_accuracy)  
    print('Validation accuracy: %d %%' % validation_accuracy)    
    
    line0 = '=================================='
    line1 = 'epoch ' + str(epoch) + ', train loss=' + str(round(training_accuracy,2))
    line2 = 'epoch ' + str(epoch) + ', valid loss=' + str(round(validation_accuracy,2))
#    line4 = '=================================='
    f = open(logfilename, 'a+') 
    f.write(line0 + '\n')
    f.write(line1 + '\n') 
    f.write(line2 + '\n')   
#    f.write(line4 + '\n')
    f.close()
    
    return