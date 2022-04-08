#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:57:04 2021

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  initialize_network.py
    *
    *  Desc: 
    *
    *  Written by:  
    *
    *  Latest Revision:  
**********************************************************************
"""
######################################################################
######################### Import Packages ############################
######################################################################

## Pytorch packages
import torch
import torch.nn as nn

######################################################################
##################### Function Definitions ###########################
######################################################################


## Set initial parameters 
def init(parameters):

    if (parameters.starting_parameters == '0'):
        
        ## Load model 
        
        if (parameters.model == 'resnet18'):
            model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
            
            # Freeze convolutional layers and only train FC layers
            for param in model.parameters():
                param.requires_grad = False
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, parameters.NUM_CLASSES)
            
        elif (parameters.model == 'vgg16'):
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            
            ## Freeze convolutional layers and only train FC layers
            for param in model.features.parameters():
                param.requires_grad = False
            
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, parameters.NUM_CLASSES)
        
    
    else:
        ## Initialize weights (Use this section if loading an initial iteration of weights)
        if (parameters.model == 'resnet18'):
            model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
            
            # Freeze convolutional layers and only train FC layers
            for param in model.parameters():
                param.requires_grad = False
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, parameters.NUM_CLASSES)
            
        elif (parameters.model == 'vgg16'):
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
            
            ## Freeze convolutional layers and only train FC layers
            for param in model.features.parameters():
                param.requires_grad = False
            
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, parameters.NUM_CLASSES)
    
        startingPoint = parameters.outf + parameters.starting_parameters # Load weights from pre-trained model
        pretrain_dict = torch.load(startingPoint) # Load weights from pre-trained model
        model_dict = model.state_dict() # Load existing parameters
        model_dict.update(pretrain_dict) # Overwrite encoder and decoder of existing model
        model.load_state_dict(model_dict) # Set weights
    
    return model

