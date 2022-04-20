#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:03:16 2022

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

        

def evalFitness_softmax(Labels, measure, nPntsBags, oneV, bag_row_ids, diffM, p):
    
    """
    ===========================================================================
    % Evaluate the fitness a measure, similar to evalFitness_minmax() for
    % classification but uses generalized mean (sometimes also named "softmax") model
    %
    % INPUT
    %    Labels         - 1xNumTrainBags double  - Training labels for each bag
    %    measure        -  measure to be evaluated after update
    %    nPntsBags      - 1xNumTrainBags double    - number of points in each bag
    %    bag_row_ids    - the row indices of measure used for each bag
    %    diffM          - Precompute differences for each bag
    %
    % OUTPUT
    %   fitness         - the fitness value using min(sum(min((ci-d)^2))) for regression.
    %
    % Written by: X. Du 03/2018
    %
    ===========================================================================
    """
    
    p1 = p[0]
    p2 = p[1]
    
    singletonLoc = np.where(nPntsBags == 1)
    nonsinletonLoc = np.where(not(nPntsBags == 1))
    
    diffM_ns = diffM(nonsinletonLoc)
    bag_row_ids_ns = bag_row_ids(nonsinletonLoc)
    labels_ns = Labels(nonsinletonLoc)
    oneV = oneV(nonsinletonLoc)
    
    fitness = 0
    
    if sum(singletonLoc):  ## if there are singleton bags
        diffM_s = vertcat(diffM{singletonLoc})
        bag_row_ids_s = vertcat(bag_row_ids{singletonLoc})
        labels_s = Labels(singletonLoc)
        
        ## Compute CI for singleton bags
        ci = sum(diffM_s.*horzcat(measure(bag_row_ids_s),ones(sum(singletonLoc),1)),2)
        fitness_s = sum((ci - labels_s)**2)
        fitness = fitness - fitness_s   
   
    
    ## Compute CI for non-singleton bags
    for i = 1:length(diffM_ns):
        ci = sum(diffM_ns{i}.*horzcat(measure(bag_row_ids_ns{i}), oneV{i}),2)
        if(labels_ns(i) ~= 1) %negative bag label=0
        fitness = fitness - (mean(ci.^(2*p1))).^(1/p1);
         else
        fitness = fitness - (mean((ci-1).^(2*p2))).^(1/p2);

    
       ## Sanity check
        if (np.isinf(fitness) || not(np.isreal(fitness))):
            fitness = real(fitness)
            if (fitness == Inf):
                fitness = -10000000
            elif (fitness == -Inf):
                fitness = -10000000

    return fitness
