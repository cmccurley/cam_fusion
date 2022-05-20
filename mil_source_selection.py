#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:10:53 2022

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

## General packages
import os
import json
import copy
from copy import deepcopy
import numpy as np
import itertools
from tqdm import tqdm
import dask.array as da

## Custom packages
from mici_feature_selection_parameters import set_parameters as set_parameters
from cm_MICI.util.cm_mi_choquet_integral_binary_dask import BinaryMIChoquetIntegral
from cm_MICI.util.cm_mi_choquet_integral_dask import MIChoquetIntegral

"""
%=====================================================================
%======================= Function Definitions ========================
%=====================================================================
"""

###############################################################################
############################# Dask MIL Selection ##############################
###############################################################################

def compute_fitness_binary_measure(indices, chi, Bags, Labels, NUM_SOURCES, Parameters):
        
    tempBags = np.zeros((len(Bags)),dtype=np.object)
    for idb, bag in enumerate(Bags):
        tempbag = bag[:,indices]
        tempBags[idb] = tempbag
    
    ## Train ChI with current set of sources
    
    chi.train_chi(tempBags, Labels, Parameters) 

    ## Return indices of NUM_SOURCES sources and corresponding fitness for the set of sources
    return indices, chi.fitness

def select_mici_binary_measure(Bags, Labels, NUM_SOURCES, savepath, Parameters):
    
    print('\n Running selection for MICI Min-Max with Binary Measure... \n')

    #######################################################################
    ########################### Select Sources ############################
    #######################################################################
    
    nAllSources = Bags[0].shape[1]
    
    ind_set = []
    fitness_set = []
    
    all_ind = np.array(list(itertools.combinations(range(nAllSources), NUM_SOURCES)))
    
    chi = BinaryMIChoquetIntegral()
    
    ## Initialize selection with each potential source
    for k in tqdm(range(all_ind.shape[0])):
        
        ind, fitness = compute_fitness_binary_measure(all_ind[k], chi, Bags, Labels, NUM_SOURCES, Parameters)
    
        ind_set.append(ind.tolist())
        fitness_set.append(fitness)
    
    #######################################################################
    ########################### Save Results ##############################
    #######################################################################
    
    ## Order fitness values from greatest to least
    top_order = (-np.array(fitness_set)).argsort().astype(np.int16).tolist()
    top_order_ind = int(top_order[0])
    
    ## Values to return - indices of selected sources
    return_indices = ind_set[top_order_ind]
    
    tempBags = np.zeros((len(Bags)),dtype=np.object)
    for idb, bag in enumerate(Bags):
        tempbag = bag[:,return_indices]
        tempBags[idb] = tempbag
    
    ## Train ChI with current set of sources
    
    chi.train_chi(tempBags, Labels, Parameters) 
    
    
    ## Define paths to save selection info
    results_savepath_text = savepath + '/results.txt'
    results_savepath_json = savepath + '/results.json'
    results_savepath_dict = savepath + '/results.npy'
    
    ## Accumulate results for saving
    results = dict()
    results['best_set_index'] = top_order_ind
    results['best_set_activations'] = return_indices
    results['best_set_fitness'] = fitness_set[top_order_ind].tolist()
#    results['ind_set_all_unordered'] = ind_set
    results['fitness_set_all_unordered'] = fitness_set
    results['binary_measure'] = chi.measure.tolist()
    results['fm'] = chi.fm
#    results['ind_ordered_max_fitness_to_min'] = top_order
    
    ## Save parameters         
    with open(results_savepath_text, 'w') as f:
        json.dump(results, f, indent=2)
    f.close   
    
    ## Save to numpy file
    np.save(results_savepath_dict, results, allow_pickle=True)
    
    ## save to JSON
    with open (results_savepath_json, 'w') as outfile:
        json.dump(results, outfile, indent=2)

    return return_indices

#def select_binary_measure(indices, Bags, Labels, NUM_SOURCES, Parameters):
#        
#    nAllSources = Bags[0].shape[1]
#    
#    ## Greedily find features which maximize MICI fitness
#    ## Start with seed and find 2 source features set, then start with those two and find third features, etc.
#    for k in range(len(indices),NUM_SOURCES): ## Greedily add sources until desired number is reached
#        
#        fitness = []
#        
#        ##### Get divergence of sorts adding each feature, incrementally ######
#        for idx in range(0,nAllSources):
#            
#            current_indices = indices + [idx] ## all previously selected sources and the new one to evaluate
#            tempBags = deepcopy(Bags)
#            for idb, bag in enumerate(tempBags):
#                tempbag = bag[:,current_indices]
#                tempBags[idb] = tempbag
#            
#            ## Train ChI with current set of sources
##            chi = BinaryMIChoquetIntegral()
##            chi.train_chi(Bags, Labels, Parameters) 
#            chi = MIChoquetIntegral()
#            chi.train_chi_softmax(Bags, Labels, Parameters)
#            fitness[idx] = chi.fitness
#        
#        ## Sort fitness values and add feature with max fitness to set of sources (largest to smallest)
#        sorted_fitness_idx = (-fitness).argsort()
#        indices.append(int(sorted_fitness_idx[0]))
#
#    ## Return indices of NUM_SOURCES sources and corresponding fitness for the set of sources
#    return indices, fitness[sorted_fitness_idx[0]]

#def select_mici_binary_measure(Bags, Labels, NUM_SOURCES, savepath, Parameters):
#    
#    print('\n Running selection for MICI Min-Max with Binary Measure... \n')
#
#    #######################################################################
#    ########################### Select Sources ############################
#    #######################################################################
#    
#    nAllSources = Bags[0].shape[1]
#    
#    ind_set = []
#    fitness_set = []
#    
#    ## Initialize selection with each potential source
#    for k in tqdm(range(nAllSources)):
#        
#        input_ind = [k]
#        
#        ind, ent = select_binary_measure(input_ind, Bags, Labels, NUM_SOURCES, Parameters)
#    
#        ind_set.append(ind)
#        fitness_set.append(ent)
#    
#    #######################################################################
#    ########################### Save Results ##############################
#    #######################################################################
#    
#    ## Order fitness values from greatest to least
#    top_order = (-np.array(fitness_set)).argsort().astype(np.int16).tolist()
#    top_order_ind = int(top_order[0])
#    
#    ## Values to return - indices of selected sources
#    return_indices = ind_set[top_order_ind]
#    
#    ## Define paths to save selection info
#    results_savepath_text = savepath + '/results.txt'
#    results_savepath_json = savepath + '/results.json'
#    results_savepath_dict = savepath + '/results.npy'
#    
#    ## Accumulate results for saving
#    results = dict()
#    results['best_set_index'] = top_order_ind
#    results['best_set_activations'] = return_indices
#    results['best_set_fitness'] = fitness_set[top_order_ind]
#    results['ind_set_all_unordered'] = ind_set
#    results['fitness_set_all_unordered'] = fitness_set
#    results['ind_ordered_max_fitness_to_min'] = top_order
#    
#    ## Save parameters         
#    with open(results_savepath_text, 'w') as f:
#        json.dump(results, f, indent=2)
#    f.close   
#    
#    ## Save to numpy file
#    np.save(results_savepath_dict, results, allow_pickle=True)
#    
#    ## save to JSON
#    with open (results_savepath_json, 'w') as outfile:
#        json.dump(results, outfile, indent=2)
#
#    return return_indices


