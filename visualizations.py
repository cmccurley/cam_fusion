#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:19:20 2021

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  visualizations.py
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

## Default packages
import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors

## Pytorch packages
import torch
import torch.nn as nn


# details = {'method': current_model,
#           'threshold': thresh,
#           'iou mean':metric_iou_mean,
#           'iou std':metric_iou_std,
#           'precision mean':metric_precision_mean,
#           'precision std':metric_precision_std,
#           'recall mean':metric_recall_mean,
#           'recall std':metric_recall_std,
#           'f1 score mean':metric_f1_score_mean,
#           'f1 score std':metric_f1_score_std}

#methods = ['gradcam','gradcam++','layercam','scorecam','ablationcam','eigencam']

if __name__== "__main__":

    ######################################################################
    ########################### Set Parameters ###########################
    ######################################################################
    
    ## Read results from .npy file
#    results_dict = np.load('./dsiac/vgg16/baseline_experiments/results.npy', allow_pickle=True).item()
    method = 'iou'
    
    thresholds = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#    thresholds = [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
    
    ## Read results from .txt file
    results_dict = dict()
    details = dict()
    with open('./dsiac/vgg16/baseline_experiments/results.txt') as f:
        for line in f:
            if not(line == '\n'):
            
                key, value = line.split(':')
                
                if (key=='method'):
                    details[key] = value[:-1]
                else:
                    details[key] = float(value)
            
            else:
            
                model_name = details['method'] + '_thresh_' + str(details['threshold'])
                results_dict[model_name] = details
                details = dict()
    
    
    ######################################################################
    ######################## Amalgamate Results ##########################
    ######################################################################
    
    mean_key = method + ' mean'
    std_key = method + ' std'
    
    vis_dict = dict()
    
    results = []
    
    idx = 0
    for key in results_dict:
        method_results = results_dict[key]
        
        ## Handle first iteration
        if not(idx):
            previous_method = method_results['method']
            vis_dict[previous_method] = dict()
            method_means = []
            method_stds = []
            
        ## Append results for current method at different thresholds
        if (method_results['method'] == previous_method):
            method_means.append(method_results[mean_key])
            method_stds.append(method_results[std_key])
        else:
            ## Close previous method
            vis_dict[previous_method]['means'] = method_means
            vis_dict[previous_method]['stds'] = method_stds
            
            ## If new method, resest list and append
            method_means = []
            method_stds = []
            method_means.append(method_results[mean_key])
            method_stds.append(method_results[std_key])
            
    
        ##Update previous method as current method
        previous_method = method_results['method']
        vis_dict[previous_method] = dict()
        idx += 1
        
    ## Close last dictionary
    vis_dict[previous_method]['means'] = method_means
    vis_dict[previous_method]['stds'] = method_stds
            
        
        
        
    ######################################################################
    ######################## Create Visualization ########################
    ######################################################################
        
    ################# Plot all approaches together #######################
     
#    plt.style.use('dark_background')
    plt.style.use('default')
    
    plt.figure()
    labels = []
        
#    cmap = matplotlib.colors.ListedColormap(['#005f73','#0a9396','#94d2bd','#e9d8a6','#ee9b00','#ca6702','#ae2012','#9b2226'])
#    color_opt = ['#005f73','#94d2bd','#e9d8a6','#ee9b00','#ca6702','#9b2226']
#    color_opt = ['cyan','lime','orange','coral','lightpink','blue']
    color_opt = ['goldenrod','fuchsia','sandybrown','aqua','rebeccapurple','papayawhip']
    fmt_opt =['o','s','*','^','X','D'] 
    line_opt = ['solid', 'dotted', 'dashed', 'solid', 'dotted', 'dashed']
                                             
    layer_idx = 0                                
    key_idx = 0
    for key in vis_dict:
        
        if(layer_idx > 4):
            layer_idx=0
            key_idx +=1
        
        current_results = vis_dict[key]
        x = thresholds
        y = current_results['means']
        y_err = current_results['stds']
        
#        plt.errorbar(x,y,yerr=y_err,fmt=fmt_opt[layer_idx], label=key, color=color_opt[key_idx])
        
        plt.plot(x, y, marker=fmt_opt[layer_idx],linestyle=line_opt[layer_idx], label=key, color=color_opt[key_idx])
        plt.fill_between(x, np.array(y)-np.array(y_err), np.array(y)+np.array(y_err), color=color_opt[key_idx], alpha=0.5 )
        
        labels.append(key)
        
        
        layer_idx +=1
        
    plt.xlabel('Confidence Threshold')
    plt.ylabel('mIoU')
    plt.title('mIoU vs Segmentation Masking Threshold')
    plt.legend(labels, loc='lower center', bbox_to_anchor=(0.5,-0.6),ncol=6, fancybox=True, shadow=True) ## below
    
    plt.xlim([0.0,0.9])
    plt.ylim([0.0,0.6])
    
#    ax = plt.gca()
#    ax.set_facecolor((0.0,0.0,0.0))
    
#    plt.show()
    
    savepath = './dsiac/vgg16/baseline_experiments/plots/all_methods.png'
    plt.savefig(savepath, bbox_inches='tight')
    
    plt.close()
    
#    plt.legend(labels, loc='center left', bbox_to_anchor=(1,0.5),ncol=1, fancybox=True, shadow=True) ## beside
    
    
        
    ######## Plot each method individually; show performance by layer #########
    
#    plt.style.use('dark_background')
    plt.style.use('default')
    
#    color_opt = ['#005f73','#94d2bd','#e9d8a6','#ee9b00','#ca6702','#9b2226']
#    color_opt = ['cyan','lime','orange','coral','lightpink','blue']
    color_opt = ['papayawhip','fuchsia','sandybrown','aqua','rebeccapurple','goldenrod']
    fmt_opt =['o','s','*','^','X','D'] 
    line_opt = ['solid', 'dotted', 'dashed', 'solid', 'dotted', 'dashed']
                         
    for method_idx, method in enumerate(['gradcam','gradcam++','layercam','scorecam','ablationcam','eigencam']):
        
        plt.figure()
        labels = []
        
        for stage_idx, stage in enumerate([1,2,3,4,5]):
            
            key = method + '_layer' + str(stage)
        
            current_results = vis_dict[key]
            x = thresholds
            y = current_results['means']
            y_err = current_results['stds']
            
#            plt.errorbar(x,y,yerr=y_err,fmt=fmt_opt[stage_idx], label=key, color=color_opt[stage_idx])
            plt.plot(x, y, marker=fmt_opt[stage_idx],linestyle=line_opt[stage_idx], label=key, color=color_opt[stage_idx])
            plt.fill_between(x, np.array(y)-np.array(y_err), np.array(y)+np.array(y_err), color=color_opt[stage_idx], alpha=0.5 )
            
            
            labels.append(key)
            
        plt.xlabel('Confidence Threshold')
        plt.ylabel('mIoU')
        plt.title(method + ': mIoU vs Segmentation Masking Threshold')
        plt.legend(labels, loc='center left', bbox_to_anchor=(1,0.5),ncol=1, fancybox=True, shadow=True) ## beside
        
        plt.xlim([0.0,0.9])
        plt.ylim([0.0,0.6])
        
#        ax = plt.gca()
#        ax.set_facecolor((0.0,0.0,0.0))
        
#        plt.show()
        
        savepath = './dsiac/vgg16/baseline_experiments/plots/method_' + method + '.png'
        plt.savefig(savepath, bbox_inches='tight')
        
    
    
    ######### Plot layer individually; show performance by method #############
#    plt.style.use('dark_background')
    plt.style.use('default')
    
#    color_opt = ['#005f73','#94d2bd','#e9d8a6','#ee9b00','#ca6702','#9b2226']
#    color_opt = ['cyan','lime','orange','coral','lightpink','blue']
    color_opt = ['papayawhip','fuchsia','sandybrown','aqua','rebeccapurple','goldenrod']
    fmt_opt =['o','s','*','^','X','D'] 
    line_opt = ['solid', 'dotted', 'dashed', 'solid', 'dotted', 'dashed']
    
    for stage_idx, stage in enumerate([1,2,3,4,5]):                  
        
        plt.figure()
        labels = []
        
        for method_idx, method in enumerate(['gradcam','gradcam++','layercam','scorecam','ablationcam','eigencam']):
        
            key = method + '_layer' + str(stage)
        
            current_results = vis_dict[key]
            x = thresholds
            y = current_results['means']
            y_err = current_results['stds']
            
#            plt.errorbar(x,y,yerr=y_err,fmt=fmt_opt[method_idx], label=key, color=color_opt[method_idx])
            plt.plot(x, y, marker=fmt_opt[method_idx],linestyle=line_opt[method_idx], label=key, color=color_opt[method_idx])
            plt.fill_between(x, np.array(y)-np.array(y_err), np.array(y)+np.array(y_err), color=color_opt[method_idx], alpha=0.5 )
            
            labels.append(key)
            
        plt.xlabel('Confidence Threshold')
        plt.ylabel('mIoU')
        plt.title('Stage ' + str(stage) + ': mIoU vs Segmentation Masking Threshold')
        plt.legend(labels, loc='center left', bbox_to_anchor=(1,0.5),ncol=1, fancybox=True, shadow=True) ## beside
        
        plt.xlim([0.0,0.9])
        plt.ylim([0.0,0.6])
        
#        ax = plt.gca()
#        ax.set_facecolor((0.0,0.0,0.0))
        
#        plt.show()
    
        savepath = './dsiac/vgg16/baseline_experiments/plots/stage_' + str(stage) + '.png'
        plt.savefig(savepath, bbox_inches='tight')
        
        plt.close()
    
    
    
    

    










