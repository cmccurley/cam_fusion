#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:51:05 2022

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
    *  This product is Copyright (c) 2020 University of Florida
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
import os
import json
import argparse
import itertools
from PIL import Image
from tqdm import tqdm
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import precision_recall_fscore_support as prfs
from skimage.filters import threshold_otsu
#import sklearn.feature_selection.mutual_info_classif as MI
from sklearn.feature_selection import mutual_info_classif as MI
from scipy.signal import correlate2d as corr2d
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances
#from Compute_EMD import compute_EMD
#import ot

# PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

## Custom packages
import otsu_thresh_parameters
import initialize_network
from util import convert_gt_img_to_mask, cam_img_to_seg_img
from dataloaders import loaderPASCAL, loaderDSIAC
from feature_selection_utilities import select_features_sort_entropy, select_features_hag, select_features_exhaustive_entropy, select_features_exhaustive_iou, select_features_otsu_test

import numpy as np
import matplotlib.pyplot as plt
from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
from cam_functions.utils.image import show_cam_on_image, preprocess_image
from cam_functions import ActivationCAM, OutputScoreCAM
from cm_choquet_integral import ChoquetIntegral

torch.manual_seed(24)
torch.set_num_threads(1)


"""
%=====================================================================
%============================ Main ===================================
%=====================================================================
"""

if __name__== "__main__":
    
    print('================= Running Main =================\n')
    
    ######################################################################
    ######################### Set Parameters #############################
    ######################################################################
    
    GEN_PLOTS = True
    SAMPLE_IDX = 5 ## Airplane sample in PASCAL test dataset
#    SAMPLE_IDX = 3 ## Number 9 sample in the MNIST dataset
    TOP_K = 10
    BOTTOM_K = 10
    NUM_SUBSAMPLE = 5000
    
    vgg_idx = [[0,63],[64,191],[192,447],[448,959],[960,1471]]
   
    
    ## Import parameters 
    args = None
    parameters = otsu_thresh_parameters.set_parameters(args)
    
    if (parameters.DATASET == 'dsiac'):
            transform = transforms.Compose([transforms.ToTensor()])
            
    elif (parameters.DATASET == 'mnist'):
        transform = transforms.Compose([transforms.Resize(size=256),
                                        transforms.CenterCrop(224),
                                        transforms.Grayscale(3),
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize(256), 
                                    transforms.CenterCrop(224)])           

    target_transform = transforms.Compose([transforms.ToTensor()])
    
    ## Define dataset-specific parameters
    
    if (parameters.DATASET == 'cifar10'):
    
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
                   'horse', 'ship', 'truck')
        
        parameters.DATABASENAME = '../../Data/cifar10'
        
        ## Create data objects from training, validation and test .txt files set in the parameters file
        trainset = torchvision.datasets.CIFAR10(root=parameters.DATABASENAME, 
                                                train=True, 
                                                download=False, 
                                                transform=transform)
        val_size = 5000
        train_size = len(trainset) - val_size
        train_ds, val_ds = random_split(trainset, [train_size, val_size])
        
        testset = torchvision.datasets.CIFAR10(root=parameters.DATABASENAME, 
                                               train=False, 
                                               download=False, 
                                               transform=transform)
        
        ## Create data loaders for the data subsets.  (Handles data loading and batching.)
        train_loader = torch.utils.data.DataLoader(train_ds, 
                                                   batch_size=parameters.BATCH_SIZE, 
                                                   shuffle=True, 
                                                   num_workers=0, 
                                                   collate_fn=None, 
                                                   pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(val_ds, 
                                                 batch_size=parameters.BATCH_SIZE*2, 
                                                 shuffle=True, 
                                                 num_workers=0, 
                                                 collate_fn=None, 
                                                 pin_memory=False)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                  batch_size=parameters.BATCH_SIZE, 
                                                  shuffle=False, 
                                                  num_workers=0, 
                                                  collate_fn=None, 
                                                  pin_memory=False)
        
    elif (parameters.DATASET == 'pascal'):
        
        parameters.pascal_data_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/ImageSets/Main'
        parameters.pascal_image_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/JPEGImages'
        parameters.pascal_gt_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        
        parameters.all_classes = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4,
        'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9,
        'cow':10, 'diningtable':11, 'dog':12, 'horse':13,
        'motorbike':14, 'person':15, 'pottedplant':16,
        'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
        
        classes = parameters.bg_classes + parameters.target_classes
        
        parameters.NUM_CLASSES = len(classes)
        
        parameters.pascal_im_size = (224,224)
        
        trainset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'train', parameters, transform, target_transform)
        validset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'val', parameters, transform, target_transform)  
        testset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'test', parameters, transform, target_transform) 
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False) 
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        
    elif (parameters.DATASET == 'dsiac'):
        
        parameters.dsiac_data_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/DSIAC/bicubic'
        parameters.dsiac_gt_path = '/home/UFAD/cmccurley/Data/Army/Data/ATR_Extracted_Data/ground_truth_labels/dsiac'
        
        parameters.bg_classes = ['background']
        parameters.target_classes = ['target']
        parameters.all_classes = {'background':0, 'target':1}
        
        classes = parameters.bg_classes + parameters.target_classes
        parameters.NUM_CLASSES = len(classes)
        parameters.dsiac_im_size = (510,720)
        
        trainset = loaderDSIAC('../input/full_path/train_dsiac.txt', parameters, 'train', transform, target_transform) 
        validset = loaderDSIAC('../input/full_path/valid_dsiac.txt', parameters, 'valid', transform, target_transform) 
        testset = loaderDSIAC('../input/full_path/test_dsiac.txt', parameters, 'test', transform, target_transform) 
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False) 
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
            
    elif (parameters.DATASET == 'mnist'):
        
        parameters.mnist_data_path = '/home/UFAD/cmccurley/Public_Release/Army/data/mnist/'
        parameters.mnist_image_path = '/home/UFAD/cmccurley/Public_Release/Army/data/mnist/'
        parameters.mnist_gt_path = '/home/UFAD/cmccurley/Public_Release/Army/data/mnist/'
        
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        
        parameters.NUM_CLASSES = len(classes)
        
        trainset = datasets.MNIST(parameters.mnist_data_path, train=True,download=True,transform=transform)
        validset = datasets.MNIST(parameters.mnist_data_path, train=False,download=True,transform=transform)
        testset = datasets.MNIST(parameters.mnist_data_path, train=True,download=True,transform=transform)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False) 
        
        parameters.mnist_target_class = 9

    ## Define files to save epoch training/validation
    logfilename = parameters.outf + parameters.loss_file

    ######################################################################
    ####################### Initialize Network ###########################
    ######################################################################
    
    model = initialize_network.init(parameters)
    
    ## Save initial weights for further training
    temptrain = parameters.outf + parameters.parameter_path
    torch.save(model.state_dict(), temptrain)
    
    ## Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
                    
    ######################################################################
    ########################### Compute CAMs #############################
    ######################################################################
    
    if (parameters.DATASET == 'mnist'):
            parameters.MNIST_MEAN = (0.1307,)
            parameters.MNIST_STD = (0.3081,)
            norm_image = transforms.Normalize(parameters.MNIST_MEAN, parameters.MNIST_STD)
            gt_transform = transforms.Grayscale(1)
            
    else:
        norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ## Turn on gradients for CAM computation 
    for param in model.parameters():
        param.requires_grad = True

    model.eval()             
    
    activation_models = dict()
    
    if (parameters.model == 'vgg16'):
        for layer_idx in parameters.layers:
            activation_models[str(layer_idx)] = OutputScoreCAM(model=model, target_layers=[model.features[layer_idx]], use_cuda=parameters.cuda)

    target_category = None
    
    experiment_savepath = parameters.outf.replace('output','feature_ranking')
      

###############################################################################
############################# Sorting Analysis ################################
###############################################################################
    
#    new_save_path =  '/sort_entropy_seed_scorecam_2_sources'
#    new_save_path =  '/sort_entropy_seed_scorecam'
    new_save_path =  '/otsu_test'
#    new_save_path =  '/all_activations'
    parameters.feature_class = 'target'
    
    ###########################################################################
    ############### Extract activation maps and importance weights ############
    ###########################################################################
    
    new_directory = experiment_savepath + new_save_path
    img_savepath = new_directory
 
    sample_idx = 0
    for data in tqdm(test_loader):
        
        if (sample_idx==SAMPLE_IDX):
            
            if GEN_PLOTS:
                try:
                    os.mkdir(new_directory)
                    
                    img_savepath = new_directory
                except:
                    img_savepath = new_directory
        
            if (parameters.DATASET == 'mnist'):
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
    
            if (parameters.feature_class == 'background'):
                category = 0
            elif (parameters.feature_class == 'target'):
                category = 1
            elif (parameters.feature_class == 'mnist_class'):
                category = parameters.mnist_target_class
        
            ############# Convert groundtruth image into mask #############
            if (parameters.DATASET == 'mnist'):
                gt_image_input = images.permute(2,3,1,0)
                gt_image_input = gt_image_input.detach().cpu().numpy()
                gt_image_input = gt_image_input[:,:,:,0]
                
                gt_img = Image.fromarray(np.uint8(gt_image_input*255))
                gt_images = gt_transform(gt_img)
                gt_img = np.asarray(gt_images)/255
                
                gt_img[np.where(gt_img>0.2)] = 1
                gt_img[np.where(gt_img<=0.1)] = 0
                
            else:
                gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
            
            
            ## Invert groundtruth image for learning background
            if (parameters.feature_class == 'background'):
            
                zeros_idx = np.where(gt_img == 0)
                ones_idx = np.where(gt_img == 1)
                gt_img[zeros_idx] = 1
                gt_img[ones_idx] = 0
#                plt.figure()
#                plt.imshow(gt_img,cmap='gray')
            
            pred_label = int(labels.detach().cpu().numpy())
            
            ## Normalize input to the model
            cam_input = norm_image(images)
            
            ########### Get Activation Maps and Model Outputs per Map #########
            scores_by_stage = dict()
            
            for stage_idx, activation_model_idx in enumerate(activation_models):
                
                ## Get activation maps
                activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=category)
                importance_weights = importance_weights[0,:].numpy()
                
#                ## Scale fitness values in each layer to [0,1]
#                importance_weights = importance_weights - np.min(importance_weights)
#                importance_weights = importance_weights / (1e-7 + np.max(importance_weights))
                
                current_dict = dict()
                current_dict['activations'] = activations
                current_dict['importance_weights'] = importance_weights
                
                sorted_idx = (-importance_weights).argsort() ## indices of features in descending order
                sorted_fitness_values = importance_weights[sorted_idx]
                
                current_dict['sorted_idx'] = sorted_idx
                current_dict['sorted_fitness_values'] = sorted_fitness_values
       
                scores_by_stage[str(stage_idx+1)] = current_dict
                
                
                if not(stage_idx):
                    all_activations = activations
                    all_importance_weights = importance_weights
                else:
                    all_activations = np.append(all_activations, activations, axis=0)
                    all_importance_weights = np.append(all_importance_weights, importance_weights, axis=0)
                            
            break
        else:
            sample_idx +=1

    ###########################################################################
    ########################### Sort Activations ##############################
    ###########################################################################
    
#    add_to = [0,64,192,448,960]
#    
#    for idk in range(all_activations.shape[0]):
#        plt.figure()
#        plt.imshow(all_activations[idk,:,:])
#        plt.axis('off')
#        
#        for ids, ext in enumerate(vgg_idx):
#            if ((ext[0] <= idk) and (idk <= ext[1])):
#                stage = ids + 1
#    
#        title = 'VGG16 IDX: ' + str(idk) + ' | Stage: ' + str(stage)
#        plt.title(title)
#        
#        savepath = img_savepath + '/activation_' + str(idk) + '.png'
#        plt.savefig(savepath)
#        plt.close()
    
  
    ## Sort largest to smallest
    sorted_idx = (-all_importance_weights).argsort() ## indices of features in descending order
    
    ## Sort smallest to largest
#    sorted_idx = sorted_idx[::-1]
    
    sorted_fitness_values = all_importance_weights[sorted_idx]
#    
#    indices = list(sorted_idx[0:8])
#    
#    indices = [56,23,1326,1003,1036,799,94,140] ## Top entropy selection seed target and background
#    indices = [571,56,920,1329,327,80,243,185] ## Top entropy selection seed scorecam
#    indices = [571,56]
#    indices = [56,23,869,473,0,983,982,981,980]
#    indices = [56,23]
#    indices = [118,85,175,135,171,178,18]
#    indices = [60,636]
#    
#    indices = list(sorted_idx[0:5]) + list(sorted_idx[::-1][0:3])
    
    indices = list(sorted_idx[0:20])
    
#    activation_set = all_activations[indices,:,:]
    
#    sub_activations = activation_set
    
#    indices = [56,23,113,168,289,199,926,839,1084,1446]
    activation_set = all_activations[indices,:,:]
    sub_activations = activation_set
#            
#    for idk in range(activation_set.shape[0]):
#        plt.figure()
#        plt.imshow(activation_set[idk,:,:])
#        plt.axis('off')
#        
#        for ids, ext in enumerate(vgg_idx):
#            if ((ext[0] <= indices[idk]) and (indices[idk] <= ext[1])):
#                stage = ids
#        
#        title = 'VGG16 IDX: ' + str(indices[idk]) + ' | Stage: ' + str(stage+1)
##        title = 'VGG16 IDX: ' + str(sorted_idx[idk]) + ' | Stage: ' + str(stage+1)
#        plt.title(title)
#        
#        savepath = img_savepath + '/activation_' + str(idk) + '.png'
#        plt.savefig(savepath)
#        plt.close()
        
#    ###########################################################################
#    #################### Plot Ordered Feature Scores ##########################
#    ###########################################################################
#    
#    if (parameters.feature_class == 'background'):
#        method = 'background'
#        title = 'Background Feature Ranking Performance'
#        figure_savepath = parameters.outf.replace('output','feature_ranking') + '/independent_feature_ranking_background.png'
#        plt_color = 'blue'
#        shade_color = 'lightskyblue'
#    elif (parameters.feature_class == 'target'):
#        method = 'target'
#        title = 'Target Feature Ranking Performance'
#        figure_savepath = parameters.outf.replace('output','feature_ranking') + '/independent_feature_ranking_target.png'
##        figure_savepath_importance = parameters.outf.replace('output','feature_ranking') + '/independent_feature_ranking_importance_miou.png'
#        plt_color = 'darkorange'
#        shade_color = 'burlywood'
#    
#    num_features = len(sorted_fitness_values)
#    x_range = np.arange(1,num_features+1)
#    y_range_mean = sorted_fitness_values
#  
#  
#    ## Get legend entry
#    max_val = np.max(y_range_mean)
#    max_idx = np.where(y_range_mean==max_val)[0][0]
#    max_val = round(max_val,4)
#
#    plt.figure()
#    
#    plt.plot(x_range, y_range_mean, marker='^',color=plt_color)
#    plt.xlabel('Index of Ranked Features', fontsize = 12)
#   
#    plt.title(title, fontsize = 14)
#    plt.xlim((1,num_features))
#    
#    if (parameters.feature_class == 'background'):
#        plt.ylabel('Background Class Output', fontsize = 12)
##        plt.ylim((0.0,0.3))
#        legend_entry = 'Max output: ' + str(max_val)
#        
#    elif (parameters.feature_class == 'target'):
#         plt.ylabel('Target Class Output', fontsize = 12)
#         legend_entry = 'Max Output: ' + str(max_val)
#    
#    plt.legend([legend_entry],loc='upper right')
#    
#    plt.savefig(figure_savepath)
#    
#    plt.close() 
#    
#    ###########################################################################
#    ################### Plot Un-ordered Feature Scores ########################
#    ###########################################################################
#    
#    ## Load ranked feature order
#    if (parameters.feature_class == 'background'):
#        method = 'background'
#        title = 'Background Feature Performance'
#        figure_savepath = parameters.outf.replace('output','feature_ranking') + '/output_scores_background.png'
#        plt_color = 'blue'
#        shade_color = 'lightskyblue'
#    elif (parameters.feature_class == 'target'):
#        method = 'target'
#        title = 'Target Feature Performance'
#        figure_savepath = parameters.outf.replace('output','feature_ranking') + '/output_scores_target.png'
#        plt_color = 'darkorange'
#        shade_color = 'burlywood'
#    
#    num_features = len(all_importance_weights)
#    x_range = np.arange(1,num_features+1)
#    y_range_mean = all_importance_weights
#  
#  
#    ## Get legend entry
#    max_val = np.max(y_range_mean)
#    max_idx = np.where(y_range_mean==max_val)[0][0]
#    max_val = round(max_val,4)
#
#    plt.figure()
#    
#    plt.plot(x_range, y_range_mean, marker='^',color=plt_color)
#    plt.xlabel('Index of Features in Network', fontsize = 12)
#   
#    plt.title(title, fontsize = 14)
#    plt.xlim((1,num_features))
#    
#    if (parameters.feature_class == 'background'):
#        plt.ylabel('Background Class Output', fontsize = 12)
##        plt.ylim((0.0,0.3))
#        legend_entry = 'Max output: ' + str(max_val)
#        
#    elif (parameters.feature_class == 'target'):
#         plt.ylabel('Target Class Output', fontsize = 12)
#         legend_entry = 'Max Output: ' + str(max_val)
#    
#    plt.legend([legend_entry],loc='upper right')
#    
#    plt.savefig(figure_savepath)
#    
#    plt.close() 
    
#    ###########################################################################
#    ##################### Plot Activations per Stage ##########################
#    ###########################################################################
#    add_to = [0,64,192,448,960]
#    
#    for idl in range(1,len(parameters.layers)+1):
#    
#        all_activations = scores_by_stage[str(idl)]['activations']
#        top_activations_idx = scores_by_stage[str(idl)]['sorted_idx'][0:TOP_K]
#        bottom_activations_idx = scores_by_stage[str(idl)]['sorted_idx'][::-1]
#        bottom_activations_idx = bottom_activations_idx[0:BOTTOM_K]
#        
#        
#        ## Plot top K activations in each layer
#        top_activation_set = all_activations[top_activations_idx,:,:]
#                
#        for idk in range(top_activation_set.shape[0]):
#            plt.figure()
#            plt.imshow(top_activation_set[idk,:,:])
#            plt.axis('off')
#            
#            for ids, ext in enumerate(vgg_idx):
#                if ((ext[0] <= top_activations_idx[idk]+add_to[idl-1]) and (top_activations_idx[idk]+add_to[idl-1] <= ext[1])):
#                    stage = ids +1
#            
#            current_idk = top_activations_idx[idk]+add_to[idl-1]
#            
#            title = 'VGG16 IDX: ' + str(current_idk) + ' | Stage: ' + str(stage)
#            plt.title(title)
#            
#            savepath = img_savepath+ '/activations/top' + '/stage_' + str(idl) + '_activation_' + str(idk) + '.png'
#            plt.savefig(savepath)
#            plt.close()
#            
#        ## Plot bottom K activations in each layer
#        bottom_activation_set = all_activations[bottom_activations_idx,:,:]
#                
#        for idk in range(bottom_activation_set.shape[0]):
#            plt.figure()
#            plt.imshow(bottom_activation_set[idk,:,:])
#            plt.axis('off')
#            
#            for ids, ext in enumerate(vgg_idx):
#                if ((ext[0] <= bottom_activations_idx[idk]+add_to[idl-1]) and (bottom_activations_idx[idk]+add_to[idl-1] <= ext[1])):
#                    stage = ids +1
#            
#            current_idk = bottom_activations_idx[idk]+add_to[idl-1]
#            
#            title = 'VGG16 IDX: ' + str(current_idk) + ' | Stage: ' + str(stage)
#            plt.title(title)
#            
#            savepath = img_savepath+ '/activations/bottom' + '/stage_' + str(idl) + '_activation_' + str(idk) + '.png'
#            plt.savefig(savepath)
#            plt.close()
#            
#    ###########################################################################
#    ##################### Sort Activations Most Different #####################
#    ###########################################################################
#    NUM_SOURCES = 6
#    
#    #################### Sort with correlation (min average correlation) ##############
    
#    activations = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
#    for idk in range(all_activations.shape[0]):
#        activations[idk,:] = all_activations[idk,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
#        
#    
#    ## Compute pairwise similarity matrix
#    dist_mat = cs(activations)
#    
#    ## Find the feature with the smallest average similarity between all current sources
#    for k in range(activation_set.shape[0],NUM_SOURCES):
#        corr_mat = dist_mat[indices,:]
#        
#        avg_dist = np.mean(corr_mat,axis=0)
#        sorted_corr_idx = (-avg_dist).argsort()
#        sorted_corr_idx = sorted_corr_idx[::-1]
#        
#        if (k == activation_set.shape[0]):
#            sorted_corr_idx_first = sorted_corr_idx
#    
#        new_idx = 0
#        while(len(np.unique(all_activations[sorted_corr_idx[new_idx],:,:] >0.01)) < 2):
#            new_idx += 1
#        
#        ## Add chosen source to set of sources
#        indices.append(sorted_corr_idx[new_idx])
#        activation_set = np.concatenate((activation_set,np.expand_dims(all_activations[sorted_corr_idx[new_idx],:,:],axis=0)),axis=0)
#        
##        plt.figure()
##        plt.imshow(all_activations[sorted_corr_idx[new_idx],:,:])
##        plt.axis('off')
        
#    #################### Sort with L1 dissimilarity (max L1 )###################
#    
#    activations = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
#    for idk in range(all_activations.shape[0]):
#        activations[idk,:] = all_activations[idk,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
#        
#    
#    ## Compute pairwise similarity matrix
#    dist_mat = manhattan_distances(activations, sum_over_features=True)
#    
#    ## Find the feature with the smallest average similarity between all current sources
#    for k in range(activation_set.shape[0],NUM_SOURCES):
#        corr_mat = dist_mat[indices,:]
#        
#        avg_dist = np.mean(corr_mat,axis=0)
#        sorted_corr_idx = (-avg_dist).argsort()
##        sorted_corr_idx = sorted_corr_idx[::-1]
#        
#        if (k == activation_set.shape[0]):
#            sorted_corr_idx_first = sorted_corr_idx
#    
#        new_idx = 0
#        while((len(np.unique(all_activations[sorted_corr_idx[new_idx],:,:] >0.01)) < 2) or (sorted_corr_idx[new_idx] in indices)):
#            new_idx += 1
#        
#        
#        ## Add chosen source to set of sources
#        indices.append(sorted_corr_idx[new_idx])
#        activation_set = np.concatenate((activation_set,np.expand_dims(all_activations[sorted_corr_idx[new_idx],:,:],axis=0)),axis=0)
#        
#        plt.figure()
#        plt.imshow(all_activations[sorted_corr_idx[new_idx],:,:])
#        plt.axis('off')
#        
#    ## Plot activations in order of least correlated to most correlated 
#    for idk in range(all_activations.shape[0]):
#        
#        plt.figure()
#        plt.imshow(all_activations[sorted_corr_idx_first[idk],:,:])
#        plt.axis('off')
#        
#        for ids, ext in enumerate(vgg_idx):
#            if ((ext[0] <= sorted_corr_idx_first[idk]) and (sorted_corr_idx_first[idk] <= ext[1])):
#                stage = ids
#        
#        title = 'VGG16 IDX: ' + str(sorted_corr_idx_first[idk]) + ' | Stage: ' + str(stage+1)
#        plt.title(title)
#        
#        savepath = img_savepath + '/activation_' + str(idk) + '.png'
#        plt.savefig(savepath)
#        plt.close()
   
    
    ################# Sort with Mutual Information (Max MI) ###################
##    handpicked_idx = [23,56] ## Layer 1
#    handpicked_idx = [56,60] ## Layer 1
#    handpicked_activations = activations[handpicked_idx,:,:]
#    
##    temp1 = compute_EMD(handpicked_activations[0,:,:].reshape((activations.shape[1]*activations.shape[2])),handpicked_activations[1,:,:].reshape((activations.shape[1]*activations.shape[2])))
#    
#    hgram, _, _ = np.histogram2d(handpicked_activations[0,:,:].reshape((activations.shape[1]*activations.shape[2])),handpicked_activations[1,:,:].reshape((activations.shape[1]*activations.shape[2])),bins=256)
#    
#    pxy = hgram / float(np.sum(hgram))
#    px = np.sum(pxy, axis=1) # marginal for x over y
#    py = np.sum(pxy, axis=0) # marginal for y over x
#    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
#    # Now we can do the calculation using the pxy, px_py 2D arrays
#    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
#    mutual_information1 = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
#    temp1 = wasserstein_distance(handpicked_activations[0,:,:].reshape((activations.shape[1]*activations.shape[2])),handpicked_activations[1,:,:].reshape((activations.shape[1]*activations.shape[2])))
#    
#    plt.figure()
#    plt.imshow(temp)
    
#    handpicked_idx = [23,56] ## Layer 1
#    handpicked_activations = activations[handpicked_idx,:,:]
#    
#    hgram, x_edges, y_edges = np.histogram2d(handpicked_activations[0,:,:].reshape((activations.shape[1]*activations.shape[2])),handpicked_activations[1,:,:].reshape((activations.shape[1]*activations.shape[2])),bins=256)
#    
#    pxy = hgram / float(np.sum(hgram))
#    px = np.sum(pxy, axis=1) # marginal for x over y
#    py = np.sum(pxy, axis=0) # marginal for y over x
#    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
#    # Now we can do the calculation using the pxy, px_py 2D arrays
#    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
#    mutual_information2 = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    
#    temp2 = compute_EMD(handpicked_activations[0,:,:].reshape((activations.shape[1]*activations.shape[2])),handpicked_activations[1,:,:].reshape((activations.shape[1]*activations.shape[2])))
    
#    temp2 = wasserstein_distance(handpicked_activations[0,:,:].reshape((activations.shape[1]*activations.shape[2])),handpicked_activations[1,:,:].reshape((activations.shape[1]*activations.shape[2])))
    
#    plt.figure()
#    plt.imshow(temp)
   
#    ###########################################################################
#    ################## Select Features Sort Histogram Entropy #################
#    ###########################################################################
#    indices_savepath = img_savepath + '/feature_indices_sort_entropy.txt'
#        
#    indices, acivation_set, _ = select_features_sort_entropy(indices, activation_set, all_activations, NUM_SOURCES, indices_savepath)
    
#    ###########################################################################
#    ########################## Select Features HAG ############################
#    ###########################################################################
#    indices_savepath = img_savepath + '/feature_indices_hag.txt'
#        
#    indices, activation_set = select_features_hag(indices, activation_set, all_activations, NUM_SOURCES, indices_savepath)
    
    images = images.permute(2,3,1,0)
    images = images.detach().cpu().numpy()
    image = images[:,:,:,0]
    
#    for SOURCES in [2,3,4,5,6,7,8]:
    
#    vgg_idx = [[0,63],[64,191],[192,447],[448,959],[960,1471]]
#    all_activations = all_activations[vgg_idx[4][0]:vgg_idx[4][1]+1,:,:]
    


    ## Loop through stages
#    for stage in [0,1,2,3,4]:
    for stage in [0]:
        
        ## Update activation set
#        scores_by_stage
    
        all_activations = scores_by_stage[str(stage+1)]['activations']
        all_importance_weights = scores_by_stage[str(stage+1)]['importance_weights']
        sorted_fitness_values = scores_by_stage[str(stage+1)]['sorted_fitness_values']
        sorted_idx = scores_by_stage[str(stage+1)]['sorted_idx']

#        indices = list(sorted_idx)
        
        indices = list(sorted_idx[0:10])
#        
        sub_activations = all_activations[indices,:,:] 
        
#        sub_activations = all_activations
        
        flipped_activations = sub_activations
       
        ## Downsample and add inverted activations
        for act_idx in range(sub_activations.shape[0]):
            
            flipped_act = np.abs(1-sub_activations[act_idx,:,:])
            
            flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
        
        sub_activations = flipped_activations
        
        for SOURCES in [2]:
            new_save_path =  new_directory + '/otsu_test_' + 'stage_' + str(stage+1) + '_sources_' + str(SOURCES)
            
            parameters.feature_class = 'target'
            
            NUM_SOURCES = SOURCES
            
            ###########################################################################
            ############### Extract activation maps and importance weights ############
            ###########################################################################
            
            
            try:
                os.mkdir(new_save_path)
                
                img_savepath = new_save_path
            except:
                img_savepath = new_save_path
        
        
            print(f'\n!!!!!!!!!!! SELECTING {str(SOURCES)} SOURCES !!!!!!!!!!!!!\n')
        
    #        ##########################################################################
    #        ################## Select Features Exhaustive Entropy ####################
    #        ##########################################################################
    #        
    #        indices_savepath = img_savepath
    #        indices, activation_set = select_features_exhaustive_entropy(all_activations, NUM_SOURCES, indices_savepath)
    #        
    #        results_savepath_dict = indices_savepath + '/results.npy'
    #        results = np.load(results_savepath_dict,allow_pickle=True).item()
            
            ###########################################################################
            ##################### Select Features Exhaustive IoU ######################
            ###########################################################################
            indices_savepath = img_savepath
            indices, activation_set = select_features_otsu_test(sub_activations, gt_img, NUM_SOURCES, indices_savepath)
            
            ###########################################################################
            ############################## Compute ChI ################################
            ###########################################################################
        
            for k in range(activation_set.shape[0]):
                plt.figure()
                plt.imshow(activation_set[k,:,:])
                plt.axis('off')
                title = 'Activation idx: ' + str(k+1)
                plt.title(title)
                
                savename = img_savepath + '/activation_idx_' + str(k+1) + '.png'
                plt.savefig(savename)
                plt.close()
            
            labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
            
            data = np.zeros((activation_set.shape[0],activation_set.shape[1]*activation_set.shape[2]))
            
            for idx in range(activation_set.shape[0]):
                data[idx,:] = np.reshape(activation_set[idx,:,:],(1,activation_set[idx,:,:].shape[0]*activation_set[idx,:,:].shape[1]))
                
            print('Computing Fuzzy Measure...')    
            
            chi = ChoquetIntegral()    
            
            # train the chi via quadratic program 
            chi.train_chi(data, labels)
            fm = chi.fm
            
            fm_savepath = img_savepath + '/fm'
            np.save(fm_savepath, fm)
                
            fm_file = fm_savepath + '.txt'
            ## Save parameters         
            with open(fm_file, 'w') as f:
                json.dump(fm, f, indent=2)
            f.close 
                
            ###########################################################################
            ############################ Print Equations ##############################
            ###########################################################################
            if (NUM_SOURCES < 7):
                eqt_dict = chi.get_linear_eqts()
                
                ## Save equations
                savename = img_savepath + '/chi_equations.txt'
                with open(savename, 'w') as f:
                    json.dump(eqt_dict, f, indent=2)
                f.close 
                
            ###########################################################################
            ################################ Test Data ################################
            ###########################################################################
            print('Testing data...')  
            
            ## Compute ChI at every pixel location
            data_out = np.zeros((data.shape[1]))
            data_out_sorts = []
            data_out_sort_labels = np.zeros((data.shape[1]))   
        
            all_sorts = chi.all_sorts
            
            if (NUM_SOURCES < 7):
            
                for idx in range(data.shape[1]):
                    data_out[idx], data_out_sort_labels[idx] = chi.chi_by_sort(data[:,idx])               
                
                ## Define custom colormap
                custom_cmap = plt.cm.get_cmap('jet', len(all_sorts))
                norm = matplotlib.colors.Normalize(vmin=1,vmax=len(all_sorts))
                
                ## Plot frequency histogram of sorts for single image
                sort_tick_labels = []
                for element in all_sorts:
                    sort_tick_labels.append(str(element))
                hist_bins = list(np.arange(0,len(all_sorts)+1)+0.5)
                plt.figure(figsize=[10,8])
                ax = plt.axes()
                plt.xlabel('Sorts', fontsize=14)
                plt.ylabel('Count', fontsize=14)
                n, bins, patches = plt.hist(data_out_sort_labels, bins=hist_bins, align='mid')
                plt.xlim([0.5,len(all_sorts)+0.5])
                ax.set_xticks(list(np.arange(1,len(all_sorts)+1)))
                ax.set_xticklabels(sort_tick_labels,rotation=45,ha='right')
                for idx, current_patch in enumerate(patches):
                    current_patch.set_facecolor(custom_cmap(norm(idx+1)))
                title = 'Histogram of Sorts'
                plt.title(title, fontsize=18)
                savename = img_savepath + '/hist_of_sorts.png'
                plt.savefig(savename)
                plt.close()
                
                ## Plot image colored by sort
                data_out_sort_labels = data_out_sort_labels.reshape((activation_set.shape[1],all_activations.shape[2]))
                plt.figure(figsize=[10,8])
                plt.imshow(data_out_sort_labels,cmap=custom_cmap)
                plt.axis('off')
                plt.colorbar()
                title = 'Image by Sorts'
                plt.title(title)
                savename = img_savepath + '/img_by_sorts.png'
                plt.savefig(savename)
                plt.close()
                
                ## Plot each sort, individually
                for idx in range(1,len(all_sorts)+1):
                    plt.figure(figsize=[10,8])
                    
                    gt_overlay = np.zeros((gt_img.shape[0],gt_img.shape[1]))
                    gt_overlay[np.where(gt_img == True)] = 1
                    plt.imshow(gt_overlay, cmap='gray', alpha=1.0)
                    
                    sort_idx = np.where(data_out_sort_labels == idx)
                    sort_overlay = np.zeros((gt_img.shape[0],gt_img.shape[1]))
                    sort_overlay[sort_idx] = idx
                    plt.imshow(sort_overlay, cmap='plasma', alpha = 0.7)
                    plt.axis('off')
                    title = 'Sort: ' + str(all_sorts[idx-1])
                    plt.title(title, fontsize=16)
                    savename = img_savepath + '/sort_idx_' + str(idx) + '.png'
                    plt.savefig(savename)
                    plt.close()
                    
            else:
                for idx in range(data.shape[1]):
                    data_out[idx] = chi.chi_quad(data[:,idx])
                
            ## Normalize ChI output
            data_out = data_out - np.min(data_out)
            data_out = data_out / (1e-7 + np.max(data_out))
            
            ###########################################################################
            ############################## Compute IoU ################################
            ###########################################################################
            ## Binarize feature
            
            gt_img = gt_img > 0
            
            try:
                img_thresh = threshold_otsu(data_out)
                binary_feature_map = data_out.reshape((activation_set.shape[1],activation_set.shape[2])) > img_thresh
            except:
                binary_feature_map = data_out < 0.1
        
            ## Compute fitness as IoU to pseudo-groundtruth
            intersection = np.logical_and(binary_feature_map, gt_img)
            union = np.logical_or(binary_feature_map, gt_img)
            
            try:
                iou_score = np.sum(intersection) / np.sum(union)
            except:
                iou_score = 0
         
            ###########################################################################
            ############################## Visualizations #############################
            ###########################################################################
        
            data_out = data_out.reshape((activation_set.shape[1],activation_set.shape[2]))
            plt.figure()
            plt.imshow(data_out)
            plt.axis('off')
            plt.colorbar()
            title = 'Raw FusionCAM; IoU: ' + str(iou_score)
            plt.title(title)
            
            savename = img_savepath + '/raw_fusioncam.png'
                
            plt.savefig(savename)
            plt.close()
            
            cam_image = show_cam_on_image(image, data_out, True)
            
            plt.figure()
            plt.imshow(cam_image, cmap='jet')
            title = 'FusionCAM; IoU: ' + str(iou_score)
            plt.title(title)
            plt.axis('off')
            
            savename = img_savepath + '/fusioncam.png'
            plt.savefig(savename)
            plt.close()
        
            sample_idx +=1

    
    
    
         
  

    
    