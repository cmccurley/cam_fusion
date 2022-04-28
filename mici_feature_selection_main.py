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
import os
import json
from PIL import Image
from tqdm import tqdm
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import matplotlib.pyplot as plt
import matplotlib.colors
from skimage.filters import threshold_otsu

import dask.array as da

# PyTorch packages
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision import datasets

## Custom packages
import initialize_network
from util import convert_gt_img_to_mask, cam_img_to_seg_img
from dataloaders import loaderPASCAL, loaderDSIAC
from feature_selection_utilities import select_features_sort_entropy, select_features_hag, select_features_exhaustive_entropy, select_features_exhaustive_iou, select_features_divergence_pos_and_neg
from feature_selection_utilities import dask_select_features_divergence_pos_and_neg
import numpy as np
#from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
from cam_functions.utils.image import show_cam_on_image, preprocess_image
from cam_functions import ActivationCAM, OutputScoreCAM
from cm_choquet_integral import ChoquetIntegral

import scipy.io as io

## Custom packages
from mici_feature_selection_parameters import set_parameters as set_parameters
from cm_MICI.util.cm_mi_choquet_integral import MIChoquetIntegral

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
#    SAMPLE_IDX = 5 ## Airplane sample in PASCAL test dataset
    SAMPLE_IDX = 3 ## Number 9 sample in the MNIST dataset
    TOP_K = 10
    BOTTOM_K = 10
    NUM_SUBSAMPLE = 5000
    
    vgg_idx = [[0,63],[64,191],[192,447],[448,959],[960,1471]]
   
    
    ## Import parameters 
    args = None
    parameters = set_parameters(args)
    
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
        
        parameters.mnist_target_class = 7
        parameters.mnist_background_class = 1
        
        target_class = 7
        background_class = 1
        
        nPBags = 3
        nNBags = 3

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
    
    experiment_savepath = parameters.outf.replace('output','mici')
      
###############################################################################
############################# Sorting Analysis ################################
###############################################################################
    
    new_save_path =  '/mici_genmean_7_versus_1'
    parameters.feature_class = 'target'
    
    ###########################################################################
    ############### Extract activation maps and importance weights ############
    ###########################################################################
    
    new_directory = experiment_savepath + new_save_path
    img_savepath = new_directory
    
    try:
        os.mkdir(new_directory)
        
        img_savepath = new_directory
    except:
        img_savepath = new_directory
    
    print('\nExtracting POSITIVE bag activation maps...\n')
    
    dask_arrays = []
    dask_gt_arrays = []
    sample_idx = 0
    for data in tqdm(train_loader):
        
        if (parameters.DATASET == 'mnist'):
            images_p, labels_p = data[0].to(device), data[1].to(device)
        else:
            images_p, labels_p, gt_images_p = data[0].to(device), data[1].to(device), data[2]
            
        if (int(labels_p.detach().cpu().numpy())==parameters.mnist_target_class):
    
            if (parameters.feature_class == 'background'):
                category = 0
            elif (parameters.feature_class == 'target'):
                category = 1
            elif (parameters.feature_class == 'mnist_class'):
                category = parameters.mnist_target_class
            
            ############# Convert groundtruth image into mask #############
            if (parameters.DATASET == 'mnist'):
                gt_image_input_p = images_p.permute(2,3,1,0)
                gt_image_input_p = gt_image_input_p.detach().cpu().numpy()
                gt_image_input_p = gt_image_input_p[:,:,:,0]
                
                gt_img_p = Image.fromarray(np.uint8(gt_image_input_p*255))
                gt_images_p = gt_transform(gt_img_p)
                gt_img_p = np.asarray(gt_images_p)/255
                
                gt_img_p[np.where(gt_img_p>0.15)] = 1
                gt_img_p[np.where(gt_img_p<=0.15)] = 0
                
            else:
                gt_img_p = convert_gt_img_to_mask(gt_images_p, labels_p, parameters)     
            
            ## Invert groundtruth image for learning background
            if (parameters.feature_class == 'background'):
            
                zeros_idx = np.where(gt_img_p == 0)
                ones_idx = np.where(gt_img_p == 1)
                gt_img_p[zeros_idx] = 1
                gt_img_p[ones_idx] = 0
            
            pred_label_p = int(labels_p.detach().cpu().numpy())
            
            ## Normalize input to the model
            cam_input_p = norm_image(images_p)
            
            ########### Get Activation Maps and Model Outputs per Map #########
            scores_by_stage_p = dict()
            
            for stage_idx, activation_model_idx in enumerate(activation_models):
                
                ## Get activation maps
                activations_p, importance_weights_p = activation_models[activation_model_idx](input_tensor=cam_input_p, target_category=target_class)
                importance_weights_p = importance_weights_p[0,:].numpy()
                
#                ## Scale fitness values in each layer to [0,1]
#                importance_weights = importance_weights - np.min(importance_weights)
#                importance_weights = importance_weights / (1e-7 + np.max(importance_weights))
                
                current_dict_p = dict()
                current_dict_p['activations'] = activations_p
#                current_dict['importance_weights'] = importance_weights
#                
#                sorted_idx = (-importance_weights).argsort() ## indices of features in descending order
#                sorted_fitness_values = importance_weights[sorted_idx]
#                
#                current_dict['sorted_idx'] = sorted_idx
#                current_dict['sorted_fitness_values'] = sorted_fitness_values
#       
                scores_by_stage_p[str(stage_idx+1)] = current_dict_p
                
                
                if not(stage_idx):
                    all_activations_p = activations_p
                    all_importance_weights_p = importance_weights_p
                else:
                    all_activations_p = np.append(all_activations_p, activations_p, axis=0)
                    all_importance_weights_p = np.append(all_importance_weights_p, importance_weights_p, axis=0)
            
            
            ## Convert to dask arrays
            activations = np.zeros((all_activations_p.shape[1]*all_activations_p.shape[2],all_activations_p.shape[0]))
            for k in range(all_activations_p.shape[0]):
                activations[:,k] = all_activations_p[k,:,:].reshape((all_activations_p.shape[1]*all_activations_p.shape[2]))
            
            if sample_idx < nPBags:
                if not(sample_idx):
                    
                    pBags = np.zeros((nPBags,),dtype=np.object)
                    pLabels = np.ones((nPBags,),dtype=np.uint8)
                    pBagGT = np.zeros((nPBags,),dtype=np.object)
                    
                    pBags[sample_idx] = activations
                    pBagGT[sample_idx] = np.expand_dims(gt_img_p,axis=0)
                    
#                    dask_all_activations_p = da.from_array(activations.T)
#                    dask_gt_p = da.from_array(np.expand_dims(gt_img_p,axis=0))
                    sample_idx += 1
                else:
                    
                    pBags[sample_idx] = activations
                    pBagGT[sample_idx] = np.expand_dims(gt_img_p,axis=0)
                    
#                    dask_all_activations_p = da.concatenate((dask_all_activations_p, da.from_array(activations.T)), axis=1)
#                    dask_gt_p = da.concatenate((dask_gt_p, da.from_array(np.expand_dims(gt_img_p,axis=0))),axis=0)
                    sample_idx += 1
            else:
                break
            
    print('\nExtracting NEGATIVE bag activation maps...\n')
    
    dask_arrays = []
    sample_idx = 0
    for data in tqdm(train_loader):
                    
        if (parameters.DATASET == 'mnist'):
            images_n, labels_n = data[0].to(device), data[1].to(device)
        else:
            images_n, labels_n, gt_images_n = data[0].to(device), data[1].to(device), data[2]
            
        if (int(labels_n.detach().cpu().numpy())==parameters.mnist_background_class):
    
            if (parameters.feature_class == 'background'):
                category = 0
            elif (parameters.feature_class == 'target'):
                category = 1
            elif (parameters.feature_class == 'mnist_class'):
                category = parameters.mnist_target_class
            
            ############# Convert groundtruth image into mask #############
            if (parameters.DATASET == 'mnist'):
                gt_image_input_n = images_n.permute(2,3,1,0)
                gt_image_input_n = gt_image_input_n.detach().cpu().numpy()
                gt_image_input_n = gt_image_input_n[:,:,:,0]
                
                gt_img_n = Image.fromarray(np.uint8(gt_image_input_n*255))
                gt_images_n = gt_transform(gt_img_n)
                gt_img_n = np.asarray(gt_images_n)/255
                
                gt_img_n[np.where(gt_img_n>0.15)] = 1
                gt_img_n[np.where(gt_img_n<=0.15)] = 0
                
            else:
                gt_img_n = convert_gt_img_to_mask(gt_images_n, labels_n, parameters)            
            
            ## Invert groundtruth image for learning background
            if (parameters.feature_class == 'background'):
            
                zeros_idx = np.where(gt_img_n == 0)
                ones_idx = np.where(gt_img_n == 1)
                gt_img_n[zeros_idx] = 1
                gt_img_n[ones_idx] = 0
#                plt.figure()
#                plt.imshow(gt_img,cmap='gray')
            
            pred_label_n = int(labels_n.detach().cpu().numpy())
            
            ## Normalize input to the model
            cam_input_n = norm_image(images_n)
            
            
            ########### Get Activation Maps and Model Outputs per Map #########
            scores_by_stage_n = dict()
            
            for stage_idx, activation_model_idx in enumerate(activation_models):
                
                ## Get activation maps
                activations_n, importance_weights_n = activation_models[activation_model_idx](input_tensor=cam_input_n, target_category=target_class)
                importance_weights_n = importance_weights_n[0,:].numpy()
                
#                ## Scale fitness values in each layer to [0,1]
#                importance_weights = importance_weights - np.min(importance_weights)
#                importance_weights = importance_weights / (1e-7 + np.max(importance_weights))
                
                current_dict_n = dict()
                current_dict_n['activations'] = activations_n
#                current_dict['importance_weights'] = importance_weights
#                
#                sorted_idx = (-importance_weights).argsort() ## indices of features in descending order
#                sorted_fitness_values = importance_weights[sorted_idx]
#                
#                current_dict['sorted_idx'] = sorted_idx
#                current_dict['sorted_fitness_values'] = sorted_fitness_values
#       
                scores_by_stage_n[str(stage_idx+1)] = current_dict_n
                
                
                if not(stage_idx):
                    all_activations_n = activations_n
                    all_importance_weights_n = importance_weights_n
                else:
                    all_activations_n = np.append(all_activations_n, activations_n, axis=0)
                    all_importance_weights_n = np.append(all_importance_weights_n, importance_weights_n, axis=0)
            
            ## Convert to dask arrays
            activations = np.zeros((all_activations_p.shape[1]*all_activations_p.shape[2],all_activations_p.shape[0]))
            for k in range(all_activations_p.shape[0]):
                activations[:,k] = all_activations_p[k,:,:].reshape((all_activations_p.shape[1]*all_activations_p.shape[2]))

            if sample_idx < nNBags:
                if not(sample_idx):
                    
                    nBags = np.zeros((nNBags,),dtype=np.object)
                    nLabels = np.zeros((nNBags,),dtype=np.uint8)
                    nBagGT = np.zeros((nNBags,),dtype=np.object)
                    
                    nBags[sample_idx] = activations
                    nBagGT[sample_idx] = np.expand_dims(gt_img_n,axis=0)
                    
#                    dask_all_activations_p = da.from_array(activations.T)
#                    dask_gt_p = da.from_array(np.expand_dims(gt_img_p,axis=0))
                    sample_idx += 1
                else:
                    
                    nBags[sample_idx] = activations
                    nBagGT[sample_idx] = np.expand_dims(gt_img_n,axis=0)
                    
#                    dask_all_activations_p = da.concatenate((dask_all_activations_p, da.from_array(activations.T)), axis=1)
#                    dask_gt_p = da.concatenate((dask_gt_p, da.from_array(np.expand_dims(gt_img_p,axis=0))),axis=0)
                    sample_idx += 1
            else:
                break
    
    
    Bags = np.concatenate((nBags,pBags))
    Labels = np.concatenate((nLabels, pLabels))
    
    ## Downsample activations
    for idx in range(len(Bags)):
        Bags[idx] = Bags[idx][:,0:64]
    
    ## Add inverted activations
    for idx in range(len(Bags)):
        for idc in range(Bags[idx].shape[1]):
            inv_act = np.expand_dims(np.abs(1-Bags[idx][:,idc]),axis=1)
            Bags[idx] = np.concatenate((Bags[idx],inv_act),axis=1)
      
    
    ## Sample from selected sources
    indices = [38, 51, 48, 8]
    for idx in range(len(Bags)):
        Bags[idx] = Bags[idx][:,indices]
        
    ###########################################################################
    ############################## Train MICI #################################
    ###########################################################################
    
    ## Initialize Choquet Integral instance
    chi_genmean = MIChoquetIntegral()
    
    # Training Stage: Learn measures given training bags and labels
    chi_genmean.train_chi_softmax(Bags, Labels, parameters) ## generalized-mean model
    
#    ## Only use first stage
#    dask_all_activations_p = dask_all_activations_p[0:64,:]
#    dask_all_activations_n = dask_all_activations_n[0:64,:] 
#    
##    dask_all_activations_p = dask_all_activations_p[960:,:]
##    dask_all_activations_n = dask_all_activations_n[960:,:] 
#
#    ## Flip activations
#    sub_activations_p = dask_all_activations_p
#    sub_activations_n = dask_all_activations_n
#        
#    flipped_activations_p = sub_activations_p
#    flipped_activations_n = sub_activations_n
#    
#    ## Downsample and add inverted activations
#    for act_idx in range(sub_activations_p.shape[0]):
#        
#        flipped_act = da.from_array(np.expand_dims(da.fabs(1-sub_activations_p[act_idx,:].compute()),axis=0))
#        
#        flipped_activations_p = da.concatenate((flipped_activations_p, flipped_act),axis=0)
#    
#    sub_activations_p = flipped_activations_p
#    
#    ## Downsample and add inverted activations
#    for act_idx in range(sub_activations_n.shape[0]):
#        
#        flipped_act = da.from_array(np.expand_dims(da.fabs(1-sub_activations_n[act_idx,:].compute()),axis=0))
#        
#        flipped_activations_n = da.concatenate((flipped_activations_n, flipped_act),axis=0)
#    
#    sub_activations_n = flipped_activations_n
#
#          
#    
#    all_activations_p = scores_by_stage_p['1']['activations']
#    all_activations_n = scores_by_stage_n['1']['activations']
#
#    images_p = images_p.permute(2,3,1,0)
#    images_p = images_p.detach().cpu().numpy()
#    image_p = images_p[:,:,:,0]
#    
#    plt.figure()
#    plt.imshow(image_p)
#    
#    plt.figure()`
#    plt.imshow(gt_img_p)
#    
#    images_n = images_n.permute(2,3,1,0)
#    images_n = images_n.detach().cpu().numpy()
#    image_n = images_n[:,:,:,0]
#    
#    plt.figure()
#    plt.imshow(image_n)
#    
#    plt.figure()
#    plt.imshow(gt_img_n)

#    ###########################################################################
#    ########################### Sort Activations ##############################
#    ###########################################################################
#  
#    ## Sort largest to smallest
#    sorted_idx = (-all_importance_weights).argsort() ## indices of features in descending order
#    sorted_fitness_values = all_importance_weights[sorted_idx]
#    indices = [56,23]
#
#    activation_set = all_activations[indices,:,:]
##            
##    for idk in range(activation_set.shape[0]):
##        plt.figure()
##        plt.imshow(activation_set[idk,:,:])
##        plt.axis('off')
##        
##        for ids, ext in enumerate(vgg_idx):
##            if ((ext[0] <= sorted_idx[idk]) and (sorted_idx[idk] <= ext[1])):
##                stage = ids
##        
##        title = 'VGG16 IDX: ' + str(sorted_idx[idk]) + ' | Stage: ' + str(stage+1)
##        plt.title(title)
##        
##        savepath = img_savepath + '/activation_' + str(idk) + '.png'
##        plt.savefig(savepath)
##        plt.close()

#    ###########################################################################
#    ############################ Select Features ##############################
#    ###########################################################################
#    for SOURCES in [4]:
#        new_save_path =  new_directory + '/sort_exhaustive_' + str(SOURCES) + '_sources'
#        
#        parameters.feature_class = 'target'
#        
#        NUM_SOURCES = SOURCES
#   
#        try:
#            os.mkdir(new_save_path)
#            
#            img_savepath = new_save_path
#        except:
#            img_savepath = new_save_path
#    
#    
#        ##########################################################################
#        ### Select Features Divergence between Positive and Negative Features ####
#        ##########################################################################
#        
#        indices_savepath = img_savepath
#        indices = dask_select_features_divergence_pos_and_neg(sub_activations_p, sub_activations_n, NUM_SOURCES, indices_savepath)
        
#    ###########################################################################
#    ############################## Compute ChI ################################
#    ###########################################################################
#    NUM_SOURCES = 4
#    indices = [38, 51, 48, 8]
#    
#    dask_data_p = sub_activations_p[indices,:]
#    dask_labels_p = dask_gt_p.reshape((dask_gt_p.shape[0], dask_gt_p.shape[1]*dask_gt_p.shape[2]))
#    dask_labels_p = dask_labels_p.ravel()
#    
#    dask_data_n = sub_activations_n[indices,:]
#    
#    print('\nComputing Fuzzy Measure...\n')    
#    
#    chi = ChoquetIntegral()    
#    
#    # train the chi via quadratic program 
#    chi.train_chi(dask_data_p[:,0:50000].compute(), dask_labels_p[0:50000].compute())
#    fm = chi.fm
#    
#    fm_savepath = img_savepath + '/fm'
#    np.save(fm_savepath, fm)
#    
#    fm_file = fm_savepath + '.txt'
#    ## Save parameters         
#    with open(fm_file, 'w') as f:
#        json.dump(fm, f, indent=2)
#        f.close 
#            
#    ###########################################################################
#    ############################ Print Equations ##############################
#    ###########################################################################
#    if (NUM_SOURCES < 7):
#        eqt_dict = chi.get_linear_eqts()
#        
#        ## Save equations
#        savename = img_savepath + '/chi_equations.txt'
#        with open(savename, 'w') as f:
#            json.dump(eqt_dict, f, indent=2)
#        f.close 
#        
#    ###########################################################################
#    ######################### Compute Sort Histograms #########################
#    ###########################################################################
#    all_sorts = chi.all_sorts
#    
#    num_samples_p = dask_data_p.shape[1] 
#    num_samples_n = dask_data_n.shape[1] 
#    num_all_sources = dask_data_p.shape[0]
#    
#    ## Used for fast feature selection (Can't use more than 9 sources)
#    encoding_vect = 10**np.arange(0,num_all_sources)[::-1]
#    bin_centers = np.dot(np.array(all_sorts),encoding_vect)
#    hist_bins = [0] + list(bin_centers+1)  
#    
#    ####### Get the sorts histogram for the positive bags #############
#    ## Sort activations for each sample
#    pi_i_p = np.argsort(dask_data_p,axis=0)[::-1,:] + 1 ## Sorts each sample from greatest to least
#
#    ## Get normalized histogram
#    sorts_for_current_activation_p = np.dot(pi_i_p.T,encoding_vect)
#    hist_p, _ = np.histogram(sorts_for_current_activation_p, bins=hist_bins, density=False)
#    hist_p += 1
#    p = hist_p/(num_samples_p + len(all_sorts))
#    
#    ####### Get the sorts histogram for the negative bags #############
#    ## Sort activations for each sample
#    pi_i_n = np.argsort(dask_data_n,axis=0)[::-1,:] + 1 ## Sorts each sample from greatest to least
#
#    ## Get normalized histogram
#    sorts_for_current_activation_n = np.dot(pi_i_n.T,encoding_vect)
#    hist_n, _ = np.histogram(sorts_for_current_activation_n, bins=hist_bins, density=False)
#    hist_n += 1
#    q = hist_n/(num_samples_n + len(all_sorts))
#   
#    ## Define custom colormap
#    custom_cmap = plt.cm.get_cmap('jet', len(all_sorts))
#    norm = matplotlib.colors.Normalize(vmin=1,vmax=len(all_sorts))
#    
#    ## Plot frequency histogram of sorts for single image
#    width = 1
#    hist_ind = np.arange(0.5,len(all_sorts)+0.5)
#    sort_tick_labels = []
#    for element in all_sorts:
#        sort_tick_labels.append(str(element))
#    hist_bins = list(np.arange(0,len(all_sorts)+1)+0.5)
#    plt.figure(figsize=[10,8])
#    ax = plt.axes()
#    plt.xlabel('Sorts', fontsize=14)
#    plt.ylabel('Count', fontsize=14)
#    plt.bar(hist_ind, q, align='edge', facecolor='navy', width=width, label='Negative Bags')
#    plt.bar([i+0.25*width for i in hist_ind], p, align='edge', facecolor='darkorange', width=0.5*width, label='Positive Bags', alpha=0.7)
#    
#    plt.xlim([0.5,len(all_sorts)+0.5])
#    ax.set_xticks(list(np.arange(1,len(all_sorts)+1)))
#    ax.set_xticklabels(sort_tick_labels,rotation=45,ha='right')
#    title = 'Histogram of Sorts'
#    plt.title(title, fontsize=18)
#    plt.legend()
#    savename = new_directory + '/hist_of_sorts_pos.png'
#    plt.savefig(savename)
#    plt.close()
#            
#    ###########################################################################
#    ################################ Test Data ################################
#    ###########################################################################
#    print('Testing positive data...')  
#   
#    sample_idx = 0
#    for data in tqdm(valid_loader):
#        
#        if (parameters.DATASET == 'mnist'):
#            images_p, labels_p = data[0].to(device), data[1].to(device)
#        else:
#            images_p, labels_p, gt_images_p = data[0].to(device), data[1].to(device), data[2]
#            
#        if (int(labels_p.detach().cpu().numpy())==parameters.mnist_target_class):
#    
#            if (parameters.feature_class == 'background'):
#                category = 0
#            elif (parameters.feature_class == 'target'):
#                category = 1
#            elif (parameters.feature_class == 'mnist_class'):
#                category = parameters.mnist_target_class
#            
#            ############# Convert groundtruth image into mask #############
#            if (parameters.DATASET == 'mnist'):
#                gt_image_input_p = images_p.permute(2,3,1,0)
#                gt_image_input_p = gt_image_input_p.detach().cpu().numpy()
#                gt_image_input_p = gt_image_input_p[:,:,:,0]
#                
#                gt_img_p = Image.fromarray(np.uint8(gt_image_input_p*255))
#                gt_images_p = gt_transform(gt_img_p)
#                gt_img_p = np.asarray(gt_images_p)/255
#                
#                gt_img_p[np.where(gt_img_p>0.15)] = 1
#                gt_img_p[np.where(gt_img_p<=0.15)] = 0
#                
#            else:
#                gt_img_p = convert_gt_img_to_mask(gt_images_p, labels_p, parameters)     
#            
#            ## Invert groundtruth image for learning background
#            if (parameters.feature_class == 'background'):
#            
#                zeros_idx = np.where(gt_img_p == 0)
#                ones_idx = np.where(gt_img_p == 1)
#                gt_img_p[zeros_idx] = 1
#                gt_img_p[ones_idx] = 0
#            
#            pred_label_p = int(labels_p.detach().cpu().numpy())
#            
#            ## Normalize input to the model
#            cam_input_p = norm_image(images_p)
#            
#            ########### Get Activation Maps and Model Outputs per Map #########
#            scores_by_stage_p = dict()
#            
#            for stage_idx, activation_model_idx in enumerate(activation_models):
#                
#                ## Get activation maps
#                activations_p, importance_weights_p = activation_models[activation_model_idx](input_tensor=cam_input_p, target_category=1)
#                importance_weights_p = importance_weights_p[0,:].numpy()
#                
##                ## Scale fitness values in each layer to [0,1]
##                importance_weights = importance_weights - np.min(importance_weights)
##                importance_weights = importance_weights / (1e-7 + np.max(importance_weights))
#                
#                current_dict_p = dict()
#                current_dict_p['activations'] = activations_p
##                current_dict['importance_weights'] = importance_weights
##                
##                sorted_idx = (-importance_weights).argsort() ## indices of features in descending order
##                sorted_fitness_values = importance_weights[sorted_idx]
##                
##                current_dict['sorted_idx'] = sorted_idx
##                current_dict['sorted_fitness_values'] = sorted_fitness_values
##       
#                scores_by_stage_p[str(stage_idx+1)] = current_dict_p
#                
#                
#                if not(stage_idx):
#                    all_activations_p = activations_p
#                    all_importance_weights_p = importance_weights_p
#                else:
#                    all_activations_p = np.append(all_activations_p, activations_p, axis=0)
#                    all_importance_weights_p = np.append(all_importance_weights_p, importance_weights_p, axis=0)
#                    
#            
#            sub_activations = all_activations_p[0:64,:,:]
#        
#            flipped_activations = sub_activations
#           
#            ## Downsample and add inverted activations
#            for act_idx in range(sub_activations.shape[0]):
#                
#                flipped_act = np.abs(1-sub_activations[act_idx,:,:])
#                
#                flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
#            
#            sub_activations = flipped_activations
#            
#            all_activations_p = sub_activations
#            
#            
#            ## Convert to dask arrays
#            activations = np.zeros((all_activations_p.shape[0],all_activations_p.shape[1]*all_activations_p.shape[2]))
#            for k in range(all_activations_p.shape[0]):
#                activations[k,:] = all_activations_p[k,:,:].reshape((all_activations_p.shape[1]*all_activations_p.shape[2]))
#            
#            try:
#                pos_savepath =  img_savepath + '/positive_sample_' + str(sample_idx)
#                os.mkdir(pos_savepath)
#            except:
#                pos_savepath =  img_savepath + '/positive_sample_' + str(sample_idx)
#        
#            activation_set_p = all_activations_p[indices,:,:]
#            data_p = activations[indices,:]
#            
#            labels_p = gt_img_p.reshape((gt_img_p.shape[0]*gt_img_p.shape[1]))
#            
#            chi = ChoquetIntegral()    
#    
#            # train the chi via quadratic program 
#            chi.train_chi(data_p, labels_p)
#        
#            sample_idx+=1
#    
#            ## Compute ChI at every pixel location
#            data_out_p = np.zeros((data_p.shape[1]))
#            data_out_sorts_p = []
#            data_out_sort_labels_p = np.zeros((data_p.shape[1]))   
#        
#            all_sorts = chi.all_sorts
#    
#            if (NUM_SOURCES < 7):
#            
#                for idx in range(data_p.shape[1]):
#                    data_out_p[idx], data_out_sort_labels_p[idx] = chi.chi_by_sort(data_p[:,idx])         
#                
#                ## Define custom colormap
#                custom_cmap = plt.cm.get_cmap('jet', len(all_sorts))
#                norm = matplotlib.colors.Normalize(vmin=1,vmax=len(all_sorts))
#                
#                ## Plot frequency histogram of sorts for single image
#                sort_tick_labels = []
#                for element in all_sorts:
#                    sort_tick_labels.append(str(element))
#                hist_bins = list(np.arange(0,len(all_sorts)+1)+0.5)
#                plt.figure(figsize=[10,8])
#                ax = plt.axes()
#                plt.xlabel('Sorts', fontsize=14)
#                plt.ylabel('Count', fontsize=14)
#                n, bins, patches = plt.hist(data_out_sort_labels_p, bins=hist_bins, align='mid')
#                plt.xlim([0.5,len(all_sorts)+0.5])
#                ax.set_xticks(list(np.arange(1,len(all_sorts)+1)))
#                ax.set_xticklabels(sort_tick_labels,rotation=45,ha='right')
#                for idx, current_patch in enumerate(patches):
#                    current_patch.set_facecolor(custom_cmap(norm(idx+1)))
#                title = 'Histogram of Sorts'
#                plt.title(title, fontsize=18)
#                savename = pos_savepath + '/hist_of_sorts_pos.png'
#                plt.savefig(savename)
#                plt.close()
#                
#                ## Plot image colored by sort
#                data_out_sort_labels_p = data_out_sort_labels_p.reshape((activation_set_p.shape[1],all_activations_p.shape[2]))
#                plt.figure(figsize=[10,8])
#                plt.imshow(data_out_sort_labels_p,cmap=custom_cmap)
#                plt.axis('off')
#                plt.colorbar()
#                title = 'Image by Sorts'
#                plt.title(title)
#                savename = pos_savepath + '/img_by_sorts_pos.png'
#                plt.savefig(savename)
#                plt.close()
#                    
#            else:
#                for idx in range(data_p.shape[1]):
#                    data_out_p[idx] = chi.chi_quad(data_p[:,idx])
#                
#            ## Normalize ChI output
#            data_out_p = data_out_p - np.min(data_out_p)
#            data_out_p = data_out_p / (1e-7 + np.max(data_out_p))
#    
#            ###########################################################################
#            ############################## Compute IoU ################################
#            ###########################################################################
#            ## Binarize feature
#            
#            gt_img_p = gt_img_p > 0
#            
#            try:
#                img_thresh_p = threshold_otsu(data_out_p)
#                binary_feature_map_p = data_out_p.reshape((activation_set_p.shape[1],activation_set_p.shape[2])) > img_thresh_p
#            except:
#                binary_feature_map_p = data_out_p < 0.1
#        
#            ## Compute fitness as IoU to pseudo-groundtruth
#            intersection = np.logical_and(binary_feature_map_p, gt_img_p)
#            union = np.logical_or(binary_feature_map_p, gt_img_p)
#            
#            try:
#                iou_score_p = np.sum(intersection) / np.sum(union)
#            except:
#                iou_score_p = 0
#         
#            ###########################################################################
#            ######################### Positive Visualizations #########################
#            ###########################################################################
#            activation_set_p = all_activations_p[indices,:,:]
#            
#            for k in range(activation_set_p.shape[0]):
#                plt.figure()
#                plt.imshow(activation_set_p[k,:,:])
#                plt.axis('off')
#                title = 'Source idx: ' + str(k+1)
#                plt.title(title)
#                
#                savename = pos_savepath + '/source_idx_' + str(k+1) + '_pos.png'
#                plt.savefig(savename)
#                plt.close()
#        
#            data_out_p = data_out_p.reshape((activation_set_p.shape[1],activation_set_p.shape[2]))
#            plt.figure()
#            plt.imshow(data_out_p)
#            plt.axis('off')
#            plt.colorbar()
#            
#            title = 'Raw FusionCAM; IoU: ' + str(iou_score_p)
#            plt.title(title)
#            
#            savename = pos_savepath + '/raw_fusioncam_pos.png'
#                
#            plt.savefig(savename)
#            plt.close()
#        
#            images_p = images_p.permute(2,3,1,0)
#            images_p = images_p.detach().cpu().numpy()
#            image_p = images_p[:,:,:,0]
#        
#            cam_image = show_cam_on_image(image_p, data_out_p, True)
#            
#            plt.figure()
#            plt.imshow(cam_image, cmap='jet')
#          
#            title = 'FusionCAM; IoU: ' + str(iou_score_p)
#            plt.title(title)
#            plt.axis('off')
#            
#            savename = pos_savepath + '/fusioncam_pos.png'
#                
#            plt.savefig(savename)
#            plt.close()
#            
#            if sample_idx > 20:
#                break
#
#
#    ###########################################################################
#    ########################### Test Negative Data ############################
#    ###########################################################################
#    print('Testing negative data...')  
#   
#    sample_idx = 0
#    for data in tqdm(valid_loader):
#        
#        if (parameters.DATASET == 'mnist'):
#            images_p, labels_p = data[0].to(device), data[1].to(device)
#        else:
#            images_p, labels_p, gt_images_p = data[0].to(device), data[1].to(device), data[2]
#            
#        if (int(labels_p.detach().cpu().numpy())==parameters.mnist_background_class):
#    
#            if (parameters.feature_class == 'background'):
#                category = 0
#            elif (parameters.feature_class == 'target'):
#                category = 1
#            elif (parameters.feature_class == 'mnist_class'):
#                category = parameters.mnist_target_class
#            
#            ############# Convert groundtruth image into mask #############
#            if (parameters.DATASET == 'mnist'):
#                gt_image_input_p = images_p.permute(2,3,1,0)
#                gt_image_input_p = gt_image_input_p.detach().cpu().numpy()
#                gt_image_input_p = gt_image_input_p[:,:,:,0]
#                
#                gt_img_p = Image.fromarray(np.uint8(gt_image_input_p*255))
#                gt_images_p = gt_transform(gt_img_p)
#                gt_img_p = np.asarray(gt_images_p)/255
#                
#                gt_img_p[np.where(gt_img_p>0.15)] = 1
#                gt_img_p[np.where(gt_img_p<=0.15)] = 0
#                
#            else:
#                gt_img_p = convert_gt_img_to_mask(gt_images_p, labels_p, parameters)     
#            
#            ## Invert groundtruth image for learning background
#            if (parameters.feature_class == 'background'):
#            
#                zeros_idx = np.where(gt_img_p == 0)
#                ones_idx = np.where(gt_img_p == 1)
#                gt_img_p[zeros_idx] = 1
#                gt_img_p[ones_idx] = 0
#            
#            pred_label_p = int(labels_p.detach().cpu().numpy())
#            
#            ## Normalize input to the model
#            cam_input_p = norm_image(images_p)
#            
#            ########### Get Activation Maps and Model Outputs per Map #########
#            scores_by_stage_p = dict()
#            
#            for stage_idx, activation_model_idx in enumerate(activation_models):
#                
#                ## Get activation maps
#                activations_p, importance_weights_p = activation_models[activation_model_idx](input_tensor=cam_input_p, target_category=1)
#                importance_weights_p = importance_weights_p[0,:].numpy()
#                
##                ## Scale fitness values in each layer to [0,1]
##                importance_weights = importance_weights - np.min(importance_weights)
##                importance_weights = importance_weights / (1e-7 + np.max(importance_weights))
#                
#                current_dict_p = dict()
#                current_dict_p['activations'] = activations_p
##                current_dict['importance_weights'] = importance_weights
##                
##                sorted_idx = (-importance_weights).argsort() ## indices of features in descending order
##                sorted_fitness_values = importance_weights[sorted_idx]
##                
##                current_dict['sorted_idx'] = sorted_idx
##                current_dict['sorted_fitness_values'] = sorted_fitness_values
##       
#                scores_by_stage_p[str(stage_idx+1)] = current_dict_p
#                
#                
#                if not(stage_idx):
#                    all_activations_p = activations_p
#                    all_importance_weights_p = importance_weights_p
#                else:
#                    all_activations_p = np.append(all_activations_p, activations_p, axis=0)
#                    all_importance_weights_p = np.append(all_importance_weights_p, importance_weights_p, axis=0)
#                    
#            
#            sub_activations = all_activations_p[0:64,:,:]
#        
#            flipped_activations = sub_activations
#           
#            ## Downsample and add inverted activations
#            for act_idx in range(sub_activations.shape[0]):
#                
#                flipped_act = np.abs(1-sub_activations[act_idx,:,:])
#                
#                flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
#            
#            sub_activations = flipped_activations
#            
#            all_activations_p = sub_activations
#            
#            
#            ## Convert to dask arrays
#            activations = np.zeros((all_activations_p.shape[0],all_activations_p.shape[1]*all_activations_p.shape[2]))
#            for k in range(all_activations_p.shape[0]):
#                activations[k,:] = all_activations_p[k,:,:].reshape((all_activations_p.shape[1]*all_activations_p.shape[2]))
#            
#            try:
#                pos_savepath =  img_savepath + '/negative_sample_' + str(sample_idx)
#                os.mkdir(pos_savepath)
#            except:
#                pos_savepath =  img_savepath + '/negative_sample_' + str(sample_idx)
#                
#            
#        activation_set_p = all_activations_p[indices,:,:]
#        data_p = activations[indices,:]
#        
#        labels_p = gt_img_p.reshape((gt_img_p.shape[0]*gt_img_p.shape[1]))
#        
##        chi = ChoquetIntegral()    
#
#        # train the chi via quadratic program 
##        chi.train_chi(data_p, labels_p)
#    
#        sample_idx+=1
#
#        ## Compute ChI at every pixel location
#        data_out_p = np.zeros((data_p.shape[1]))
#        data_out_sorts_p = []
#        data_out_sort_labels_p = np.zeros((data_p.shape[1]))   
#    
#        all_sorts = chi.all_sorts
#
#        if (NUM_SOURCES < 7):
#        
#            for idx in range(data_p.shape[1]):
#                data_out_p[idx], data_out_sort_labels_p[idx] = chi.chi_by_sort(data_p[:,idx])         
#            
#            ## Define custom colormap
#            custom_cmap = plt.cm.get_cmap('jet', len(all_sorts))
#            norm = matplotlib.colors.Normalize(vmin=1,vmax=len(all_sorts))
#            
#            ## Plot frequency histogram of sorts for single image
#            sort_tick_labels = []
#            for element in all_sorts:
#                sort_tick_labels.append(str(element))
#            hist_bins = list(np.arange(0,len(all_sorts)+1)+0.5)
#            plt.figure(figsize=[10,8])
#            ax = plt.axes()
#            plt.xlabel('Sorts', fontsize=14)
#            plt.ylabel('Count', fontsize=14)
#            n, bins, patches = plt.hist(data_out_sort_labels_p, bins=hist_bins, align='mid')
#            plt.xlim([0.5,len(all_sorts)+0.5])
#            ax.set_xticks(list(np.arange(1,len(all_sorts)+1)))
#            ax.set_xticklabels(sort_tick_labels,rotation=45,ha='right')
#            for idx, current_patch in enumerate(patches):
#                current_patch.set_facecolor(custom_cmap(norm(idx+1)))
#            title = 'Histogram of Sorts'
#            plt.title(title, fontsize=18)
#            savename = pos_savepath + '/hist_of_sorts_pos.png'
#            plt.savefig(savename)
#            plt.close()
#            
#            ## Plot image colored by sort
#            data_out_sort_labels_p = data_out_sort_labels_p.reshape((activation_set_p.shape[1],all_activations_p.shape[2]))
#            plt.figure(figsize=[10,8])
#            plt.imshow(data_out_sort_labels_p,cmap=custom_cmap)
#            plt.axis('off')
#            plt.colorbar()
#            title = 'Image by Sorts'
#            plt.title(title)
#            savename = pos_savepath + '/img_by_sorts_pos.png'
#            plt.savefig(savename)
#            plt.close()
#                
#        else:
#            for idx in range(data_p.shape[1]):
#                data_out_p[idx] = chi.chi_quad(data_p[:,idx])
#            
#        ## Normalize ChI output
#        data_out_p = data_out_p - np.min(data_out_p)
#        data_out_p = data_out_p / (1e-7 + np.max(data_out_p))
#
#        ###########################################################################
#        ############################## Compute IoU ################################
#        ###########################################################################
#        ## Binarize feature
#        
#        gt_img_p = gt_img_p > 0
#        
#        try:
#            img_thresh_p = threshold_otsu(data_out_p)
#            binary_feature_map_p = data_out_p.reshape((activation_set_p.shape[1],activation_set_p.shape[2])) > img_thresh_p
#        except:
#            binary_feature_map_p = data_out_p < 0.1
#    
#        ## Compute fitness as IoU to pseudo-groundtruth
#        intersection = np.logical_and(binary_feature_map_p, gt_img_p)
#        union = np.logical_or(binary_feature_map_p, gt_img_p)
#        
#        try:
#            iou_score_p = np.sum(intersection) / np.sum(union)
#        except:
#            iou_score_p = 0
#     
#        ###########################################################################
#        ######################### Positive Visualizations #########################
#        ###########################################################################
#        activation_set_p = all_activations_p[indices,:,:]
#        
#        for k in range(activation_set_p.shape[0]):
#            plt.figure()
#            plt.imshow(activation_set_p[k,:,:])
#            plt.axis('off')
#            title = 'Source idx: ' + str(k+1)
#            plt.title(title)
#            
#            savename = pos_savepath + '/source_idx_' + str(k+1) + '_pos.png'
#            plt.savefig(savename)
#            plt.close()
#    
#        data_out_p = data_out_p.reshape((activation_set_p.shape[1],activation_set_p.shape[2]))
#        plt.figure()
#        plt.imshow(data_out_p)
#        plt.axis('off')
#        plt.colorbar()
#        
#        title = 'Raw FusionCAM; IoU: ' + str(iou_score_p)
#        plt.title(title)
#        
#        savename = pos_savepath + '/raw_fusioncam_pos.png'
#            
#        plt.savefig(savename)
#        plt.close()
#    
#        images_p = images_p.permute(2,3,1,0)
#        images_p = images_p.detach().cpu().numpy()
#        image_p = images_p[:,:,:,0]
#    
#        cam_image = show_cam_on_image(image_p, data_out_p, True)
#        
#        plt.figure()
#        plt.imshow(cam_image, cmap='jet')
#      
#        title = 'FusionCAM; IoU: ' + str(iou_score_p)
#        plt.title(title)
#        plt.axis('off')
#        
#        savename = pos_savepath + '/fusioncam_pos.png'
#            
#        plt.savefig(savename)
#        plt.close()
#        
#        if sample_idx > 20:
#            break        
#                
#                
                
            
    