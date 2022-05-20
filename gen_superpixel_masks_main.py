#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:20:39 2022

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
from skimage.segmentation import slic

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
#from feature_selection_utilities import select_features_sort_entropy, select_features_hag, select_features_exhaustive_entropy, select_features_exhaustive_iou, select_features_divergence_pos_and_neg
#from feature_selection_utilities import dask_select_features_divergence_pos_and_neg
import numpy as np
#from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
from cam_functions.utils.image import show_cam_on_image, preprocess_image
from cam_functions import ActivationCAM, OutputScoreCAM
#from cm_choquet_integral import ChoquetIntegral

#import scipy.io as io

## Custom packages
#from mici_feature_selection_parameters import set_parameters as set_parameters
#from cm_MICI.util.cm_mi_choquet_integral import MIChoquetIntegral
from cm_MICI.util.cm_mi_choquet_integral_dask import MIChoquetIntegral
#from cm_MICI.util.cm_mi_choquet_integral_binary_dask import BinaryMIChoquetIntegral
import mil_source_selection

torch.manual_seed(24)
torch.set_num_threads(1)


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
    
    ######################################################################
    ######################### Input Parameters ###########################
    ######################################################################
    
    parser.add_argument('--DATASET', help='Dataset selection.', default='mnist', type=str)
    parser.add_argument('--DATABASENAME', help='Relative path to the training data list.', default='', type=str)
 
    parser.add_argument('--target_classes', help='List of target classes.', nargs='+', default=['aeroplane'])
    parser.add_argument('--bg_classes', help='List of background classes.', nargs='+', default=['cat'])
    parser.add_argument('--layers', help='Layers to compute CAM at.', nargs='+', default=[4,9,16,23,30])
#    parser.add_argument('--CAM_SEG_THRESH', help='Hard threshold to convert CAM to segmentation decision.', nargs='+', default=[0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9])
    
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
    parser.add_argument('--maxIterations', help='Max number of iteration.', default=500, type=int)
    parser.add_argument('--fitnessThresh', help='Stopping criteria: when fitness change is less than the threshold.', default=0.0001, type=float)
    parser.add_argument('--eta', help='Percentage of time to make small-scale mutation.', default=0.5, type=float)
    parser.add_argument('--sampleVar', help='Variance around sample mean.', default=0.1, type=float)
    parser.add_argument('--mean', help='Mean of ci in fitness function. Is always set to 1 if the positive label is "1".', default=1.0, type=float)
    parser.add_argument('--analysis', action='store_true', help='If ="1", record all intermediate results.', default=False)
    parser.add_argument('--p', help='p power value for the softmax. p(1) should->Inf, p(2) should->-Inf.', nargs='+', default=[10,-10])
    parser.add_argument('--use_parallel', action='store_true', help='If ="1", use parallel processing for population sampling.', default=True)
    
    ## Parameters for binary fuzzy measure sampling
    parser.add_argument('--U', help='How many times we are willing to sample a new measure before deeming the search exhaustive.', default=500, type=int)
    parser.add_argument('--Q', help='How many times the best fitness value can remain unchanged before stopping.', default=2, type=int)
    
    return parser.parse_args(args)

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
        
        parameters.mnist_target_class = 8
        parameters.mnist_background_class = 3
        
        target_class = 8
        background_class = 3
        
        nPBagsTrain = 2
        nNBagsTrain = 2
        nPBagsTest = 20
        nNBagsTest = 20

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
    
###############################################################################
###############################################################################
###############################################################################
    
    experiment_savepath = parameters.outf.replace('output','superpixel_masks')
    new_save_path =  '/mici_genmean_7_versus_1'
    parameters.feature_class = 'target'
    INCLUDE_INVERTED = False
    
    Bags = []
    Labels = []
    GT = []
    
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
    sample_idx = 0
    for data in tqdm(train_loader):
        
        if (parameters.DATASET == 'mnist'):
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
            
        if (int(labels.detach().cpu().numpy())==parameters.mnist_target_class):
    
#            if (parameters.feature_class == 'background'):
#                category = 0
#            elif (parameters.feature_class == 'target'):
#                category = 1
#            elif (parameters.feature_class == 'mnist_class'):
#                category = parameters.mnist_target_class
            
            ############# Convert groundtruth image into mask #############
            if (parameters.DATASET == 'mnist'):
                gt_image_input = images.permute(2,3,1,0)
                gt_image_input = gt_image_input.detach().cpu().numpy()
                gt_image_input = gt_image_input[:,:,:,0]
                
                gt_img = Image.fromarray(np.uint8(gt_image_input*255))
                gt_images = gt_transform(gt_img)
                gt_img = np.asarray(gt_images)/255
                
                gt_img[np.where(gt_img>0.15)] = 1
                gt_img[np.where(gt_img<=0.15)] = 0
                
            else:
                gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)     
            
            ## Invert groundtruth image for learning background
            if (parameters.feature_class == 'background'):
            
                zeros_idx = np.where(gt_img == 0)
                ones_idx = np.where(gt_img == 1)
                gt_img[zeros_idx] = 1
                gt_img[ones_idx] = 0
            
            pred_label = int(labels.detach().cpu().numpy())
            
            ## Normalize input to the model
            cam_input = norm_image(images)
            
            ########### Get Activation Maps and Model Outputs per Map #########
            scores_by_stage = dict()
            
            for stage_idx, activation_model_idx in enumerate(activation_models):
                
                ## Get activation maps
                activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=target_class)
                importance_weights = importance_weights[0,:].numpy()
                
                current_dict = dict()
                current_dict['activations'] = activations
                scores_by_stage[str(stage_idx+1)] = current_dict
                
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
            
            ########### Exctract out the stage we're interested in and invert if desired #########
#            all_activations = all_activations[0:64,:,:]
            all_activations = all_activations[scores_by_stage['1']['sorted_idx'][0:10],:,:]
            
            if INCLUDE_INVERTED:
                # Flip activations
                flipped_activations = all_activations
           
                ## Downsample and add inverted activations
                for act_idx in range(flipped_activations.shape[0]):
                    
                    flipped_act = np.abs(1-flipped_activations[act_idx,:,:])
                    flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
    
                all_activations = flipped_activations

            ########################## Convert to dask arrays ####################################
            activations = np.zeros((all_activations.shape[1]*all_activations.shape[2],all_activations.shape[0]))
            for k in range(all_activations.shape[0]):
                activations[:,k] = all_activations[k,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
            
            if sample_idx < nPBagsTrain:
                if not(sample_idx):
                    
                    pBags = np.zeros((nPBagsTrain,),dtype=np.object)
                    pLabels = np.ones((nPBagsTrain,),dtype=np.uint8)
                    pBagGT = np.zeros((nPBagsTrain,),dtype=np.object)
                    
                    pBags[sample_idx] = da.from_array(activations)
                    pBagGT[sample_idx] = da.from_array(np.expand_dims(gt_img,axis=0))
                    
                    sample_idx += 1
                else:
                    
                    pBags[sample_idx] = da.from_array(activations)
                    pBagGT[sample_idx] = da.from_array(np.expand_dims(gt_img,axis=0))

                    sample_idx += 1
            else:
                break
            
    print('\nExtracting NEGATIVE bag activation maps...\n')
    sample_idx = 0
    for data in tqdm(train_loader):
        
        if (parameters.DATASET == 'mnist'):
            images, labels = data[0].to(device), data[1].to(device)
        else:
            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
            
        if (int(labels.detach().cpu().numpy())==parameters.mnist_background_class):
    
#            if (parameters.feature_class == 'background'):
#                category = 0
#            elif (parameters.feature_class == 'target'):
#                category = 1
#            elif (parameters.feature_class == 'mnist_class'):
#                category = parameters.mnist_target_class
            
            ############# Convert groundtruth image into mask #############
            if (parameters.DATASET == 'mnist'):
                gt_image_input = images.permute(2,3,1,0)
                gt_image_input = gt_image_input.detach().cpu().numpy()
                gt_image_input = gt_image_input[:,:,:,0]
                
                gt_img = Image.fromarray(np.uint8(gt_image_input*255))
                gt_images = gt_transform(gt_img)
                gt_img = np.asarray(gt_images)/255
                
                gt_img[np.where(gt_img>0.15)] = 1
                gt_img[np.where(gt_img<=0.15)] = 0
                
            else:
                gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)     
            
            ## Invert groundtruth image for learning background
            if (parameters.feature_class == 'background'):
            
                zeros_idx = np.where(gt_img == 0)
                ones_idx = np.where(gt_img == 1)
                gt_img[zeros_idx] = 1
                gt_img[ones_idx] = 0
            
            pred_label = int(labels.detach().cpu().numpy())
            
            ## Normalize input to the model
            cam_input = norm_image(images)
            
            ########### Get Activation Maps and Model Outputs per Map #########
            scores_by_stage = dict()
            
            for stage_idx, activation_model_idx in enumerate(activation_models):
                
                ## Get activation maps
                activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=target_class)
                importance_weights = importance_weights[0,:].numpy()
                
                current_dict = dict()
                current_dict['activations'] = activations
                scores_by_stage[str(stage_idx+1)] = current_dict
                
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
            
            ########### Exctract out the stage we're interested in and invert if desired #########
#            all_activations = all_activations[0:64,:,:]
            all_activations = all_activations[scores_by_stage['1']['sorted_idx'][0:10],:,:]
            
            if INCLUDE_INVERTED:
                # Flip activations
                flipped_activations = all_activations
           
                ## Downsample and add inverted activations
                for act_idx in range(flipped_activations.shape[0]):
                    
                    flipped_act = np.abs(1-flipped_activations[act_idx,:,:])
                    flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
    
                all_activations = flipped_activations

            ########################## Convert to dask arrays ####################################
            activations = np.zeros((all_activations.shape[1]*all_activations.shape[2],all_activations.shape[0]))
            for k in range(all_activations.shape[0]):
                activations[:,k] = all_activations[k,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
            
            if sample_idx < nNBagsTrain:
                if not(sample_idx):
                    
                    nBags = np.zeros((nPBagsTrain,),dtype=np.object)
                    nLabels = np.zeros((nPBagsTrain,),dtype=np.uint8)
                    nBagGT = np.zeros((nPBagsTrain,),dtype=np.object)
                    
                    nBags[sample_idx] = da.from_array(activations)
                    nBagGT[sample_idx] = da.from_array(np.expand_dims(gt_img,axis=0))
                    
                    sample_idx += 1
                else:
                    
                    nBags[sample_idx] = da.from_array(activations)
                    nBagGT[sample_idx] = da.from_array(np.expand_dims(gt_img,axis=0))

                    sample_idx += 1
            else:
                break
            
    ########################## Concatenate Information ############################
    Bags = np.concatenate((nBags,pBags))
    Labels = np.concatenate((nLabels, pLabels))
    GT = np.concatenate((nBagGT, pBagGT))
    
    if (parameters.run_mode == 'select_sources'):
        
        ###########################################################################
        ############################ Select Sources ###############################
        ###########################################################################
        
        for SOURCES in [4]:
            new_save_path =  new_directory + '/mici_min_max_binary_measure_stage_1_' + str(SOURCES) + '_sources'
            
            parameters.feature_class = 'target'
            
            NUM_SOURCES = SOURCES
       
            try:
                os.mkdir(new_save_path)
                
                img_savepath = new_save_path
            except:
                img_savepath = new_save_path
        
        
            ##########################################################################
            ### Select Features Divergence between Positive and Negative Features ####
            ##########################################################################
            
            indices_savepath = img_savepath
            indices = mil_source_selection.select_mici_binary_measure(Bags, Labels, NUM_SOURCES, indices_savepath, parameters)
        
            print('\n!!!SOURCES SELECTED!!!')
        ###########################################################################
        ######################### Train on Selected Sources #######################
        ###########################################################################
        
        ## Sample from selected sources
#        indices = [4,5,6,8]
        tempBags = np.zeros((len(Bags)),dtype=np.object)
        for idb, bag in enumerate(Bags):
            tempbag = bag[:,indices]
            tempBags[idb] = tempbag
            
        ###########################################################################
        ############################## Train MICI #################################
        ###########################################################################
        print('\n Training MICI Generalized-Mean... \n')
        
        parameters.use_parallel = False
        
        ## Initialize Choquet Integral instance
        chi = MIChoquetIntegral()
        
        # Training Stage: Learn measures given training bags and labels
        chi.train_chi_softmax(tempBags, Labels, parameters) ## generalized-mean model
        
        ###########################################################################
        ######################## Extract Learned Measure ##########################
        ###########################################################################
        
        fm = chi.fm
        
        fm_savepath = img_savepath + '/fm'
        np.save(fm_savepath, fm)
        
        fm_file = fm_savepath + '.txt'
        ## Save parameters         
        with open(fm_file, 'w') as f:
            json.dump(fm, f, indent=2)
            f.close 
                
#        ##########################################################################
#        ########################### Print Equations ##############################
#        ##########################################################################
#    
#        eqt_dict = chi.get_linear_eqts()
#            
#        ## Save equations
#        savename = img_savepath + '/chi_equations.txt'
#        with open(savename, 'w') as f:
#            json.dump(eqt_dict, f, indent=2)
#        f.close 
    
    elif (parameters.run_mode == 'test'):
        ###########################################################################
        ############################### Test MICI #################################
        ###########################################################################
        
        indices = [4, 5, 6, 8]
        
        chi = MIChoquetIntegral()
        
#        del Bags
#        del Labels
        
        new_save_path =  new_directory + '/mici_min_max_binary_measure_stage_1_' + str(4) + '_sources'
            
        parameters.feature_class = 'target'
       
        try:
            os.mkdir(new_save_path)
            
            img_savepath = new_save_path
        except:
            img_savepath = new_save_path
            
#        measure_path = new_save_path + '/fm.npy'
#        measure = np.load(measure_path, allow_pickle=True).item()
#        chi.fm = measure
        
        
#        results_path = new_save_path + '/results.npy'
#        results = np.load(results_path, allow_pickle=True).item()
#        chi.fm = results['fm']
#        chi.measure = ['measure']
        
        ## 7 vs 1 pos target class for pos and neg CAM
        measure = np.array([0.0521, 0.0408, 0.0378, 0.18105, 0.62718, 0.9199, 0.7004, 0.0618, 0.18359, 0.40604, 0.92228, 0.992003, 0.99372, 0.43601, 1.0])
        chi.measure = measure
        
#        ## 7 vs 1 neg target class for neg CAM
#        measure = np.array([0.0187, 0.0113, 0.057, 0.0517, 0.1204, 0.0915, 0.9515, 0.0745, 0.9353, 0.1242, 0.5868, 0.9751, 0.9938, 0.9762, 1.0])
#        chi.measure = measure
        
        
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
        ###########################################################################
        ################################ Test Data ################################
        ###########################################################################
        print('Testing positive data...')  
       
        sample_idx = 0
        for data in tqdm(valid_loader):
            
            if (parameters.DATASET == 'mnist'):
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                
            if (int(labels.detach().cpu().numpy())==parameters.mnist_target_class):
                
                if sample_idx < nPBagsTest:
        
#                    if (parameters.feature_class == 'background'):
#                        category = 0
#                    elif (parameters.feature_class == 'target'):
#                        category = 1
#                    elif (parameters.feature_class == 'mnist_class'):
#                        category = parameters.mnist_target_class
                    
                    ############# Convert groundtruth image into mask #############
                    if (parameters.DATASET == 'mnist'):
                        gt_image_input = images.permute(2,3,1,0)
                        gt_image_input = gt_image_input.detach().cpu().numpy()
                        gt_image_input = gt_image_input[:,:,:,0]
                        
                        gt_img = Image.fromarray(np.uint8(gt_image_input*255))
                        gt_images = gt_transform(gt_img)
                        gt_img = np.asarray(gt_images)/255
                        
                        gt_img[np.where(gt_img>0.15)] = 1
                        gt_img[np.where(gt_img<=0.15)] = 0
                        
                    else:
                        gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)     
                    
                    ## Invert groundtruth image for learning background
                    if (parameters.feature_class == 'background'):
                    
                        zeros_idx = np.where(gt_img == 0)
                        ones_idx = np.where(gt_img == 1)
                        gt_img[zeros_idx] = 1
                        gt_img[ones_idx] = 0
                    
                    pred_label = int(labels.detach().cpu().numpy())
                    
                    ## Normalize input to the model
                    cam_input = norm_image(images)
                    
                    ########### Get Activation Maps and Model Outputs per Map #########
                    scores_by_stage = dict()
                    
                    for stage_idx, activation_model_idx in enumerate(activation_models):
                        
                        ## Get activation maps
                        activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=target_class)
                        importance_weights = importance_weights[0,:].numpy()
                        
                        current_dict = dict()
                        current_dict['activations'] = activations
                        scores_by_stage[str(stage_idx+1)] = current_dict
                        
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
                    
                    ########### Exctract out the stage we're interested in and invert if desired #########
        #            all_activations = all_activations[0:64,:,:]
                    all_activations = all_activations[scores_by_stage['1']['sorted_idx'][0:10],:,:]
                    
                    if INCLUDE_INVERTED:
                        # Flip activations
                        flipped_activations = all_activations
                   
                        ## Downsample and add inverted activations
                        for act_idx in range(flipped_activations.shape[0]):
                            
                            flipped_act = np.abs(1-flipped_activations[act_idx,:,:])
                            flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
            
                        all_activations = flipped_activations
        
                    ########################## Convert to dask arrays ####################################
                    activations = np.zeros((all_activations.shape[1]*all_activations.shape[2],all_activations.shape[0]))
                    for k in range(all_activations.shape[0]):
                        activations[:,k] = all_activations[k,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
                
                    activations = activations[:,indices]
                
                    Bags = np.zeros((1,),dtype=np.object)
#                    Bags[0] = activations
                    Bags[0] = da.from_array(activations)
                    
                    sample_idx += 1
                        
                    try:
                        pos_savepath =  img_savepath + '/positive_sample_' + str(sample_idx)
                        os.mkdir(pos_savepath)
                    except:
                        pos_savepath =  img_savepath + '/positive_sample_' + str(sample_idx)
            
                    ## Compute ChI on the bag
                    data_out = chi.compute_chi(Bags,1,chi.measure)
                    
                    ## Normalize ChI output
                    data_out = data_out - np.min(data_out)
                    data_out = data_out / (1e-7 + np.max(data_out))
                    
                    Bag = Bags[0].compute()
            
                    ###########################################################################
                    ############################## Compute IoU ################################
                    ###########################################################################
                    ## Binarize feature
                    
                    gt_img = gt_img > 0
                    
                    try:
                        img_thresh = threshold_otsu(data_out)
                        binary_feature_map = data_out.reshape((data_out.shape[1],data_out.shape[2])) > img_thresh
                    except:
                        binary_feature_map = data_out < 0.1
                
                    ## Compute fitness as IoU to pseudo-groundtruth
                    intersection = np.logical_and(binary_feature_map, data_out)
                    union = np.logical_or(binary_feature_map, data_out)
                    
                    try:
                        iou_score = np.sum(intersection) / np.sum(union)
                    except:
                        iou_score = 0
                        
                    plt.figure()
                    plt.imshow(binary_feature_map.reshape((gt_img.shape[0],gt_img.shape[1])))
                    title = 'Otsu Thresholded Image: t=' + str(img_thresh) + '; IoU: ' + str(iou_score)
                    plt.title(title)
                        
                    images = images.permute(2,3,1,0)
                    images = images.detach().cpu().numpy()
                    image = images[:,:,:,0]    
                    
                    ## Get SLIC segmented image
                    
                    sp_mask = np.zeros((gt_img.shape[0],gt_img.shape[1]))
                    
                    spLabels = slic(image).reshape((image.shape[0]*image.shape[1]))
                    num_sp = np.unique(spLabels)
                    
                    for lab in num_sp:
                        sp_ind = np.where(spLabels == lab)
                        sp_val = np.sum(data_out[sp_ind])/len(data_ou[sp_ind])
                    
                    plt.figure()
                    plt.imshow(binary_feature_map)
                    title = 'SLIC Thresholded Image: t=' + str(img_thresh) + '; IoU: ' + str(iou_score)
                    plt.title(title)
                 
                    ###########################################################################
                    ######################### Positive Visualizations #########################
                    ###########################################################################
                    
                    for k in range(Bag.shape[1]):
                        plt.figure()
                        plt.imshow(Bag[:,k].reshape((all_activations.shape[1],all_activations.shape[2])))
                        plt.axis('off')
                        title = 'Source idx: ' + str(k+1)
                        plt.title(title)
                        
                        savename = pos_savepath + '/source_idx_' + str(k+1) + '_pos.png'
                        plt.savefig(savename)
                        plt.close()
                
                    data_out = data_out.reshape((all_activations.shape[1],all_activations.shape[2]))
                    plt.figure()
                    plt.imshow(data_out)
                    plt.axis('off')
                    plt.colorbar()
                    
                    title = 'Raw FusionCAM; IoU: ' + str(iou_score)
                    plt.title(title)
                    
                    savename = pos_savepath + '/raw_fusioncam_pos.png'
                        
                    plt.savefig(savename)
                    plt.close()
                
#                    images = images.permute(2,3,1,0)
#                    images = images.detach().cpu().numpy()
#                    image = images[:,:,:,0]
                
                    cam_image = show_cam_on_image(image, data_out, True)
                    
                    plt.figure()
                    plt.imshow(cam_image, cmap='jet')
                  
                    title = 'FusionCAM; IoU: ' + str(iou_score)
                    plt.title(title)
                    plt.axis('off')
                    
                    savename = pos_savepath + '/fusioncam_pos.png'
                        
                    plt.savefig(savename)
                    plt.close()
                    
                else:
                    break
        ###########################################################################
        ########################## Test Negative Data #############################
        ###########################################################################
        print('Testing negative data...')  
       
        sample_idx = 0
        for data in tqdm(valid_loader):
            
            if (parameters.DATASET == 'mnist'):
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                
            if (int(labels.detach().cpu().numpy())==parameters.mnist_background_class):
                
                if sample_idx < nPBagsTest:
        
#                    if (parameters.feature_class == 'background'):
#                        category = 0
#                    elif (parameters.feature_class == 'target'):
#                        category = 1
#                    elif (parameters.feature_class == 'mnist_class'):
#                        category = parameters.mnist_target_class
                    
                    ############# Convert groundtruth image into mask #############
                    if (parameters.DATASET == 'mnist'):
                        gt_image_input = images.permute(2,3,1,0)
                        gt_image_input = gt_image_input.detach().cpu().numpy()
                        gt_image_input = gt_image_input[:,:,:,0]
                        
                        gt_img = Image.fromarray(np.uint8(gt_image_input*255))
                        gt_images = gt_transform(gt_img)
                        gt_img = np.asarray(gt_images)/255
                        
                        gt_img[np.where(gt_img>0.15)] = 1
                        gt_img[np.where(gt_img<=0.15)] = 0
                        
                    else:
                        gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)     
                    
                    ## Invert groundtruth image for learning background
                    if (parameters.feature_class == 'background'):
                    
                        zeros_idx = np.where(gt_img == 0)
                        ones_idx = np.where(gt_img == 1)
                        gt_img[zeros_idx] = 1
                        gt_img[ones_idx] = 0
                    
                    pred_label = int(labels.detach().cpu().numpy())
                    
                    ## Normalize input to the model
                    cam_input = norm_image(images)
                    
                    ########### Get Activation Maps and Model Outputs per Map #########
                    scores_by_stage = dict()
                    
                    for stage_idx, activation_model_idx in enumerate(activation_models):
                        
                        ## Get activation maps
                        activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=target_class)
                        importance_weights = importance_weights[0,:].numpy()
                        
                        current_dict = dict()
                        current_dict['activations'] = activations
                        scores_by_stage[str(stage_idx+1)] = current_dict
                        
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
                    
                    ########### Exctract out the stage we're interested in and invert if desired #########
        #            all_activations = all_activations[0:64,:,:]
                    all_activations = all_activations[scores_by_stage['1']['sorted_idx'][0:10],:,:]
                    
                    if INCLUDE_INVERTED:
                        # Flip activations
                        flipped_activations = all_activations
                   
                        ## Downsample and add inverted activations
                        for act_idx in range(flipped_activations.shape[0]):
                            
                            flipped_act = np.abs(1-flipped_activations[act_idx,:,:])
                            flipped_activations = np.concatenate((flipped_activations, np.expand_dims(flipped_act,axis=0)),axis=0)
            
                        all_activations = flipped_activations
        
                    ########################## Convert to dask arrays ####################################
                    activations = np.zeros((all_activations.shape[1]*all_activations.shape[2],all_activations.shape[0]))
                    for k in range(all_activations.shape[0]):
                        activations[:,k] = all_activations[k,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
                
                    activations = activations[:,indices]
                
                    Bags = np.zeros((1,),dtype=np.object)
#                    Bags[0] = activations
                    Bags[0] = da.from_array(activations)
                    
                    sample_idx += 1
                        
                    try:
                        pos_savepath =  img_savepath + '/negative_sample_' + str(sample_idx)
                        os.mkdir(pos_savepath)
                    except:
                        pos_savepath =  img_savepath + '/negative_sample_' + str(sample_idx)
            
                    ## Compute ChI on the bag
                    data_out = chi.compute_chi(Bags,1,chi.measure)
                    
                    ## Normalize ChI output
                    data_out = data_out - np.min(data_out)
                    data_out = data_out / (1e-7 + np.max(data_out))
                    
                    Bag = Bags[0].compute()
            
                    ###########################################################################
                    ############################## Compute IoU ################################
                    ###########################################################################
                    ## Binarize feature
                    
                    gt_img = gt_img > 0
                    
                    try:
                        img_thresh = threshold_otsu(data_out)
                        binary_feature_map = data_out.reshape((data_out.shape[1],data_out.shape[2])) > img_thresh
                    except:
                        binary_feature_map = data_out < 0.1
                
                    ## Compute fitness as IoU to pseudo-groundtruth
                    intersection = np.logical_and(binary_feature_map, data_out)
                    union = np.logical_or(binary_feature_map, data_out)
                    
                    try:
                        iou_score = np.sum(intersection) / np.sum(union)
                    except:
                        iou_score = 0
                 
                    ###########################################################################
                    ######################### Positive Visualizations #########################
                    ###########################################################################
                    
                    for k in range(Bag.shape[1]):
                        plt.figure()
                        plt.imshow(Bag[:,k].reshape((all_activations.shape[1],all_activations.shape[2])))
                        plt.axis('off')
                        title = 'Source idx: ' + str(k+1)
                        plt.title(title)
                        
                        savename = pos_savepath + '/source_idx_' + str(k+1) + '_neg.png'
                        plt.savefig(savename)
                        plt.close()
                
                    data_out = data_out.reshape((all_activations.shape[1],all_activations.shape[2]))
                    plt.figure()
                    plt.imshow(data_out)
                    plt.axis('off')
                    plt.colorbar()
                    
                    title = 'Raw FusionCAM; IoU: ' + str(iou_score)
                    plt.title(title)
                    
                    savename = pos_savepath + '/raw_fusioncam_neg.png'
                        
                    plt.savefig(savename)
                    plt.close()
                
                    images = images.permute(2,3,1,0)
                    images = images.detach().cpu().numpy()
                    image = images[:,:,:,0]
                
                    cam_image = show_cam_on_image(image, data_out, True)
                    
                    plt.figure()
                    plt.imshow(cam_image, cmap='jet')
                  
                    title = 'FusionCAM; IoU: ' + str(iou_score)
                    plt.title(title)
                    plt.axis('off')
                    
                    savename = pos_savepath + '/fusioncam_neg.png'
                        
                    plt.savefig(savename)
                    plt.close()
                    
                else:
                    break

                
                
            
    