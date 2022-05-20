#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:53:57 2022

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
#from feature_selection_utilities import select_features_sort_entropy, select_features_hag, select_features_exhaustive_entropy, select_features_exhaustive_iou, select_features_divergence_pos_and_neg
#from feature_selection_utilities import dask_select_features_divergence_pos_and_neg
import numpy as np
#from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
from cam_functions.utils.image import show_cam_on_image, preprocess_image
from cam_functions import ActivationCAM, OutputScoreCAM
#from cm_choquet_integral import ChoquetIntegral

#import scipy.io as io

## Custom packages
from mici_feature_selection_parameters import set_parameters as set_parameters
#from cm_MICI.util.cm_mi_choquet_integral import MIChoquetIntegral
#from cm_MICI.util.cm_mi_choquet_integral_dask import MIChoquetIntegral
#from cm_MICI.util.cm_mi_choquet_integral_binary_dask import BinaryMIChoquetIntegral
#import mil_source_selection

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
        
#        parameters.mnist_target_class = 7
#        parameters.mnist_background_class = 1
        
#        target_class = 7
#        background_class = 1
        
        nBagsTrain = 2000
        
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
    
    experiment_savepath = parameters.outf.replace('output','importance_vs_iou')
#    new_save_path =  '/mnist_7_vs_1'
    parameters.feature_class = 'target'
    INCLUDE_INVERTED = False
    
    new_directory = experiment_savepath
    savepath = new_directory
    
    try:
        os.mkdir(new_directory)
        
        savepath = new_directory
    except:
        savepath = new_directory
        
    ###########################################################################
    ############### Extract activation maps and importance weights ############
    ###########################################################################
    
    with tqdm(total=nBagsTrain) as progress_bar:
        
        for tClass in [0,1,2,3,4,5,6,7,8,9]:
            
            parameters.mnist_target_class = tClass
            target_class = tClass
            
            print(f'\n!!!! CLASS {str(tClass)} !!!!\n')
        
        
            sample_idx = 0
            for data in train_loader:
                
                if (parameters.DATASET == 'mnist'):
                    images, labels = data[0].to(device), data[1].to(device)
                else:
                    images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                    
                if (int(labels.detach().cpu().numpy())==parameters.mnist_target_class):
                    
                    pred_label = int(labels.detach().cpu().numpy())
                    
                    ## Normalize input to the model
                    cam_input = norm_image(images)
                    
                    ########### Get Activation Maps and Model Outputs per Map #########
                    scores_by_stage = dict()
                    
                    for stage_idx, activation_model_idx in enumerate(activation_models):
                        
                        ## Get activation maps
                        _, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=target_class)
                        importance_weights = importance_weights[0,:].numpy()
                        
                        if not(stage_idx):
                            all_importance_weights = importance_weights
                        else:
                            all_importance_weights = np.append(all_importance_weights, importance_weights, axis=0)
                    
                    ########### Exctract out the stage we're interested in and invert if desired #########
                    
                    ## Evaluate IoU of feature maps
                    if sample_idx < nBagsTrain:
                        
                        if (not(sample_idx) and not(tClass)):
                            class_vals_mean = np.zeros((10,len(all_importance_weights)))
                            class_vals_std = np.zeros((10,len(all_importance_weights)))
                            current_class_vals = np.zeros((nBagsTrain,len(all_importance_weights)))
                            
                        elif not(sample_idx):
                            current_class_vals = np.zeros((nBagsTrain,len(all_importance_weights)))
                                
                        current_class_vals[sample_idx,:] = all_importance_weights
                            
                        sample_idx += 1
                        progress_bar.update()
                    else:
                        
                        ## Get mean and variance of values for current class
                        current_class_mean = np.mean(current_class_vals,axis=0)
                        current_class_std = np.std(current_class_vals,axis=0)
                        
                        ## Add values to global results
                        class_vals_mean[tClass,:] = current_class_mean
                        class_vals_std[tClass,:] = current_class_std
                        
                        results_savepath = savepath + '/importance_means.npy'   
                        np.save(results_savepath, class_vals_mean, allow_pickle=True)
                        
                        results_savepath = savepath + '/importance_stds.npy'   
                        np.save(results_savepath, class_vals_std, allow_pickle=True)
    
                        progress_bar.reset()
                        
                        break
       
        #######################################################################
        ########################### Save results ##############################
        #######################################################################
        results_savepath = savepath + '/importance_means.npy'   
        np.save(results_savepath, class_vals_mean, allow_pickle=True)
        
        results_savepath = savepath + '/importance_stds.npy'   
        np.save(results_savepath, class_vals_std, allow_pickle=True)
        
        #######################################################################
        ########################### Plot results ##############################
        #######################################################################
        fig, ax = plt.subplots(10,1, figsize=(20,25))
        title = 'CAM Importance Values by Class'
        plt.suptitle(title)
            
        x_range = np.arange(0,class_vals_mean.shape[1])
        
        for tClass in [0,1,2,3,4,5,6,7,8,9]:
        
            y_mean_vect = class_vals_mean[tClass,:]
            y_std_vect = class_vals_std[tClass,:]
            
            ax[tClass].plot(x_range, y_mean_vect,color='blue')
            ax[tClass].fill_between(x_range, y_mean_vect-y_std_vect, y_mean_vect+y_std_vect, color='lightskyblue', alpha=0.5)
                
            ax[tClass].set(ylabel='Importance Value')
            
            title = 'Class ' + str(tClass)
            ax[tClass].set(title=title)
       
        
        ax[-1].set(xlabel='Source Index')
        
        saveto = savepath+'/importance_values.png'
        plt.savefig(saveto)
        plt.close()

        
                
    
   