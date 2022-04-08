#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:19:29 2022

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
    *  Latest Revision: April 22, 2021
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

## General packages
import os
import json
import argparse
import itertools
from tqdm import tqdm
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import precision_recall_fscore_support as prfs
from skimage.filters import threshold_otsu

# PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

## Custom packages
import cam_fusion_parameters
import initialize_network
from util import convert_gt_img_to_mask, cam_img_to_seg_img
from dataloaders import loaderPASCAL, loaderDSIAC

import numpy as np
import matplotlib.pyplot as plt
from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
from cam_functions.utils.image import show_cam_on_image, preprocess_image
from cam_functions import ActivationCAM
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
    SAMPLE_IDX = 5
    TOP_K = 7
    BOTTOM_K = 1
    NUM_SUBSAMPLE = 5000
    
   
    
    ## Import parameters 
    args = None
    parameters = cam_fusion_parameters.set_parameters(args)
    
    ## Define data transforms
    if (parameters.DATASET == 'dsiac'):
        transform = transforms.Compose([transforms.ToTensor()])
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
    
    norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ## Turn on gradients for CAM computation 
    for param in model.parameters():
        param.requires_grad = True

    model.eval()             
            
    if (parameters.model == 'vgg16'):
        activation_model = ActivationCAM(model=model, target_layers=[model.features[4]], use_cuda=parameters.cuda)
        gradcam_model = GradCAM(model=model, target_layers=[model.features[30]], use_cuda=parameters.cuda)

    target_category = None
    
    experiment_savepath = parameters.outf.replace('output','cam_fusion')
    
    
############################### View Activations ##############################
#    
#    new_directory = experiment_savepath + new_save_path
#    
#    sample_idx = 0
#    for data in tqdm(test_loader):
#        
#        if (sample_idx==5):
#
#            images, labels = data[0].to(device), data[1].to(device)
#            
#            ###############################################################
#            ################# Extract activation maps #####################
#            ###############################################################
#            
#            try:
#                make_dir = new_directory + '/layer' + str(5)
#                os.mkdir(make_dir)
#                
#                img_savepath = make_dir
#            except:
#                img_savepath = make_dir
#            
#            ## Normalize input to the model
#            cam_input = norm_image(images)
#            
#            ## Get activation maps
#            activations, _ = activation_model(input_tensor=cam_input, target_category=int(labels))
#            
#            for idx in range(activations.shape[0]):
#                
#                savename = img_savepath + '/activation_' + str(idx) + '.png'
#                
#                x = activations[idx,:,:]
#                
#                plt.figure()
#                plt.imshow(x)
#                plt.axis('off')
#                
#                plt.savefig(savename)
#                
#                plt.close()
#            
#            break
#        
#        else:
#            sample_idx+=1
    

########################### Train/Test Same Image ############################# 
    
#    ###########################################################################
#    ############### Extract activation maps and importance weights ############
#    ###########################################################################
#    sample_idx = 0
#    for data in tqdm(test_loader):
#        
##        if (idx==SAMPLE_IDX):
#        if (sample_idx<101):
#            
#            print(str(sample_idx))
#            
#            if GEN_PLOTS:
#                try:
#                    new_directory = experiment_savepath + new_save_path + '/img_' + str(sample_idx) 
#                    os.mkdir(new_directory)
#                    
#                    img_savepath = new_directory
#                except:
#                    img_savepath = new_directory
#        
#            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
#            
#            ############# Convert groundtruth image into mask #############
#            gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
#            
#            ## Normalize input to the model
#            cam_input = norm_image(images)
#            
#            ##################### Get Activation Maps ####################
#            activations, importance_weights = activation_model(input_tensor=cam_input, target_category=int(labels))
#            importance_weights = importance_weights[0,:]
#            
#            #################### Plot Activation Maps ####################
#            
#            images = images.permute(2,3,1,0)
#            images = images.detach().cpu().numpy()
#            image = images[:,:,:,0]
#            img = image
#            
##            if GEN_PLOTS:
##                plt.figure()
##                plt.imshow(img)
##                plt.axis('off')
##                
##                savename = img_savepath + '/input.png'
##                
##                plt.savefig(savename)
##                plt.close()
##                
##                for k, img in enumerate(activations):
##                        plt.figure()
##                        plt.imshow(img)
##                        plt.axis('off')
##                        
##                        title = 'Activation: ' + str(k) + ', Importance: ' + str(importance_weights[k])
##                        
##                        plt.title(title)
##                        
##                        savename = img_savepath + '/all_activations/activation_idx_' + str(k) + '.png'
##                
##                        plt.savefig(savename)
##                        plt.close()
#                    
##            break
#        
##        sample_idx +=1
#    
#            ###########################################################################
#            ################# Get most and least important activation maps ############
#            ###########################################################################
#            
#            ## Get most important and least important activations
#            top_k_activations_idx = (-importance_weights).argsort()[:TOP_K]
#            bottom_k_activations_idx = (importance_weights).argsort()[:BOTTOM_K]
#            
#            top_k_activations = activations[top_k_activations_idx]
#            bottom_k_activations = activations[bottom_k_activations_idx]
#            
#            top_k_weights = importance_weights[top_k_activations_idx]
#            bottom_k_weights = importance_weights[bottom_k_activations_idx]
#              
#            ###########################################################################
#            ############################## Generate Plots #############################
#            ###########################################################################
#                  
#            if GEN_PLOTS:
#                for k, img in enumerate(top_k_activations):
#                            plt.figure()
#                            plt.imshow(img)
#                            plt.axis('off')
#        #                    plt.colorbar()
#                            
#                            title = 'Top Activation: ' + str(k) + ', Importance: ' + str(top_k_weights[k])
#                            
#                            plt.title(title)
#                            
#                            savename = img_savepath + '/top_activation_idx_' + str(k) + '.png'
#                    
#                            plt.savefig(savename)
#                            plt.close()
#                            
#                for k, img in enumerate(bottom_k_activations):
#                            plt.figure()
#                            plt.imshow(img)
#                            plt.axis('off')
#        #                    plt.colorbar()
#                            
#                            title = 'Bottom Activation: ' + str(k) + ', Importance: ' + str(bottom_k_weights[k])
#                            
#                            plt.title(title)
#                            
#                            savename = img_savepath + '/bottom_activation_idx_' + str(k) + '.png'
#                    
#                            plt.savefig(savename)
#                            plt.close()
#            
#            
#                ## Plot GradCAM results
#                grayscale_cam = gradcam_model(input_tensor=cam_input,target_category=int(labels))
#                grayscale_cam = grayscale_cam[0, :]
#             
#                
#                gt_images = gt_images.detach().cpu().numpy()
#                gt_image = gt_images[0,:,:]
#                
#                if(labels.detach().cpu().numpy()==0):
#                    for key in parameters.bg_classes:
#                        value = parameters.all_classes[key]
#                        gt_image[np.where(gt_image==value)] = 1
#                        
#                elif(labels.detach().cpu().numpy()==1):
#                    for key in parameters.target_classes:
#                        value = parameters.all_classes[key]
#                        gt_image[np.where(gt_image==value)] = 1
#                
#                ## Convert segmentation image to mask
#                gt_image[np.where(gt_image!=1)] = 0
#                
#                ## Show image and groundtruth
#                plt.figure()
#                plt.imshow(gt_image, cmap='gray')
#                plt.axis('off')
#                
#                savename = img_savepath + '/grountruth_mask.png'
#                
#                plt.savefig(savename)
#                plt.close()
#            
#                ## Plot GradCAM
#                plt.figure()
#                cam_image = show_cam_on_image(image, grayscale_cam, True)
#                plt.imshow(cam_image, cmap='jet')
#                plt.axis('off')
#                plt.title('GradCAM')
#                
#                savename = img_savepath + '/gradcam.png'
#                
#                plt.savefig(savename)
#                plt.close()
#    
#                
#            ###########################################################################
#            ############################## Compute ChI ################################
#            ###########################################################################
#            
#            
#            all_activations = np.append(top_k_activations, bottom_k_activations, axis=0)
#            
#            ########################## Pixel-level fusion #############################
#            labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
#            
#            data = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
#            
#            for idx in range(all_activations.shape[0]):
#                data[idx,:] = np.reshape(activations[idx,:,:],(1,activations[idx,:,:].shape[0]*activations[idx,:,:].shape[1]))
#                
##            print('Computing Fuzzy Measure...')    
#            
#            chi = ChoquetIntegral()    
#            
#            # train the chi via quadratic program 
#            chi.train_chi(data, labels)
#        
##            # print out the learned chi variables. (in this case, all 1's) 
##            print(chi.fm)
#            
#            fm = chi.fm
#            
##            print('Testing data...')  
#            
#            ## Compute ChI at every pixel location
#            data_out = np.zeros((data.shape[1]))
#            
#            for idx in range(data.shape[1]):
#                data_out[idx] = chi.chi_quad(data[:,idx])
#                
#            ## Normalize ChI output
#            data_out = data_out - np.min(data_out)
#            data_out = data_out / (1e-7 + np.max(data_out))
#        
#            data_out = data_out.reshape((all_activations.shape[1],all_activations.shape[2]))
#            plt.figure()
#            plt.imshow(data_out)
#            plt.axis('off')
#            plt.colorbar()
#            
#            savename = img_savepath + '/raw_fuzzcam.png'
#                
#            plt.savefig(savename)
#            plt.close()
#            
#            
#            cam_image = show_cam_on_image(image, data_out, True)
#            
#            plt.figure()
#            plt.imshow(cam_image, cmap='jet')
#            plt.title('FuzzCAM')
#            plt.axis('off')
#            
#            savename = img_savepath + '/fuzzcam.png'
#                
#            plt.savefig(savename)
#            plt.close()
#
#            sample_idx +=1
#           
#        else:
#            break
    
#    labels = np.array([1])
#    
#    h = np.zeros((all_activations.shape[0],1))
#    
#    for idx in range(all_activations.shape[0]):
#        
#        err = np.sum((all_activations[idx,:,:] - gt_img)**2)
#        err /= (all_activations.shape[1] * all_activations.shape[2])
        
########################### Train set/Test holdout ############################ 
        
################################################################################
############################# Train Large Dataset ##############################
################################################################################
#    
#    ###########################################################################
#    ########################### Extract Training Set ##########################
#    ###########################################################################
#    sample_idx = 0
#    for data in tqdm(test_loader):
#        
#        if (sample_idx<50):
#            
##            print(str(sample_idx))
#        
#            ## Load sample and groundtruth
#            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
#            
#            ###############################################################
#            ############# Convert groundtruth image into mask #############
#            ###############################################################
#            
#            gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
#            
#            ###############################################################
#            ################# Extract activation maps #####################
#            ###############################################################
#            
#            ## Normalize input to the model
#            cam_input = norm_image(images)
#            
#            ## Get activation maps
#            activations, importance_weights = activation_model(input_tensor=cam_input, target_category=int(labels))
#            importance_weights = importance_weights[0,:]
#    
#            ###################################################################
#            ########### Get most and least important activation maps ##########
#            ###################################################################
#            
#            top_k_activations_idx = (-importance_weights).argsort()[:TOP_K]
#            bottom_k_activations_idx = (importance_weights).argsort()[:BOTTOM_K]
#            
#            top_k_activations = activations[top_k_activations_idx]
#            bottom_k_activations = activations[bottom_k_activations_idx]
#            
#            top_k_weights = importance_weights[top_k_activations_idx]
#            bottom_k_weights = importance_weights[bottom_k_activations_idx]
#      
#            ###################################################################
#            ############# Combine Activaiton Maps and Groundtruth #############
#            ###################################################################
#            all_activations = np.append(top_k_activations, bottom_k_activations, axis=0)
#            labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
#    
#            data = np.zeros((all_activations.shape[0],
#                             all_activations.shape[1]*all_activations.shape[2]))
#    
#            for idx in range(all_activations.shape[0]):
#                data[idx,:] = np.reshape(activations[idx,:,:],
#                    (1,activations[idx,:,:].shape[0]*activations[idx,:,:].shape[1]))
#                
#            ###################################################################
#            ###################### Sub-sample Data ############################
#            ###################################################################
#            num_samples = data.shape[1]
#            
#            subsampled_idx = np.random.permutation(num_samples)[0:NUM_SUBSAMPLE]
#            data_sampled = data[:,subsampled_idx]
#            labels_sampled = labels[subsampled_idx]
#            
#            ###################################################################
#            ############# Add to all training data and groundtruth ############
#            ###################################################################
#            
#            if not(sample_idx):
#                train_data = data_sampled
#                train_labels = labels_sampled
#            else:
#                train_data = np.append(train_data, data_sampled, axis=1)
#                train_labels = np.append(train_labels, labels_sampled)
#                
#            sample_idx += 1
#            
#        else:
#            break
#            
#    ###########################################################################
#    ########################## Compute Fuzzy Measure ##########################
#    ###########################################################################
#        
#    print('Computing Fuzzy Measure...')    
#    
#    chi = ChoquetIntegral()    
#    
#    # train the chi via quadratic program 
#    chi.train_chi(train_data, train_labels)
#    fm = chi.fm
#    
#    ## Save learned FM
#    new_directory = experiment_savepath + new_save_path
#    new_savepath = new_directory + '/fm'
#    np.save(new_savepath,fm)
#    
#    fm_file = new_savepath + '.txt'
#    ## Save parameters         
#    with open(fm_file, 'w') as f:
#        json.dump(fm, f, indent=2)
#    f.close 
#    
#    del data
#    ###########################################################################
#    ############################## Test Data ##################################
#    ###########################################################################
#    print('Testing data...')
#    
#    sample_idx = 0
#    for data in tqdm(test_loader):
#        
#        if (sample_idx < 100):
#            
#            print(str(sample_idx))
#            
#            if GEN_PLOTS:
#                try:
#                    new_directory = experiment_savepath + new_save_path + '/img_' + str(sample_idx) 
#                    os.mkdir(new_directory)
#                    
#                    img_savepath = new_directory
#                except:
#                    img_savepath = new_directory
#        
#            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
#            
#            ############# Convert groundtruth image into mask #############
#            gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
#            
#            ## Normalize input to the model
#            cam_input = norm_image(images)
#            
#            ##################### Get Activation Maps ####################
#            activations, importance_weights = activation_model(input_tensor=cam_input, target_category=int(labels))
#            importance_weights = importance_weights[0,:]
#            
#            #################### Plot Activation Maps ####################
#            
#            images = images.permute(2,3,1,0)
#            images = images.detach().cpu().numpy()
#            image = images[:,:,:,0]
#            img = image
#
#            ###########################################################################
#            ################# Get most and least important activation maps ############
#            ###########################################################################
#            
#            ## Get most important and least important activations
#            top_k_activations_idx = (-importance_weights).argsort()[:TOP_K]
#            bottom_k_activations_idx = (importance_weights).argsort()[:BOTTOM_K]
#            
#            top_k_activations = activations[top_k_activations_idx]
#            bottom_k_activations = activations[bottom_k_activations_idx]
#            
#            top_k_weights = importance_weights[top_k_activations_idx]
#            bottom_k_weights = importance_weights[bottom_k_activations_idx]
#              
#            ###########################################################################
#            ############################## Generate Plots #############################
#            ###########################################################################
#                  
#            if GEN_PLOTS:
#                for k, img in enumerate(top_k_activations):
#                            plt.figure()
#                            plt.imshow(img)
#                            plt.axis('off')
#        #                    plt.colorbar()
#                            
#                            title = 'Top Activation: ' + str(k) + ', Importance: ' + str(top_k_weights[k])
#                            
#                            plt.title(title)
#                            
#                            savename = img_savepath + '/top_activation_idx_' + str(k) + '.png'
#                    
#                            plt.savefig(savename)
#                            plt.close()
#                            
#                for k, img in enumerate(bottom_k_activations):
#                            plt.figure()
#                            plt.imshow(img)
#                            plt.axis('off')
#        #                    plt.colorbar()
#                            
#                            title = 'Bottom Activation: ' + str(k) + ', Importance: ' + str(bottom_k_weights[k])
#                            
#                            plt.title(title)
#                            
#                            savename = img_savepath + '/bottom_activation_idx_' + str(k) + '.png'
#                    
#                            plt.savefig(savename)
#                            plt.close()
#            
#            
#                ## Plot GradCAM results
#                grayscale_cam = gradcam_model(input_tensor=cam_input,target_category=int(labels))
#                grayscale_cam = grayscale_cam[0, :]
#             
#                
#                gt_images = gt_images.detach().cpu().numpy()
#                gt_image = gt_images[0,:,:]
#                
#                if(labels.detach().cpu().numpy()==0):
#                    for key in parameters.bg_classes:
#                        value = parameters.all_classes[key]
#                        gt_image[np.where(gt_image==value)] = 1
#                        
#                elif(labels.detach().cpu().numpy()==1):
#                    for key in parameters.target_classes:
#                        value = parameters.all_classes[key]
#                        gt_image[np.where(gt_image==value)] = 1
#                
#                ## Convert segmentation image to mask
#                gt_image[np.where(gt_image!=1)] = 0
#                
#                ## Show image and groundtruth
#                plt.figure()
#                plt.imshow(gt_image, cmap='gray')
#                plt.axis('off')
#                
#                savename = img_savepath + '/grountruth_mask.png'
#                
#                plt.savefig(savename)
#                plt.close()
#            
#                ## Plot GradCAM
#                plt.figure()
#                cam_image = show_cam_on_image(image, grayscale_cam, True)
#                plt.imshow(cam_image, cmap='jet')
#                plt.axis('off')
#                plt.title('GradCAM')
#                
#                savename = img_savepath + '/gradcam.png'
#                
#                plt.savefig(savename)
#                plt.close()
#    
#           
#            all_activations = np.append(top_k_activations, bottom_k_activations, axis=0)
#            
#            labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
#            
#            data = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
#            
#            for idx in range(all_activations.shape[0]):
#                data[idx,:] = np.reshape(activations[idx,:,:],
#                    (1,activations[idx,:,:].shape[0]*activations[idx,:,:].shape[1]))
#            
#            ## Compute ChI at every pixel location
#            data_out = np.zeros((data.shape[1]))
#            
#            for idx in range(data.shape[1]):
#                data_out[idx] = chi.chi_quad(data[:,idx])
#                
#            ## Normalize ChI output
#            data_out = data_out - np.min(data_out)
#            data_out = data_out / (1e-7 + np.max(data_out))
#        
#            data_out = data_out.reshape((all_activations.shape[1],all_activations.shape[2]))
#            plt.figure()
#            plt.imshow(data_out)
#            plt.axis('off')
#            plt.colorbar()
#            
#            savename = img_savepath + '/raw_fuzzcam.png'
#                
#            plt.savefig(savename)
#            plt.close()
#            
#            
#            cam_image = show_cam_on_image(image, data_out, True)
#            
#            plt.figure()
#            plt.imshow(cam_image, cmap='jet')
#            plt.title('FuzzCAM')
#            plt.axis('off')
#            
#            savename = img_savepath + '/fuzzcam.png'
#                
#            plt.savefig(savename)
#            plt.close()
#
#            sample_idx +=1
#           
#        else:
#            break
    
    
    
########################### Train/Test Same Image (Single Image: First Layer) ############################# 
    
#    ###########################################################################
#    ############### Extract activation maps and importance weights ############
#    ###########################################################################
#    
#    new_directory = experiment_savepath + new_save_path
#    
#    sample_idx = 0
#    for data in tqdm(test_loader):
#        
#        if (sample_idx==SAMPLE_IDX):
#            
#            if GEN_PLOTS:
#                try:
#                    os.mkdir(new_directory)
#                    
#                    img_savepath = new_directory
#                except:
#                    img_savepath = new_directory
#        
#            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
#            
#            ############# Convert groundtruth image into mask #############
#            gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
#            
#            ## Normalize input to the model
#            cam_input = norm_image(images)
#            
#            ##################### Get Activation Maps ####################
#            activations, importance_weights = activation_model(input_tensor=cam_input, target_category=int(labels))
##            importance_weights = importance_weights[0,:]
#            
#            #################### Plot Activation Maps ####################
#            
#            images = images.permute(2,3,1,0)
#            images = images.detach().cpu().numpy()
#            image = images[:,:,:,0]
#            img = image
#            
#            if GEN_PLOTS:
#                plt.figure()
#                plt.imshow(img)
#                plt.axis('off')
#                
#                savename = img_savepath + '/input.png'
#                
#                plt.savefig(savename)
#                plt.close()
#                
#                for k, img in enumerate(activations):
#                        plt.figure()
#                        plt.imshow(img)
#                        plt.axis('off')
#                        
#                        title = 'Activation: ' + str(k) 
#                        
#                        plt.title(title)
#                        
#                        savename = img_savepath + '/all_activations_layer_1/activation_idx_' + str(k) + '.png'
#                
#                        plt.savefig(savename)
#                        plt.close()
#                    
#            break
#        else:
#            sample_idx +=1
#
#    ###########################################################################
#    ####################### Hand-pick activations #############################
#    ###########################################################################
#    
##    handpicked_idx = [20,56,60,37,23,41] ## Layer 1
#    handpicked_idx = [20,56,60,37,23] ## Layer 1
#    
#    handpicked_activations = activations[handpicked_idx,:,:]
#      
#    ###########################################################################
#    ############################## Generate Plots #############################
#    ###########################################################################
#                  
#    if GEN_PLOTS:
#        for k, img in enumerate(handpicked_activations):
#            plt.figure()
#            plt.imshow(img)
#            plt.axis('off')
#            plt.colorbar()
#            
#            title = 'Hand-picked Activation: ' + str(handpicked_idx[k])
#            
#            plt.title(title)
#            
#            savename = img_savepath + '/hand_picked_activation_idx_' + str(handpicked_idx[k]) + '.png'
#    
#            plt.savefig(savename)
#            plt.close()
#                    
##    
##        ## Plot GradCAM results
##        grayscale_cam = gradcam_model(input_tensor=cam_input,target_category=int(labels))
##        grayscale_cam = grayscale_cam[0, :]
##     
#        
#        gt_images = gt_images.detach().cpu().numpy()
#        gt_image = gt_images[0,:,:]
#        
#        if(labels.detach().cpu().numpy()==0):
#            for key in parameters.bg_classes:
#                value = parameters.all_classes[key]
#                gt_image[np.where(gt_image==value)] = 1
#                
#        elif(labels.detach().cpu().numpy()==1):
#            for key in parameters.target_classes:
#                value = parameters.all_classes[key]
#                gt_image[np.where(gt_image==value)] = 1
#        
#        ## Convert segmentation image to mask
#        gt_image[np.where(gt_image!=1)] = 0
#        
#        ## Show image and groundtruth
#        plt.figure()
#        plt.imshow(gt_image, cmap='gray')
#        plt.axis('off')
#        
#        savename = img_savepath + '/grountruth_mask.png'
#        
#        plt.savefig(savename)
#        plt.close()
#    
##        ## Plot GradCAM
##        plt.figure()
##        cam_image = show_cam_on_image(image, grayscale_cam, True)
##        plt.imshow(cam_image, cmap='jet')
##        plt.axis('off')
##        plt.title('GradCAM')
##        
##        savename = img_savepath + '/gradcam.png'
##        
##        plt.savefig(savename)
##        plt.close()
#    
#                
#    ###########################################################################
#    ############################## Compute ChI ################################
#    ###########################################################################
#    
#    all_activations = handpicked_activations
#    
#    labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
#    
#    data = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
#    
#    for idx in range(all_activations.shape[0]):
#        data[idx,:] = np.reshape(activations[idx,:,:],(1,activations[idx,:,:].shape[0]*activations[idx,:,:].shape[1]))
#        
#    print('Computing Fuzzy Measure...')    
#    
#    chi = ChoquetIntegral()    
#    
#    # train the chi via quadratic program 
#    chi.train_chi(data, labels)
#    fm = chi.fm
#    
#    fm_savepath = img_savepath + '/fm'
#    np.save(fm_savepath, fm)
#        
#    fm_file = fm_savepath + '.txt'
#    ## Save parameters         
#    with open(fm_file, 'w') as f:
#        json.dump(fm, f, indent=2)
#    f.close 
#    
#    print('Testing data...')  
#    
#    ## Compute ChI at every pixel location
#    data_out = np.zeros((data.shape[1]))
#    
#    for idx in range(data.shape[1]):
#        data_out[idx] = chi.chi_quad(data[:,idx])
#        
#    ## Normalize ChI output
#    data_out = data_out - np.min(data_out)
#    data_out = data_out / (1e-7 + np.max(data_out))
#
#    data_out = data_out.reshape((all_activations.shape[1],all_activations.shape[2]))
#    plt.figure()
#    plt.imshow(data_out)
#    plt.axis('off')
#    plt.colorbar()
#    
#    savename = img_savepath + '/raw_fuzzcam.png'
#        
#    plt.savefig(savename)
#    plt.close()
#    
#    
#    cam_image = show_cam_on_image(image, data_out, True)
#    
#    plt.figure()
#    plt.imshow(cam_image, cmap='jet')
#    plt.title('FuzzCAM')
#    plt.axis('off')
#    
#    savename = img_savepath + '/fuzzcam.png'
#        
#    plt.savefig(savename)
#    plt.close()
#
#    sample_idx +=1
    
    
################################################################################
####################### Test Segmentation Performance ##########################
################################################################################
#    
#    results_file = parameters.outf.replace('output','cam_fusion') + '/segmentation_experiments/results.txt'
#    new_save_path = '/segmentation_experiments'
#    
#    results_dict = dict()
#    visualization_results = dict()
#    visualization_results['thresholds'] = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#    
#    num_samples = len(test_loader)
#    num_thresh = len(visualization_results['thresholds'])
#    
#    norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#    
#    methods = \
#    {"gradcam": GradCAM,
#     "scorecam": ScoreCAM,
#     "gradcam++": GradCAMPlusPlus,
#     "ablationcam": AblationCAM,
#     "eigencam": EigenCAM,
#     "layercam": LayerCAM}
#    
#    if (parameters.model == 'vgg16'):
#        cam_layers = \
#        {4:"layer1",
#         9:"layer2",
#         16:"layer3",
#         23:"layer4",
#         30:"layer5"}
#
#    ## Turn on gradients for CAM computation 
#    for param in model.parameters():
#        param.requires_grad = True
#
#    model.eval()
#    
#    for current_cam in ['layercam']:
#        for layer in [4,9,16,23,30]:      
#            
#            ###############################################################
#            ######## Define CAM/Choquet and learn FM elements #############
#            ###############################################################
#            cam_algorithm = methods[current_cam]
#    
#            if (parameters.model == 'vgg16'):
#                cam = cam_algorithm(model=model, target_layers=[model.features[layer]], use_cuda=parameters.cuda)
#                activation_model = ActivationCAM(model=model, target_layers=[model.features[layer]], use_cuda=parameters.cuda)
#                
#            ###########################################################################
#            ########################### Extract Training Set ##########################
#            ###########################################################################
#            sample_idx = 0
#            for data in tqdm(test_loader):
#                
#                if (sample_idx<50):
#                
#                    ## Load sample and groundtruth
#                    images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
#                    
#                    ###############################################################
#                    ############# Convert groundtruth image into mask #############
#                    ###############################################################
#                    
#                    gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
#                    
#                    ###############################################################
#                    ################# Extract activation maps #####################
#                    ###############################################################
#                    
#                    ## Normalize input to the model
#                    cam_input = norm_image(images)
#                    
#                    ## Get activation maps
#                    activations, importance_weights = activation_model(input_tensor=cam_input, target_category=int(labels))
#                    importance_weights = importance_weights[0,:]
#            
#                    ###################################################################
#                    ########### Get most and least important activation maps ##########
#                    ###################################################################
#                    
#                    top_k_activations_idx = (-importance_weights).argsort()[:TOP_K]
#                    bottom_k_activations_idx = (importance_weights).argsort()[:BOTTOM_K]
#                    
#                    top_k_activations = activations[top_k_activations_idx]
#                    bottom_k_activations = activations[bottom_k_activations_idx]
#                    
#                    top_k_weights = importance_weights[top_k_activations_idx]
#                    bottom_k_weights = importance_weights[bottom_k_activations_idx]
#              
#                    ###################################################################
#                    ############# Combine Activaiton Maps and Groundtruth #############
#                    ###################################################################
#                    all_activations = np.append(top_k_activations, bottom_k_activations, axis=0)
#                    labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
#            
#                    data = np.zeros((all_activations.shape[0],
#                                     all_activations.shape[1]*all_activations.shape[2]))
#            
#                    for idx in range(all_activations.shape[0]):
#                        data[idx,:] = np.reshape(activations[idx,:,:],
#                            (1,activations[idx,:,:].shape[0]*activations[idx,:,:].shape[1]))
#                        
#                    ###################################################################
#                    ###################### Sub-sample Data ############################
#                    ###################################################################
#                    num_samples = data.shape[1]
#                    
#                    subsampled_idx = np.random.permutation(num_samples)[0:NUM_SUBSAMPLE]
#                    data_sampled = data[:,subsampled_idx]
#                    labels_sampled = labels[subsampled_idx]
#                    
#                    ###################################################################
#                    ############# Add to all training data and groundtruth ############
#                    ###################################################################
#                    
#                    if not(sample_idx):
#                        train_data = data_sampled
#                        train_labels = labels_sampled
#                    else:
#                        train_data = np.append(train_data, data_sampled, axis=1)
#                        train_labels = np.append(train_labels, labels_sampled)
#                        
#                    sample_idx += 1
#                    
#                else:
#                    break
#                        
#            ###########################################################################
#            ########################## Compute Fuzzy Measure ##########################
#            ###########################################################################
#                
#            print('Computing Fuzzy Measure...')    
#            
#            chi = ChoquetIntegral()    
#            
#            # train the chi via quadratic program 
#            chi.train_chi(train_data, train_labels)
#            fm = chi.fm
#            
#            ## Save learned FM
#            new_directory = experiment_savepath + new_save_path
#            new_savepath = new_directory + '/fm_layer_' + cam_layers[layer]
#            np.save(new_savepath,fm)
#            
#            fm_file = new_savepath + '.txt'
#            ## Save parameters         
#            with open(fm_file, 'w') as f:
#                json.dump(fm, f, indent=2)
#            f.close 
#    
#            ###############################################################
#            ###################### Test Model #############################
#            ###############################################################
#            
#            n_samples = len(test_loader)
##            metric_iou = np.zeros(n_samples, dtype="float32")
#            metric_iou = np.zeros((num_thresh,num_samples))
#            current_model = 'fuzzcam_choquet_lse' + '_' + cam_layers[layer]
#
#            target_category = None
#            sample_idx = 0
#            for data in tqdm(test_loader):
#          
#                images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
#                
#                ############# Convert groundtruth image into mask #############
#                gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
#                
#                ## Normalize input to the model
#                cam_input = norm_image(images)
#                
#                ##################### Get FuzzCAM Segmentation ####################
#          
#                ## Get activation maps
#                activations, importance_weights = activation_model(input_tensor=cam_input, target_category=int(labels))
#                importance_weights = importance_weights[0,:]
#        
#                ###################################################################
#                ########### Get most and least important activation maps ##########
#                ###################################################################
#                
#                top_k_activations_idx = (-importance_weights).argsort()[:TOP_K]
#                bottom_k_activations_idx = (importance_weights).argsort()[:BOTTOM_K]
#                
#                top_k_activations = activations[top_k_activations_idx]
#                bottom_k_activations = activations[bottom_k_activations_idx]
#                
#                top_k_weights = importance_weights[top_k_activations_idx]
#                bottom_k_weights = importance_weights[bottom_k_activations_idx]
#          
#                ###################################################################
#                ############# Combine Activaiton Maps and Groundtruth #############
#                ###################################################################
#                all_activations = np.append(top_k_activations, bottom_k_activations, axis=0)
#                labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
#                
#                test_data = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
#                
#                for idx in range(all_activations.shape[0]):
#                    test_data[idx,:] = np.reshape(activations[idx,:,:],
#                        (1,activations[idx,:,:].shape[0]*activations[idx,:,:].shape[1]))
#            
#                ## Compute ChI at every pixel location
#                data_out = np.zeros((test_data.shape[1]))
#                
#                for idx in range(test_data.shape[1]):
#                    data_out[idx] = chi.chi_quad(test_data[:,idx])
#                    
#                ## Normalize ChI output
#                data_out = data_out - np.min(data_out)
#                data_out = data_out / (1e-7 + np.max(data_out))
#            
#                data_out = data_out.reshape((all_activations.shape[1],all_activations.shape[2]))
#                
#                ## Collect results over various thresholds 
#                for thresh_idx, thresh in enumerate([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
#                
##                    print(current_model+'_thresh_'+str(thresh))
#                    
#                    ## Binarize CAM for segmentation
#                    pred_img = cam_img_to_seg_img(data_out, thresh)
#                    pred_img = pred_img.astype('int')
#                    
#                    ##################### Evaluate segmentation ###################
#                    intersection = np.logical_and(pred_img, gt_img)
#                    union = np.logical_or(pred_img, gt_img)
#                    
#                    ## Catch divide by zero(union of prediction and groundtruth is empty)
#                    try:
#                        iou_score = np.sum(intersection) / np.sum(union)
#                    except:
#                        iou_score = 0
#                        
#                    metric_iou[thresh_idx,sample_idx] = round(iou_score,5)
#                
#                sample_idx +=1            
#         
#            
#        ####################### Save results after each layer #################
#            ## Compute statistics over the entire test set
#            
#            metric_iou_mean = np.zeros(len([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
#            metric_iou_std = np.zeros(len([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
#            
#            for k in range(len([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])):
#                
#                metric_iou_mean[k] = round(np.mean(metric_iou[k,:]),3)
#                metric_iou_std[k] = round(np.std(metric_iou[k,:]),3)
#            
#            ## Amalgamate results into one dictionary
#            
#            for k, thresh in enumerate([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
#                
#                
#                details = {'method': 'fuzzcam_choquet_lse',
#                           'threshold': thresh,
#                           'iou mean':metric_iou_mean[k],
#                           'iou std':metric_iou_std[k]}
#            
#                ## Save results to global dictionary of results
#                model_name = current_model + '_thresh_' + str(thresh)
#                results_dict[model_name] = details
#            
#                ## Write results to text file
#                with open(results_file, 'a+') as f:
#                    for key, value in details.items():
#                        f.write('%s:%s\n' % (key, value))
#                    f.write('\n')
#                    f.close()
#            
#            ## Clean up memory
#            del details
#            del metric_iou_mean
#            del metric_iou_std
#            del metric_iou 
#            torch.cuda.empty_cache()
#
#            
#        ## Save results
#        results_savepath = parameters.outf.replace('output','cam_fusion') + '/segmentation_experiments/results.npy'
#        np.save(results_savepath, results_dict, allow_pickle=True)
   
    
############################ Over-fitting Experiments ##########################
#    
#    ###########################################################################
#    ############### Extract activation maps and importance weights ############
#    ###########################################################################
#    
#    new_directory = experiment_savepath + new_save_path
#    img_savepath = new_directory
#    
#    sample_idx = 0
#    for data in tqdm(test_loader):
#        
#        if (sample_idx==SAMPLE_IDX):
#            
#            if GEN_PLOTS:
#                try:
#                    os.mkdir(new_directory)
#                    
#                    img_savepath = new_directory
#                except:
#                    img_savepath = new_directory
#        
#            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
#            
#            ############# Convert groundtruth image into mask #############
#            gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
#            
#            ## Normalize input to the model
#            cam_input = norm_image(images)
#            
#            ##################### Get Activation Maps ####################
#            activations, _ = activation_model(input_tensor=cam_input, target_category=int(labels))
#                    
#            break
#        else:
#            sample_idx +=1
#
#    ###########################################################################
#    ####################### Hand-pick activations #############################
#    ###########################################################################
#    
###    handpicked_idx = [20,56,60,37,23,41] ## Layer 1
#    handpicked_idx = [20,23,37,56,60] ## Layer 1
##    
##    handpicked_idx = [23,56] ## Layer 1
#    handpicked_activations = activations[handpicked_idx,:,:]
#    
##    ## Add inverted activation maps
##    for k in range(handpicked_activations.shape[0]):
##        act_map = handpicked_activations[k,:,:]
##        
##        inv_map = (1-act_map)
##        
##        if not(k):
##            all_inv_maps = np.expand_dims(inv_map, axis=0)
##        else:
##            all_inv_maps = np.concatenate((all_inv_maps,np.expand_dims(inv_map, axis=0)), axis=0)
##            
##        plt.figure()
##        plt.imshow(act_map)
##        plt.colorbar()
##        plt.figure()
##        plt.imshow(inv_map)
##        plt.colorbar()
##    
##    handpicked_activations = np.concatenate((handpicked_activations, all_inv_maps),axis=0)
#        
#    
#    ## [gt,56]
##    handpicked_activations = np.concatenate((np.expand_dims(gt_img,axis=0),np.expand_dims(activations[23,:,:],axis=0),np.expand_dims(activations[37,:,:],axis=0)), axis=0)
#    
#    ###########################################################################
#    ############################## Compute ChI ################################
#    ###########################################################################
#    
#    all_activations = handpicked_activations
#    
#    for k in range(all_activations.shape[0]):
#        plt.figure()
#        plt.imshow(all_activations[k,:,:])
#        plt.axis('off')
#        title = 'Activation idx: ' + str(k)
#        plt.title(title)
#        
#        savename = img_savepath + '/activation_idx_' + str(k) + '.png'
#        plt.savefig(savename)
#        plt.close()
#        
#    
#    labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
#    
#    data = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
#    
#    for idx in range(all_activations.shape[0]):
#        data[idx,:] = np.reshape(all_activations[idx,:,:],(1,all_activations[idx,:,:].shape[0]*all_activations[idx,:,:].shape[1]))
#        
#    print('Computing Fuzzy Measure...')    
#    
#    chi = ChoquetIntegral()    
#    
#    # train the chi via quadratic program 
#    chi.train_chi(data, labels)
#    fm = chi.fm
#    
#    fm_savepath = img_savepath + '/fm'
#    np.save(fm_savepath, fm)
#        
#    fm_file = fm_savepath + '.txt'
#    ## Save parameters         
#    with open(fm_file, 'w') as f:
#        json.dump(fm, f, indent=2)
#    f.close 
#    
#    ###########################################################################
#    ################################ Test Data ################################
#    ###########################################################################
#    print('Testing data...')  
#    
#    ## Compute ChI at every pixel location
#    data_out = np.zeros((data.shape[1]))
#    
#    for idx in range(data.shape[1]):
#        data_out[idx] = chi.chi_quad(data[:,idx])
#        
#    plt.figure()
#    plt.imshow(data_out.reshape((all_activations.shape[1],all_activations.shape[2])))
#    plt.axis('off')
#    plt.title('FuzzCAM No Normalization')
#    plt.colorbar()
#        
#    ## Normalize ChI output
#    data_out = data_out - np.min(data_out)
#    data_out = data_out / (1e-7 + np.max(data_out))
#    
#    ###########################################################################
#    ############################## Compute IoU ################################
#    ###########################################################################
#    ## Binarize feature
#    
#    gt_img = gt_img > 0
#    
#    try:
#        img_thresh = threshold_otsu(data_out)
#        binary_feature_map = data_out.reshape((all_activations.shape[1],all_activations.shape[2])) > img_thresh
#    except:
#        binary_feature_map = data_out < 0.1
#
#    ## Compute fitness as IoU to pseudo-groundtruth
#    intersection = np.logical_and(binary_feature_map, gt_img)
#    union = np.logical_or(binary_feature_map, gt_img)
#    
#    try:
#        iou_score = np.sum(intersection) / np.sum(union)
#    except:
#        iou_score = 0
# 
#    savename = img_savepath + '/raw_fuzzcam.png'
#    
#    ###########################################################################
#    ############################## Visualizations #############################
#    ###########################################################################
#
#    data_out = data_out.reshape((all_activations.shape[1],all_activations.shape[2]))
#    plt.figure()
#    plt.imshow(data_out)
#    plt.axis('off')
#    plt.colorbar()
#    
#    title = 'Raw FuzzCAM; IoU: ' + str(iou_score)
#    plt.title(title)
#    
#    savename = img_savepath + '/raw_fuzzcam.png'
#        
#    plt.savefig(savename)
#    plt.close()
#    
#    images = images.permute(2,3,1,0)
#    images = images.detach().cpu().numpy()
#    image = images[:,:,:,0]
#    cam_image = show_cam_on_image(image, data_out, True)
#    
#    plt.figure()
#    plt.imshow(cam_image, cmap='jet')
##    plt.title('FuzzCAM')
#    title = 'FuzzCAM; IoU: ' + str(iou_score)
#    plt.title(title)
#    plt.axis('off')
#    
#    savename = img_savepath + '/fuzzcam.png'
#        
#    plt.savefig(savename)
#    plt.close()
#
#    sample_idx +=1
    
###############################################################################
############################# Sorting Analysis ################################
###############################################################################
    
    new_save_path = '/sort_analysis/23_56'
    
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
        
            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
            
            ############# Convert groundtruth image into mask #############
            gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
            
            ## Normalize input to the model
            cam_input = norm_image(images)
            
            ##################### Get Activation Maps ####################
            activations, _ = activation_model(input_tensor=cam_input, target_category=int(labels))
                    
            break
        else:
            sample_idx +=1

    ###########################################################################
    ####################### Hand-pick activations #############################
    ###########################################################################
        
    handpicked_idx = [23,56] ## Layer 1
    handpicked_activations = activations[handpicked_idx,:,:]
    
#    ## Add inverted activation maps
#    for k in range(handpicked_activations.shape[0]):
#        act_map = handpicked_activations[k,:,:]
#        
#        inv_map = (1-act_map)
#        
#        if not(k):
#            all_inv_maps = np.expand_dims(inv_map, axis=0)
#        else:
#            all_inv_maps = np.concatenate((all_inv_maps,np.expand_dims(inv_map, axis=0)), axis=0)
#            
#    handpicked_activations = np.concatenate((handpicked_activations, all_inv_maps),axis=0)
    
    ## Add zero set or one set
#    handpicked_activations = np.concatenate((np.expand_dims(np.zeros((activations.shape[1],activations.shape[2])),axis=0),handpicked_activations), axis=0)
#    handpicked_activations = np.concatenate((np.expand_dims(np.ones((activations.shape[1],activations.shape[2])),axis=0),handpicked_activations), axis=0)        
    
    ## [gt,56]
#    handpicked_activations = np.concatenate((np.expand_dims(gt_img,axis=0),np.expand_dims(activations[23,:,:],axis=0),np.expand_dims(activations[37,:,:],axis=0)), axis=0)
#    handpicked_activations = np.concatenate((np.expand_dims(gt_img,axis=0),np.expand_dims(activations[56,:,:],axis=0)), axis=0)
    
    
    ###########################################################################
    ############################## Compute ChI ################################
    ###########################################################################
    
    all_activations = handpicked_activations
    
    for k in range(all_activations.shape[0]):
        plt.figure()
        plt.imshow(all_activations[k,:,:])
        plt.axis('off')
        title = 'Activation idx: ' + str(k+1)
        plt.title(title)
        
        savename = img_savepath + '/activation_idx_' + str(k+1) + '.png'
        plt.savefig(savename)
        plt.close()
        
    
    labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
    
    data = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
    
    for idx in range(all_activations.shape[0]):
        data[idx,:] = np.reshape(all_activations[idx,:,:],(1,all_activations[idx,:,:].shape[0]*all_activations[idx,:,:].shape[1]))
        
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
    
    for idx in range(data.shape[1]):
        data_out[idx], sort_label = chi.chi_by_sort(data[:,idx])
        data_out_sort_labels[idx] = sort_label
        
    
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
    data_out_sort_labels = data_out_sort_labels.reshape((all_activations.shape[1],all_activations.shape[2]))
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
        binary_feature_map = data_out.reshape((all_activations.shape[1],all_activations.shape[2])) > img_thresh
    except:
        binary_feature_map = data_out < 0.1

    ## Compute fitness as IoU to pseudo-groundtruth
    intersection = np.logical_and(binary_feature_map, gt_img)
    union = np.logical_or(binary_feature_map, gt_img)
    
    try:
        iou_score = np.sum(intersection) / np.sum(union)
    except:
        iou_score = 0
 
    savename = img_savepath + '/raw_fuzzcam.png'
    
    ###########################################################################
    ############################## Visualizations #############################
    ###########################################################################

    data_out = data_out.reshape((all_activations.shape[1],all_activations.shape[2]))
    plt.figure()
    plt.imshow(data_out)
    plt.axis('off')
    plt.colorbar()
    
    title = 'Raw FuzzCAM; IoU: ' + str(iou_score)
    plt.title(title)
    
    savename = img_savepath + '/raw_fuzzcam.png'
        
    plt.savefig(savename)
    plt.close()
    
    images = images.permute(2,3,1,0)
    images = images.detach().cpu().numpy()
    image = images[:,:,:,0]
    cam_image = show_cam_on_image(image, data_out, True)
    
    plt.figure()
    plt.imshow(cam_image, cmap='jet')
#    plt.title('FuzzCAM')
    title = 'FuzzCAM; IoU: ' + str(iou_score)
    plt.title(title)
    plt.axis('off')
    
    savename = img_savepath + '/fuzzcam.png'
        
    plt.savefig(savename)
    plt.close()

    sample_idx +=1
    
    
    
    
         
  

    
    