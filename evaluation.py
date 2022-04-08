#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:56:29 2021

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  util.py
    *
    *  Desc: This file contains the helper and utility functions available 
    *        for MIL UNet.
    *
    *  Written by:  Guohao Yu, Weihuang Xu, Dr. Alina Zare, 
    *  Connor H. McCurley, Deanna L. Gran
    *
    *  Latest Revision:  2019-11-18
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


def convert_gt_img_to_mask(gt_images, labels, parameters):
    
    gt_images = gt_images.detach().cpu().numpy()
    gt_image = gt_images[0,:,:]
                
    if(labels.detach().cpu().numpy()==0):
        for key in parameters.bg_classes:
            value = parameters.all_classes[key]
            gt_image[np.where(gt_image==value)] = 1
            
    elif(labels.detach().cpu().numpy()==1):
        for key in parameters.target_classes:
            value = parameters.all_classes[key]
            gt_image[np.where(gt_image==value)] = 1
    
    ## Convert segmentation image to mask
    gt_image[np.where(gt_image!=1)] = 0
    
    return gt_image

def cam_img_to_seg_img(cam_img, CAM_SEG_THRESH):
    
    cam_img[cam_img>=CAM_SEG_THRESH] = 1
    cam_img[cam_img<CAM_SEG_THRESH] = 0
    
    return cam_img




