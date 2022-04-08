#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Mar  5 12:02:36 2021
Compute EMD between two images' superpixel signatures
@author: jpeeples
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def img_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""
    
    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i,j], i, j])
            count += 1
    return sig

def plot_flow(sig1, sig2, flow, arrow_width_scale=3):
    """Plots the flow computed by cv2.EMD
    
    The source images are retrieved from the signatures and
    plotted in a combined image, with the first image in the
    red channel and the second in the green. Arrows are
    overplotted to show moved earth, with arrow thickness
    indicating the amount of moved earth."""
    
    img1 = sig_to_img(sig1)
    img2 = sig_to_img(sig2)
    combined = np.dstack((img1, img2, 0*img2))
    # RGB values should be between 0 and 1
    combined /= combined.max()
    print('Red channel is "before"; green channel is "after"; yellow means "unchanged"')
    plt.imshow(combined)

    flows = np.transpose(np.nonzero(flow))
    for src, dest in flows:
        # Skip the pixel value in the first element, grab the
        # coordinates. It'll be useful later to transpose x/y.
        start = sig1[src, 1:][::-1]
        end = sig2[dest, 1:][::-1]
        if np.all(start == end):
            # Unmoved earth shows up as a "flow" from a pixel
            # to that same exact pixel---don't plot mini arrows
            # for those pixels
            continue
        
        # Add a random shift to arrow positions to reduce overlap.
        shift = np.random.random(1) * .3 - .15
        start = start + shift
        end = end + shift
        
        mag = flow[src, dest] * arrow_width_scale
        plt.quiver(*start, *(end - start), angles='xy',
                   scale_units='xy', scale=1, color='white',
                   edgecolor='black', linewidth=mag/3,
                   width=mag, units='dots',
                   headlength=5,
                   headwidth=3,
                   headaxislength=4.5)
    
    plt.title("Earth moved from img1 to img2")
    
def sig_to_img(sig):
    """Convert a signature back to a 2D image"""
    intsig = sig.astype(int)
    img = np.empty((intsig[:, 1].max()+1, intsig[:, 2].max()+1), dtype=float)
    for i in range(sig.shape[0]):
        img[intsig[i, 1], intsig[i, 2]] = sig[i, 0]
    return img

def compute_EMD(arr1,arr2):
    
    sig1 = img_to_sig(arr1.astype(np.float32))
    sig2 = img_to_sig(arr2.astype(np.float32))
   
    try:
        EMD_score, _, flow = cv2.EMD(sig1,sig2,cv2.DIST_L2)
    except:
        EMD_score = 10000
       
    if EMD_score < 0: #Need distance to be positive, negative just indicates direction
        EMD_score = abs(EMD_score)
        
    return EMD_score, flow