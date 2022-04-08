#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:16:55 2022

@author: cmccurley
"""


savepath = './synthetic'

###############################################################################
########################### Import Packages ###################################
###############################################################################

import json
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from cm_choquet_integral import ChoquetIntegral

###############################################################################
########################## Define Global Parameters ###########################
###############################################################################
NUM_SOURCES = 3

###############################################################################
############################# Load Data #######################################
###############################################################################
data = scipy.io.loadmat('./synthetic3sources/demo_data_cl.mat')
X = data['X']

NUM_SAMPLES = X.shape[0]
NUM_FEATURES = X.shape[1]

#plt.figure()
#plt.scatter(X[:,0],X[:,1], s=40, facecolor='none', edgecolor='navy')
#plt.title('Template', fontsize=18)

###############################################################################
########################### Create Groundtruth ################################
###############################################################################
X0 = X
neg_coords = np.where((X[:,0]-X[:,1] < 0) | (X[:,0]-X[:,1] > 0.3))
pos_coords = np.where((X[:,0]-X[:,1] >= 0) & (X[:,0]-X[:,1] <= 0.3))

X0_Labels = np.zeros((X0.shape[0]))
X0_Labels[pos_coords] = 1

plt.figure()
plt.scatter(X0[neg_coords,0],X0[neg_coords,1], s=40, facecolor='none', edgecolor='navy')
plt.scatter(X0[pos_coords,0],X0[pos_coords,1], s=40, facecolor='none', edgecolor='darkorange')
plt.title('Groundtruth', fontsize=18)
img_path = savepath + '/groundtruth.png'
plt.savefig(img_path)
plt.close()

###############################################################################
############################ Create Source 1 ##################################
###############################################################################
X1 = X
neg_coords = np.where((X[:,0]-X[:,1] < 0))
pos_coords = np.where((X[:,0]-X[:,1] >= 0))

X1_Labels = np.zeros((X1.shape[0]))
X1_Labels[pos_coords] = 1

plt.figure()
plt.scatter(X1[neg_coords,0],X1[neg_coords,1], s=40, facecolor='none', edgecolor='navy')
plt.scatter(X1[pos_coords,0],X1[pos_coords,1], s=40, facecolor='none', edgecolor='darkorange')
plt.title('Source 1', fontsize=18)
img_path = savepath + '/source1.png'
plt.savefig(img_path)
plt.close()

###############################################################################
############################ Create Source 2 ##################################
###############################################################################
X2 = X
neg_coords = np.where((X[:,0]-X[:,1] >= 0.3))
pos_coords = np.where((X[:,0]-X[:,1] < 0.3))

X2_Labels = np.zeros((X2.shape[0]))
X2_Labels[pos_coords] = 1

plt.figure()
plt.scatter(X2[neg_coords,0],X2[neg_coords,1], s=40, facecolor='none', edgecolor='navy')
plt.scatter(X2[pos_coords,0],X2[pos_coords,1], s=40, facecolor='none', edgecolor='darkorange')
plt.title('Source 2', fontsize=18)
img_path = savepath + '/source2.png'
plt.savefig(img_path)
plt.close()

###############################################################################
############################ Create Source 3 ##################################
###############################################################################
X3 = X
neg_coords = np.where((-X[:,0]-X[:,1] < -1))
pos_coords = np.where((-X[:,0]-X[:,1] >= -1))

X3_Labels = np.zeros((X3.shape[0]))
X3_Labels[pos_coords] = 1

plt.figure()
plt.scatter(X3[neg_coords,0],X3[neg_coords,1], s=40, facecolor='none', edgecolor='navy')
plt.scatter(X3[pos_coords,0],X3[pos_coords,1], s=40, facecolor='none', edgecolor='darkorange')
plt.title('Source 3', fontsize=18)
img_path = savepath + '/source3.png'
plt.savefig(img_path)
plt.close()

###############################################################################
############################## Combine Data ###################################
###############################################################################
X_train = np.zeros((NUM_SAMPLES,NUM_SOURCES))
X_train[:,0], X_train[:,1], X_train[:,2] = X1_Labels, X2_Labels, X3_Labels
X_train = X_train.T
Y_train = X0_Labels

###############################################################################
########################## Train Choquet Integral #############################
###############################################################################
print('Computing Fuzzy Measure...')    
            
chi = ChoquetIntegral()    

# Train the ChI via quadratic program 
chi.train_chi(X_train, Y_train)
fm = chi.fm

## Save measure values     
fm_savepath = savepath + '/fm'
np.save(fm_savepath, fm)
    
fm_file = fm_savepath + '.txt'
    
with open(fm_file, 'w') as f:
    json.dump(fm, f, indent=2)
f.close 
    
###########################################################################
############################ Print Equations ##############################
###########################################################################
## Save equations
eqt_dict = chi.get_linear_eqts()

savename = savepath + '/chi_equations.txt'
with open(savename, 'w') as f:
    json.dump(eqt_dict, f, indent=2)
f.close 
                
###############################################################################
########################### Test Choquet Integral #############################
###############################################################################
            
X_test = X_train
Y_test = Y_train

print('Testing data...')  
            
## Compute ChI at every pixel location
data_out = np.zeros((X_test.shape[1]))
data_out_sorts = []
data_out_sort_labels = np.zeros((X_test.shape[1]))   

all_sorts = chi.all_sorts

if (NUM_SOURCES < 7):

    for idx in range(X_test.shape[1]):
        data_out[idx], data_out_sort_labels[idx] = chi.chi_by_sort(X_test[:,idx])               
    
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
    savename = savepath + '/hist_of_sorts.png'
    plt.savefig(savename)
    plt.close()
    
    ## Plot image colored by sort
    plt.figure(figsize=[10,8])
    plt.scatter(X[:,0],X[:,1],c=data_out_sort_labels)
    plt.axis('off')
    plt.colorbar()
    title = 'Image by Sorts'
    plt.title(title)
    savename = savepath + '/img_by_sorts.png'
    plt.savefig(savename)
    plt.close()
    
#    ## Plot each sort, individually
#    for idx in range(1,len(all_sorts)+1):
#        plt.figure(figsize=[10,8])
#        
#        gt_overlay = np.zeros((gt_img.shape[0],gt_img.shape[1]))
#        gt_overlay[np.where(gt_img == True)] = 1
#        plt.imshow(gt_overlay, cmap='gray', alpha=1.0)
#        
#        sort_idx = np.where(data_out_sort_labels == idx)
#        sort_overlay = np.zeros((gt_img.shape[0],gt_img.shape[1]))
#        sort_overlay[sort_idx] = idx
#        plt.imshow(sort_overlay, cmap='plasma', alpha = 0.7)
#        plt.axis('off')
#        title = 'Sort: ' + str(all_sorts[idx-1])
#        plt.title(title, fontsize=16)
#        savename = savepath + '/sort_idx_' + str(idx) + '.png'
#        plt.savefig(savename)
#        plt.close()
        
else:
    for idx in range(data.shape[1]):
        data_out[idx] = chi.chi_quad(X_test[:,idx])
    

############################################################################
############################### Compute MSE ################################
############################################################################
## Compute sum of squared errors
diff = Y_test - data_out
err = np.dot(diff, diff)


###########################################################################
############################## Visualizations #############################
###########################################################################
colors = ['navy','darkorange']
newcmp = LinearSegmentedColormap.from_list("Custom", colors, N=256)


#plt.scatter(X0[neg_coords,0],X0[neg_coords,1], s=40, facecolor='none', edgecolor='navy')

plt.figure()
plt.scatter(X[:,0],X[:,1], s=80, c=data_out, marker='o',facecolors='none', edgecolors='face', cmap=newcmp)
#plt.axis('off')
plt.colorbar()
title = 'ChI Fusion Result'
plt.title(title)
savename = savepath + '/chi_result.png'
plt.savefig(savename)
plt.close()

plt.figure()
plt.scatter(X[:,0],X[:,1], s=80, c=data_out, marker='o',facecolors='none', edgecolors='face', cmap=newcmp)
#plt.axis('off')
plt.colorbar()
title = 'ChI Fusion Result; SSE: ' + str(np.round(err,3))
plt.title(title)
savename = savepath + '/chi_result_with_sse.png'
plt.savefig(savename)
plt.close()

#plt.figure()
#plt.scatter(X[:,0],X[:,1],c=err)
#plt.axis('off')
#plt.colorbar()
#title = 'ChI Fusion MSE Error'
#plt.title(title)
#savename = savepath + '/chi_mse_error_image.png'
#plt.savefig(savename)
#plt.close()







