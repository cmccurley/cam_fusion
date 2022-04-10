#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:55:50 2022

@author: cmccurley
"""

import os
import json
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.pairwise import manhattan_distances
from sklearn_extra.cluster import KMedoids
from skimage.filters import threshold_otsu


import dask.array as da
from cm_choquet_integral import ChoquetIntegral


def select_features_sort_entropy(indices, activation_set, all_activations, NUM_SOURCES, savepath):
    
    num_samples = activation_set.shape[1]* activation_set.shape[2]
    num_all_sources = all_activations.shape[0]
    
    ## Reshape activations
    activations = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
    for idk in range(all_activations.shape[0]):
        activations[idk,:] = all_activations[idk,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
        
    ## Find the feature which maximizes the entropy of the sort histogram 
    for k in range(activation_set.shape[0],NUM_SOURCES):
        
#        print(f'Selecting source {str(k+1)} of {str(NUM_SOURCES)}...')
        
        num_sources  = activation_set.shape[0] + 1 
        
        ## Update list of all possible sorts
        all_sources = list(np.arange(num_sources)+1)
        all_sorts = []
        all_sorts_tuple = list(itertools.permutations(all_sources))
        for sort in all_sorts_tuple:
            all_sorts.append(list(sort))
        
        feature_ent = np.zeros((num_all_sources))
        
        ## Used for fast feature selection (Can't use more than 9 sources)
        encoding_vect = 10**np.arange(0,num_sources)[::-1]
        bin_centers = np.dot(np.array(all_sorts),encoding_vect)
        hist_bins = [0] + list(bin_centers+1)  
        
#        hist_bins = list(np.arange(0,len(all_sorts)+1)+0.5)  
        
        ####### Get entropy of sorts adding each feature, incrementally #######
        for idx in range(0,num_all_sources):
#            
#            if not(idx % 200):
#                print(f'Source: {str(idx)}')
            
            current_indices = indices + [idx]
            
#            ## This code works but is really slow
#            ################## Get the sorts of each sample ###################
#            num_current_sources = len(current_indices)
#            
#            sorts_for_current_activation = np.zeros((num_samples))
#            
#            for ids in range(0,num_samples):
#                pi_i = np.argsort(activations[current_indices,ids])[::-1][:num_current_sources] + 1 ## Sort greatest to least
#                sort_idx = [idx for idx, val in enumerate(all_sorts) if (sum(pi_i == val ) == num_current_sources)]
#                sorts_for_current_activation[ids] = int(sort_idx[0]+1)
            
#           ################## Get the sorts of each sample ###################
            ## Sort activations for each sample
            pi_i = np.argsort(activations[current_indices,:],axis=0)[::-1,:] + 1 ## Sorts each sample from greatest to least

            ## Get normalized histogram
            sorts_for_current_activation = np.dot(pi_i.T,encoding_vect)
            hist, _ = np.histogram(sorts_for_current_activation, bins=hist_bins, density=False)
            hist += 1
            hist = hist/(num_samples + len(all_sorts))
            
#            ## This works but is slower
#            hist = np.zeros((len(all_sorts)))
#            
#            for ids in range(0,len(all_sorts)):
#                hist[ids] = np.count_nonzero((pi_i==np.expand_dims(np.array(all_sorts[ids]),axis=1)).all(axis=0))
#            
#            ## Normalize histogram
#            hist = hist + 1
#            hist = hist/(num_samples + len(all_sorts))
            
            ########## Compute the entropy of the sort histogram ##############
#            hist, bin_edges = np.histogram(sorts_for_current_activation, bins=hist_bins, density=True)
            feature_ent[idx] = -(hist*np.log(np.abs(hist))).sum()
            
#            ## Define custom colormap
#            custom_cmap = plt.cm.get_cmap('jet', len(all_sorts))
#            norm = matplotlib.colors.Normalize(vmin=1,vmax=len(all_sorts))
            
#            ## Plot frequency histogram of sorts for single image
#            sort_tick_labels = []
#            for element in all_sorts:
#                sort_tick_labels.append(str(element))
#            hist_bins = list(np.arange(0,len(all_sorts)+1)+0.5)
#            plt.figure(figsize=[10,8])
#            ax = plt.axes()
#            plt.xlabel('Sorts', fontsize=14)
#            plt.ylabel('Count', fontsize=14)
#            n, bins, patches = plt.hist(sorts_for_current_activation, bins=hist_bins, align='mid')
#            plt.xlim([0.5,len(all_sorts)+0.5])
#            ax.set_xticks(list(np.arange(1,len(all_sorts)+1)))
#            ax.set_xticklabels(sort_tick_labels,rotation=45,ha='right')
#            for idx, current_patch in enumerate(patches):
#                current_patch.set_facecolor(custom_cmap(norm(idx+1)))
#            title = 'Histogram of Sorts'
#            plt.title(title, fontsize=18)
#            savename = img_savepath + '/hist_of_sorts.png'
#            plt.savefig(savename)
#            plt.close()
        
        ## Sort entropy values and add feature with max entropy to set of sources (largest to smallest)
        sorted_corr_idx = (-feature_ent).argsort()
        
        ## Smallest to largest
#        sorted_corr_idx = sorted_corr_idx[::-1]
    
        new_idx = 0
        while((len(np.unique(all_activations[sorted_corr_idx[new_idx],:,:] >0.01)) < 2) or (sorted_corr_idx[new_idx] in indices)):
            new_idx += 1
        
        ## Add chosen source to set of sources
        indices.append(int(sorted_corr_idx[new_idx]))
        activation_set = np.concatenate((activation_set,np.expand_dims(all_activations[sorted_corr_idx[new_idx],:,:],axis=0)),axis=0)
        
#        ## Save parameters         
#        textfile = open(savepath, 'w')
#        for element in indices:
#            textfile.write(str(element) + " ")
#        textfile.close()
        
#        plt.figure()
#        plt.imshow(all_activations[sorted_corr_idx[new_idx],:,:])
#        plt.axis('off')

    return indices, activation_set, feature_ent[sorted_corr_idx[new_idx]]


def select_features_hag(indices, activation_set, all_activations, NUM_SOURCES, savepath):
    
    print('Selecting features with hierarchical clustering...')
    
    NUM_CLUSTERS = 20
    
    num_seeds = len(indices)
    
    
    kmedoids = KMedoids(n_clusters=3, metric='l1', random_state=12)
    
    cluster_savepath = savepath.replace('feature_indices_hag.txt','cluster_') 
 
#    ## Create new directory to save all activations
#    try:
#        new_directory = cluster_savepath + 'activations'
#        os.mkdir(new_directory)
#    except:
#        new_directory = cluster_savepath + 'activations'
#            
#    ## Plot all activations for reference to dendrogram
#    for idk in range(all_activations.shape[0]):
#        act_savepath = new_directory + '/activation_' + str(idk) + '.png'
#        title = 'Activation: ' + str(idk)
#        
#        plt.figure()
#        plt.imshow(all_activations[idk,:,:])
#        plt.axis('off')
#        plt.title(title)
#        plt.savefig(act_savepath)
#        plt.close()
    
    ## Reshape activations
    activations = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
    for idk in range(all_activations.shape[0]):
        activations[idk,:] = all_activations[idk,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
        
    ## Compute pairwise similarity matrix
    print('Computing dissimilarity matrix...')
    dist_mat = manhattan_distances(activations, sum_over_features=True)
    
    ## Compute linkage matrix
    print('Computing linkage...')
    Z = linkage(dist_mat, 'average')
    
#    ## Plot dendrogram
#    fig_savepath = cluster_savepath + 'dendrogram.png'
#    plt.figure(figsize=(20,20))
#    dn = dendrogram(Z, truncate_mode='lastp',p=12, show_contracted=True)
#    plt.savefig(fig_savepath)
#    plt.close()
    
    ## Get cluster labels for each activation map
    print('Clustering data...')
    labels = fcluster(Z,t=10,criterion='maxclust')
    
#    dend = dendrogram(linkage(activations,method='average'))
#    clustering = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, affinity='l1', linkage='average')
#    clustering.fit(activations)
#    labels = clustering.labels_
#    
    ## Plot activations, save in individual folder for each cluster
    all_cluster_indices = []
    centers = np.zeros((NUM_CLUSTERS,all_activations.shape[1],all_activations.shape[2]))
    for idk in range(1,NUM_CLUSTERS+1):
        cluster_indices = np.where(labels==idk)[0]
        cluster_activations = all_activations[cluster_indices,:,:]
        
        all_cluster_indices.append(cluster_indices)
        
#        ## Create new directory to save activations in cluster
#        try:
#            new_directory = cluster_savepath + str(idk)
#            os.mkdir(new_directory)
#        except:
#            new_directory = cluster_savepath + str(idk)
#        
#        for idx in range(len(cluster_indices)):
#            title = 'Cluster: ' + str(idk) + '; Activation: ' + str(cluster_indices[idx])
#            fig_savepath = new_directory + '/activation_' + str(cluster_indices[idx]) + '.png'
#            plt.imshow(cluster_activations[idx,:,:])
#            plt.axis('off')
#            plt.savefig(fig_savepath)
#            plt.close()
            
 
        ## Initialize cluster with mediod of clusters
        
        if (len(cluster_indices) == 1):
            center_idx = 0
            potential_features = np.expand_dims(all_activations[cluster_indices[center_idx],:,:],axis=0)
            potential_feature_indices = cluster_indices
        else:
            ## k-medoids of activations in cluster, pick center as 
            ## activation with the max sort entropy compared to the current set
            kmedoids.fit(activations[cluster_indices,:])
            potential_features_raveled = kmedoids.cluster_centers_
            
            potential_features = np.zeros((potential_features_raveled.shape[0],all_activations.shape[1],all_activations.shape[2]))
            for k in range(potential_features_raveled.shape[0]):
                potential_features[k,:,:] = potential_features_raveled[k,:].reshape((potential_features.shape[1],potential_features.shape[2]))
                
            potential_feature_indices = kmedoids.medoid_indices_
            potential_feature_indices = cluster_indices[potential_feature_indices]
            
            
        if not(idk):
            activation_set = potential_features
            indices = potential_feature_indices.tolist()
        else:
            activation_set = np.concatenate((activation_set, potential_features),axis=0)
            indices = indices + potential_feature_indices.tolist()
    
    
    activation_set = activation_set[num_seeds::,:,:]
    
    ## Remove zero activations
    valid_idx = []
    for k in range(activation_set.shape[0]):
        if np.count_nonzero(activation_set[k]> 0.01):
            valid_idx.append(k)
    
    activation_set = activation_set[valid_idx,:,:]
    
    ## Remove seeded values
    
    indices = indices[num_seeds::]
    
    indices = np.array(indices)[valid_idx]
    indices = indices.tolist()
    
    for idk in range(activation_set.shape[0]):
        title = 'Activation: ' + str(indices[idk])
        plt.figure()
        plt.imshow(activation_set[idk,:,:])
        plt.axis('off')
        plt.title(title)


    #######################################################################
    ################### Select Sources for Sort Entropy ###################
    #######################################################################
    
    ind_set = []
    act_set = []
    ent_set = []
    
    print('Picking features by sort entropy...')
    for k in range(activation_set.shape[0]):
        input_ind = [k]
        input_act_set = np.expand_dims(activation_set[k,:,:],axis=0)
        input_path = cluster_savepath + 'temp_file.txt'
        
        ind, act, ent = select_features_sort_entropy(input_ind, input_act_set, activation_set, NUM_SOURCES, input_path)
    
        ind_set.append(ind)
        act_set.append(act)
        ent_set.append(ent)
        
    top_order_ind = (-np.array(ent_set)).argsort()[0]
    
    return_indices = ind_set[top_order_ind]
    return_activation_set = act_set[top_order_ind]

    return return_indices, return_activation_set

def select_features_exhaustive_entropy(all_activations, NUM_SOURCES, savepath):
    
    print('Running exhaustive selection...')

    #######################################################################
    ################### Select Sources for Sort Entropy ###################
    #######################################################################
    
    ind_set = []
    act_set = []
    ent_set = []
    
    print('Picking features by sort entropy...')
    for k in tqdm(range(all_activations.shape[0])):
        
#        print(f'{str(k)}')
        input_ind = [k]
        input_act_set = np.expand_dims(all_activations[k,:,:],axis=0)
        input_path = savepath + '/this_file_doesnt_matter.txt'
        
        ind, act, ent = select_features_sort_entropy(input_ind, input_act_set, all_activations, NUM_SOURCES, input_path)
    
        ind_set.append(ind)
        act_set.append(act)
        ent_set.append(ent)
    
    top_order = (-np.array(ent_set)).argsort().astype(np.int16).tolist()
    top_order_ind = int(top_order[0])
    
    return_indices = ind_set[top_order_ind]
    return_activation_set = act_set[top_order_ind]
    
    results_savepath_text = savepath + '/results.txt'
    results_savepath_json = savepath + '/results.json'
    results_savepath_dict = savepath + '/results.npy'
    
    ## Accumulate results for saving
    results = dict()
    results['best_set_index'] = top_order_ind
    results['best_set_activations'] = return_indices
    results['best_set_ent'] = ent_set[top_order_ind]
    results['ind_set_all_unordered'] = ind_set
    results['ent_set_all_unordered'] = ent_set
    results['ind_ordered_max_entropy_to_min'] = top_order
    
    ## Save parameters         
    with open(results_savepath_text, 'w') as f:
        json.dump(results, f, indent=2)
    f.close   
    
    ## Save to numpy file
    np.save(results_savepath_dict, results, allow_pickle=True)
    
    ## save to JSON
    with open (results_savepath_json, 'w') as outfile:
        json.dump(results, outfile, indent=2)

    return return_indices, return_activation_set

def select_features_iou(indices, activation_set, all_activations, gt_img, labels, chi, NUM_SOURCES, savepath):
    
    num_all_sources = all_activations.shape[0]
    
    ## Reshape activations
    activations = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
    for idk in range(all_activations.shape[0]):
        activations[idk,:] = all_activations[idk,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
        
    ## Find the feature which maximizes the entropy of the sort histogram 
    for k in range(activation_set.shape[0],NUM_SOURCES):
        
        num_sources  = activation_set.shape[0] + 1 
        
        ## Update list of all possible sorts
        all_sources = list(np.arange(num_sources)+1)
        all_sorts = []
        all_sorts_tuple = list(itertools.permutations(all_sources))
        for sort in all_sorts_tuple:
            all_sorts.append(list(sort))
        
        feature_iou = np.zeros((num_all_sources))
        
        ####### Get entropy of sorts adding each feature, incrementally #######
        for ids in tqdm(range(0,num_all_sources)):
            
            current_indices = indices + [ids]
            
            current_activation_set = all_activations[current_indices,:,:]
            
            ## Train ChI via quadratic program 
            data = np.zeros((current_activation_set.shape[0],current_activation_set.shape[1]*current_activation_set.shape[2]))
    
            for idx in range(current_activation_set.shape[0]):
                data[idx,:] = np.reshape(current_activation_set[idx,:,:],(1,current_activation_set[idx,:,:].shape[0]*current_activation_set[idx,:,:].shape[1]))
            
            # train the chi via quadratic program 
            chi.train_chi(data, labels)
            
            ## Test on train and get IoU
            data_out = np.zeros((data.shape[1]))
            
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
                binary_feature_map = data_out.reshape((current_activation_set.shape[1],current_activation_set.shape[2])) > img_thresh
            except:
                binary_feature_map = data_out > 0.1
                
                if (binary_feature_map.shape[0] == 50176):
                    binary_feature_map = binary_feature_map.reshape((gt_img.shape[0],gt_img.shape[1]))
        
            ## Compute fitness as IoU to pseudo-groundtruth
            intersection = np.logical_and(binary_feature_map, gt_img)
            union = np.logical_or(binary_feature_map, gt_img)
            
            try:
                iou_score = np.sum(intersection) / np.sum(union)
            except:
                iou_score = 0
                
            ## Compute IoU
            feature_iou[ids] = iou_score
            
        ## Sort entropy values and add feature with max entropy to set of sources (largest to smallest)
        sorted_corr_idx = (-feature_iou).argsort()
    
        new_idx = 0
        while((len(np.unique(all_activations[sorted_corr_idx[new_idx],:,:] >0.01)) < 2) or (sorted_corr_idx[new_idx] in indices)):
            new_idx += 1
        
        ## Add chosen source to set of sources
        indices.append(int(sorted_corr_idx[new_idx]))
        activation_set = np.concatenate((activation_set,np.expand_dims(all_activations[sorted_corr_idx[new_idx],:,:],axis=0)),axis=0)

    return indices, activation_set, feature_iou[sorted_corr_idx[new_idx]]

def select_features_exhaustive_iou(all_activations, gt_img, NUM_SOURCES, savepath):
    
    print('\n Running exhaustive selection... \n')

    #######################################################################
    ################### Select Sources for Sort Entropy ###################
    #######################################################################
    
    labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
    
    ind_set = []
    act_set = []
    ent_set = []
    
    ## Define ChI instance
    chi = ChoquetIntegral()    
    
#    print('Picking features by sort entropy...')
    for k in tqdm(range(all_activations.shape[0])):
        
#        print(f'{str(k)}')
        input_ind = [k]
        input_act_set = np.expand_dims(all_activations[k,:,:],axis=0)
        input_path = savepath + '/this_file_doesnt_matter.txt'
        
        ind, act, ent = select_features_iou(input_ind, input_act_set, all_activations, gt_img, labels, chi, NUM_SOURCES, input_path)
    
        ind_set.append(ind)
        act_set.append(act)
        ent_set.append(ent)
    
    
    top_order = (-np.array(ent_set)).argsort().astype(np.int16).tolist()
    top_order_ind = int(top_order[0])
    
    return_indices = ind_set[top_order_ind]
    return_activation_set = act_set[top_order_ind]
    
    results_savepath_text = savepath + '/results.txt'
    results_savepath_json = savepath + '/results.json'
    results_savepath_dict = savepath + '/results.npy'
    
    ## Accumulate results for saving
    results = dict()
    results['best_set_index'] = top_order_ind
    results['best_set_activations'] = return_indices
    results['best_set_ent'] = ent_set[top_order_ind]
    results['ind_set_all_unordered'] = ind_set
    results['ent_set_all_unordered'] = ent_set
    results['ind_ordered_max_entropy_to_min'] = top_order
    
    ## Save parameters         
    with open(results_savepath_text, 'w') as f:
        json.dump(results, f, indent=2)
    f.close   
    
    ## Save to numpy file
    np.save(results_savepath_dict, results, allow_pickle=True)
    
    ## save to JSON
    with open (results_savepath_json, 'w') as outfile:
        json.dump(results, outfile, indent=2)

    return return_indices, return_activation_set

###############################################################################
############################# Otsu Verification ###############################
###############################################################################
    
def select_features_otsu(indices, activation_set, all_activations, gt_img, labels, chi, NUM_SOURCES, savepath):
    
    num_all_sources = all_activations.shape[0]
    
    ## Reshape activations
    activations = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
    for idk in range(all_activations.shape[0]):
        activations[idk,:] = all_activations[idk,:,:].reshape((all_activations.shape[1]*all_activations.shape[2]))
        
    ## Find the feature which maximizes the entropy of the sort histogram 
    for k in range(activation_set.shape[0],NUM_SOURCES):
        
        num_sources  = activation_set.shape[0] + 1 
        
        ## Update list of all possible sorts
        all_sources = list(np.arange(num_sources)+1)
        all_sorts = []
        all_sorts_tuple = list(itertools.permutations(all_sources))
        for sort in all_sorts_tuple:
            all_sorts.append(list(sort))
        
        feature_iou = np.zeros((num_all_sources))
        
        ####### Get entropy of sorts adding each feature, incrementally #######
        for ids in tqdm(range(0,num_all_sources)):
            
            new_savepath = savepath + '_plus_' + str(ids)
        
            try:
                os.mkdir(new_savepath)   
            except:
                print(' ')
            
            current_indices = indices + [ids]
            
            current_activation_set = all_activations[current_indices,:,:]
            
            ## Train ChI via quadratic program 
            data = np.zeros((current_activation_set.shape[0],current_activation_set.shape[1]*current_activation_set.shape[2]))
    
            for idx in range(current_activation_set.shape[0]):
                data[idx,:] = np.reshape(current_activation_set[idx,:,:],(1,current_activation_set[idx,:,:].shape[0]*current_activation_set[idx,:,:].shape[1]))
            
            # train the chi via quadratic program 
            chi.train_chi(data, labels)
            
            ## Test on train and get IoU
            data_out = np.zeros((data.shape[1]))
            
            for idx in range(data.shape[1]):
                data_out[idx] = chi.chi_quad(data[:,idx])
            
            ## Normalize ChI output
            data_out = data_out - np.min(data_out)
            data_out = data_out / (1e-7 + np.max(data_out))
            
            ## Plot saliency map
            plt.figure()
            plt.imshow(data_out.reshape((activation_set.shape[1],activation_set.shape[2])))
            plt.axis('off')
            plt.colorbar()
            title = 'Raw FusionCAM'
            plt.title(title)
            
            savename = new_savepath + '/raw_fusioncam.png'
                
            plt.savefig(savename)
            plt.close()
            
            ###########################################################################
            ############################## Compute IoU ################################
            ###########################################################################
            ## Binarize feature
            
            gt_img = gt_img > 0
            
            try:
                img_thresh = threshold_otsu(data_out)
                binary_feature_map = data_out.reshape((current_activation_set.shape[1],current_activation_set.shape[2])) > img_thresh
            except:
                binary_feature_map = data_out > 0.1
                
                if (binary_feature_map.shape[0] == 50176):
                    binary_feature_map = binary_feature_map.reshape((gt_img.shape[0],gt_img.shape[1]))
        
            ## Compute fitness as IoU to pseudo-groundtruth
            intersection = np.logical_and(binary_feature_map, gt_img)
            union = np.logical_or(binary_feature_map, gt_img)
            
            try:
                iou_score = np.sum(intersection) / np.sum(union)
            except:
                iou_score = 0
                
            ## Compute IoU
            feature_iou[ids] = iou_score
            
            
            
            ## Plot activation maps
            for idr in range(current_activation_set.shape[0]):
                plt.figure()
                plt.imshow(current_activation_set[idr,:,:])
                plt.axis('off')
                title = 'Activation idx: ' + str(idr+1)
                plt.title(title)
                
                savename = new_savepath + '/activation_idx_' + str(idr+1) + '.png'
                plt.savefig(savename)
                plt.close()
        
        
        
            otsu_image = np.zeros((binary_feature_map.shape[0],binary_feature_map.shape[1]))
            otsu_image[binary_feature_map] = 1
            
            ## Plot CAM over Otsu Image
            plt.figure()
            plt.imshow(otsu_image, cmap='gray')
            plt.axis('off')
            plt.colorbar()
            title = 'Otsu Thresh:  ' + str(img_thresh)
            plt.title(title)
            
            savename = new_savepath + '/otsu_threshold_image.png'
                
            plt.savefig(savename)
            plt.close()
            
            
            otsu_overlay_image = np.zeros((binary_feature_map.shape[0],binary_feature_map.shape[1]))
            otsu_overlay_image[binary_feature_map] =  data_out.reshape((current_activation_set.shape[1],current_activation_set.shape[2]))[binary_feature_map] 
            
            ## Plot CAM over Otsu Image
            plt.figure()
            plt.imshow(otsu_overlay_image)
            plt.axis('off')
            plt.colorbar()
            title = 'Otsu Thresh:  ' + str(img_thresh) + '; IoU: ' + str(iou_score)
            plt.title(title)
            
            savename = new_savepath + '/otsu_threshold_image_overlay.png'
                
            plt.savefig(savename)
            plt.close()
            
        ## Sort entropy values and add feature with max entropy to set of sources (largest to smallest)
        sorted_corr_idx = (-feature_iou).argsort()
    
        new_idx = 0
        while((len(np.unique(all_activations[sorted_corr_idx[new_idx],:,:] >0.01)) < 2) or (sorted_corr_idx[new_idx] in indices)):
            new_idx += 1
        
        ## Add chosen source to set of sources
        indices.append(int(sorted_corr_idx[new_idx]))
        activation_set = np.concatenate((activation_set,np.expand_dims(all_activations[sorted_corr_idx[new_idx],:,:],axis=0)),axis=0)
        
        

    return indices, activation_set, feature_iou[sorted_corr_idx[new_idx]]

def select_features_otsu_test(all_activations, gt_img, NUM_SOURCES, savepath):
    
    print('\n Running exhaustive selection... \n')

    #######################################################################
    ################### Select Sources for Sort Entropy ###################
    #######################################################################
    
    labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
    
    ind_set = []
    act_set = []
    ent_set = []
    
    ## Define ChI instance
    chi = ChoquetIntegral()    
    
#    print('Picking features by sort entropy...')
    for k in tqdm(range(all_activations.shape[0])):
        
#        print(f'{str(k)}')
        input_ind = [k]
        input_act_set = np.expand_dims(all_activations[k,:,:],axis=0)
        input_path = savepath + '/seed_' + str(k)
        
        ind, act, ent = select_features_otsu(input_ind, input_act_set, all_activations, gt_img, labels, chi, NUM_SOURCES, input_path)
    
        ind_set.append(ind)
        act_set.append(act)
        ent_set.append(ent)
    
    
    top_order = (-np.array(ent_set)).argsort().astype(np.int16).tolist()
    top_order_ind = int(top_order[0])
    
    return_indices = ind_set[top_order_ind]
    return_activation_set = act_set[top_order_ind]
    
    results_savepath_text = savepath + '/results.txt'
    results_savepath_json = savepath + '/results.json'
    results_savepath_dict = savepath + '/results.npy'
    
    ## Accumulate results for saving
    results = dict()
    results['best_set_index'] = top_order_ind
    results['best_set_activations'] = return_indices
    results['best_set_ent'] = ent_set[top_order_ind]
    results['ind_set_all_unordered'] = ind_set
    results['ent_set_all_unordered'] = ent_set
    results['ind_ordered_max_entropy_to_min'] = top_order
    
    ## Save parameters         
    with open(results_savepath_text, 'w') as f:
        json.dump(results, f, indent=2)
    f.close   
    
    ## Save to numpy file
    np.save(results_savepath_dict, results, allow_pickle=True)
    
    ## save to JSON
    with open (results_savepath_json, 'w') as outfile:
        json.dump(results, outfile, indent=2)

    return return_indices, return_activation_set

###############################################################################
############################# MIL Selection ###################################
###############################################################################

def select_features_divergence(indices, activation_set, all_activations_p, all_activations_n, NUM_SOURCES):
    
    num_samples = activation_set.shape[1]* activation_set.shape[2]
    num_all_sources = all_activations_p.shape[0]
    
    ## Reshape activations
    activations_p = np.zeros((all_activations_p.shape[0],all_activations_p.shape[1]*all_activations_p.shape[2]))
    for idk in range(all_activations_p.shape[0]):
        activations_p[idk,:] = all_activations_p[idk,:,:].reshape((all_activations_p.shape[1]*all_activations_p.shape[2]))
        
    activations_n = np.zeros((all_activations_n.shape[0],all_activations_n.shape[1]*all_activations_n.shape[2]))
    for idk in range(all_activations_n.shape[0]):
        activations_n[idk,:] = all_activations_n[idk,:,:].reshape((all_activations_n.shape[1]*all_activations_n.shape[2]))
        
    ## Find the feature which maximizes the entropy of the sort histogram 
    for k in range(activation_set.shape[0],NUM_SOURCES): ## Greedily add sources until desired number is reached
        
#        print(f'Selecting source {str(k+1)} of {str(NUM_SOURCES)}...')
        
        num_sources  = activation_set.shape[0] + 1 
        
        ## Update list of all possible sorts
        all_sources = list(np.arange(num_sources)+1)
        all_sorts = []
        all_sorts_tuple = list(itertools.permutations(all_sources))
        for sort in all_sorts_tuple:
            all_sorts.append(list(sort))
        
        divergence = np.zeros((num_all_sources))
        
        ## Used for fast feature selection (Can't use more than 9 sources)
        encoding_vect = 10**np.arange(0,num_sources)[::-1]
        bin_centers = np.dot(np.array(all_sorts),encoding_vect)
        hist_bins = [0] + list(bin_centers+1)  
        
#        hist_bins = list(np.arange(0,len(all_sorts)+1)+0.5)  
        
        ##### Get divergence of sorts adding each feature, incrementally ######
        for idx in range(0,num_all_sources):
            
            current_indices = indices + [idx]
            
            ####### Get the sorts histogram for the positive bags #############
            ## Sort activations for each sample
            pi_i_p = np.argsort(activations_p[current_indices,:],axis=0)[::-1,:] + 1 ## Sorts each sample from greatest to least

            ## Get normalized histogram
            sorts_for_current_activation_p = np.dot(pi_i_p.T,encoding_vect)
            hist_p, _ = np.histogram(sorts_for_current_activation_p, bins=hist_bins, density=False)
            hist_p += 1
            p = hist_p/(num_samples + len(all_sorts))
            
            ####### Get the sorts histogram for the negative bags #############
            ## Sort activations for each sample
            pi_i_n = np.argsort(activations_n[current_indices,:],axis=0)[::-1,:] + 1 ## Sorts each sample from greatest to least

            ## Get normalized histogram
            sorts_for_current_activation_n = np.dot(pi_i_n.T,encoding_vect)
            hist_n, _ = np.histogram(sorts_for_current_activation_n, bins=hist_bins, density=False)
            hist_n += 1
            q = hist_n/(num_samples + len(all_sorts))

            ########## Compute the divergence between histograms ##############
            ## KL(Pos||Neg), Deviation of positive from negative
            divergence[idx] = np.sum(np.multiply(p,np.log2(np.divide(p,q))))
        
#            ## KL(Neg||Pos), Deviation of negative from positive
#            divergence[idx] = np.sum(np.multiply(q,np.log2(np.divide(q,p))))
            
#            ## JS(Pos,Neg), Symmetric
#            m = 0.5*(p+q)
#            divergence[idx] = 0.5*np.sum(np.multiply(p,np.log2(np.divide(p,m)))) + 0.5*np.sum(np.multiply(q,np.log2(np.divide(q,m))))
            
        
        ## Sort entropy values and add feature with max entropy to set of sources (largest to smallest)
        sorted_corr_idx = (-divergence).argsort()
    
        new_idx = 0
        while((len(np.unique(all_activations_p[sorted_corr_idx[new_idx],:,:] >0.01)) < 2) or (sorted_corr_idx[new_idx] in indices)):
            new_idx += 1
        
        ## Add chosen source to set of sources
        indices.append(int(sorted_corr_idx[new_idx]))
        activation_set = np.concatenate((activation_set,np.expand_dims(all_activations_p[sorted_corr_idx[new_idx],:,:],axis=0)),axis=0)

    return indices, activation_set, divergence[sorted_corr_idx[new_idx]]

def select_features_divergence_pos_and_neg(all_activations_p, all_activations_n, NUM_SOURCES, savepath):
    
    print('\n Running exhaustive selection... \n')

    #######################################################################
    ################### Select Sources for Sort Entropy ###################
    #######################################################################
    
    num_all_sources = all_activations_p.shape[0]
    
    ind_set = []
    act_set = []
    ent_set = []
    
    ## Initialize selection with each source in positive bag
    for k in tqdm(range(num_all_sources)):
        
        input_ind = [k]
        input_act_set = np.expand_dims(all_activations_p[k,:,:],axis=0)
        
        ind, act, ent = select_features_divergence(input_ind, input_act_set, all_activations_p, all_activations_n, NUM_SOURCES)
    
        ind_set.append(ind)
        act_set.append(act)
        ent_set.append(ent)
    
    ## Order divergences from greatest to least
    top_order = (-np.array(ent_set)).argsort().astype(np.int16).tolist()
    top_order_ind = int(top_order[0])
    
    return_indices = ind_set[top_order_ind]
    return_activation_set = act_set[top_order_ind]
    
    results_savepath_text = savepath + '/results.txt'
    results_savepath_json = savepath + '/results.json'
    results_savepath_dict = savepath + '/results.npy'
    
    ## Accumulate results for saving
    results = dict()
    results['best_set_index'] = top_order_ind
    results['best_set_activations'] = return_indices
    results['best_set_ent'] = ent_set[top_order_ind]
    results['ind_set_all_unordered'] = ind_set
    results['ent_set_all_unordered'] = ent_set
    results['ind_ordered_max_entropy_to_min'] = top_order
    
    ## Save parameters         
    with open(results_savepath_text, 'w') as f:
        json.dump(results, f, indent=2)
    f.close   
    
    ## Save to numpy file
    np.save(results_savepath_dict, results, allow_pickle=True)
    
    ## save to JSON
    with open (results_savepath_json, 'w') as outfile:
        json.dump(results, outfile, indent=2)

    return return_indices, return_activation_set

###############################################################################
############################# Dask MIL Selection ##############################
###############################################################################

def dask_select_features_divergence(indices, activations_p, activations_n, NUM_SOURCES):
    
    num_samples_p = activations_p.shape[1] 
    num_samples_n = activations_n.shape[1] 
    num_all_sources = activations_p.shape[0]
    
#    ## Reshape activations
#    activations_p = np.zeros((all_activations_p.shape[0],all_activations_p.shape[1]*all_activations_p.shape[2]))
#    for idk in range(all_activations_p.shape[0]):
#        activations_p[idk,:] = all_activations_p[idk,:,:].reshape((all_activations_p.shape[1]*all_activations_p.shape[2]))
#        
#    activations_n = np.zeros((all_activations_n.shape[0],all_activations_n.shape[1]*all_activations_n.shape[2]))
#    for idk in range(all_activations_n.shape[0]):
#        activations_n[idk,:] = all_activations_n[idk,:,:].reshape((all_activations_n.shape[1]*all_activations_n.shape[2]))
        
    ## Find the feature which maximizes the entropy of the sort histogram 
    for k in range(len(indices),NUM_SOURCES): ## Greedily add sources until desired number is reached
        
#        print(f'Selecting source {str(k+1)} of {str(NUM_SOURCES)}...')
        
        num_sources  = len(indices)+ 1 
        
        ## Update list of all possible sorts
        all_sources = list(np.arange(num_sources)+1)
        all_sorts = []
        all_sorts_tuple = list(itertools.permutations(all_sources))
        for sort in all_sorts_tuple:
            all_sorts.append(list(sort))
        
        divergence = np.zeros((num_all_sources))
        
        ## Used for fast feature selection (Can't use more than 9 sources)
        encoding_vect = 10**np.arange(0,num_sources)[::-1]
        bin_centers = np.dot(np.array(all_sorts),encoding_vect)
        hist_bins = [0] + list(bin_centers+1)  
        
#        hist_bins = list(np.arange(0,len(all_sorts)+1)+0.5)  
        
        ##### Get divergence of sorts adding each feature, incrementally ######
        for idx in range(0,num_all_sources):
            
            current_indices = indices + [idx]
            
            ####### Get the sorts histogram for the positive bags #############
            ## Sort activations for each sample
            pi_i_p = np.argsort(activations_p[current_indices,:],axis=0)[::-1,:] + 1 ## Sorts each sample from greatest to least

            ## Get normalized histogram
            sorts_for_current_activation_p = np.dot(pi_i_p.T,encoding_vect)
            hist_p, _ = np.histogram(sorts_for_current_activation_p, bins=hist_bins, density=False)
            hist_p += 1
            p = hist_p/(num_samples_p + len(all_sorts))
            
            ####### Get the sorts histogram for the negative bags #############
            ## Sort activations for each sample
            pi_i_n = np.argsort(activations_n[current_indices,:],axis=0)[::-1,:] + 1 ## Sorts each sample from greatest to least

            ## Get normalized histogram
            sorts_for_current_activation_n = np.dot(pi_i_n.T,encoding_vect)
            hist_n, _ = np.histogram(sorts_for_current_activation_n, bins=hist_bins, density=False)
            hist_n += 1
            q = hist_n/(num_samples_n + len(all_sorts))

            ########## Compute the divergence between histograms ##############
            ## KL(Pos||Neg), Deviation of positive from negative
            divergence[idx] = np.sum(np.multiply(p,np.log2(np.divide(p,q))))
        
#            ## KL(Neg||Pos), Deviation of negative from positive
#            divergence[idx] = np.sum(np.multiply(q,np.log2(np.divide(q,p))))
            
#            ## JS(Pos,Neg), Symmetric
#            m = 0.5*(p+q)
#            divergence[idx] = 0.5*np.sum(np.multiply(p,np.log2(np.divide(p,m)))) + 0.5*np.sum(np.multiply(q,np.log2(np.divide(q,m))))
            
        
        ## Sort entropy values and add feature with max entropy to set of sources (largest to smallest)
        sorted_corr_idx = (-divergence).argsort()
    
        new_idx = 0
        while((da.sum(activations_p[sorted_corr_idx[new_idx],:] > 0.01).compute() < 2) or (sorted_corr_idx[new_idx] in indices)):
            new_idx += 1
        
        ## Add chosen source to set of sources
        indices.append(int(sorted_corr_idx[new_idx]))

    return indices, divergence[sorted_corr_idx[new_idx]]

def dask_select_features_divergence_pos_and_neg(all_activations_p, all_activations_n, NUM_SOURCES, savepath):
    
    print('\n Running exhaustive selection... \n')

    #######################################################################
    ################### Select Sources for Sort Entropy ###################
    #######################################################################
    
    num_all_sources = all_activations_p.shape[0]
    
    ind_set = []
    ent_set = []
    
    ## Initialize selection with each source in positive bag
    for k in tqdm(range(num_all_sources)):
        
        input_ind = [k]
        
        ind, ent = dask_select_features_divergence(input_ind, all_activations_p, all_activations_n, NUM_SOURCES)
    
        ind_set.append(ind)
        ent_set.append(ent)
    
    ## Order divergences from greatest to least
    top_order = (-np.array(ent_set)).argsort().astype(np.int16).tolist()
    top_order_ind = int(top_order[0])
    
    return_indices = ind_set[top_order_ind]
#    return_activation_set = act_set[top_order_ind]
    
    results_savepath_text = savepath + '/results.txt'
    results_savepath_json = savepath + '/results.json'
    results_savepath_dict = savepath + '/results.npy'
    
    ## Accumulate results for saving
    results = dict()
    results['best_set_index'] = top_order_ind
    results['best_set_activations'] = return_indices
    results['best_set_ent'] = ent_set[top_order_ind]
    results['ind_set_all_unordered'] = ind_set
    results['ent_set_all_unordered'] = ent_set
    results['ind_ordered_max_entropy_to_min'] = top_order
    
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