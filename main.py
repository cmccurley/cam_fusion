# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:46:23 2020

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  cnn_bag_classifier_main.py
    *  Name:  Connor H. McCurley
    *  Date:  2021-04-22
    *  Desc:  Trains MIL bag-level classifier based on ResNet18.
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
import json
import argparse
from tqdm import tqdm
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import precision_recall_fscore_support as prfs

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
import parameters
import initialize_network
import util
from util import convert_gt_img_to_mask, cam_img_to_seg_img
from dataloaders import loaderPASCAL, loaderDSIAC

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
    
    ## Import parameters 
    args = None
    parameters = parameters.set_parameters(args)
    
    parameter_file = parameters.outf.replace('output','baseline_experiments') + '/parameters_null.txt'
    results_file = parameters.outf.replace('output','baseline_experiments') + '/results_eigencam.txt'
    
#    ## Read parameters
#    parser = argparse.ArgumentParser()
#    parameters = parser.parse_args()
#    with open(parameter_file, 'r') as f:
#        parameters.__dict__ = json.load(f)
    
    ## Define data transforms
    if (parameters.run_mode == 'cam') or (parameters.run_mode == 'test-cams'):
        
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
    else:
        if (parameters.DATASET == 'dsiac'):
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        elif (parameters.DATASET == 'mnist'):
            parameters.MNIST_MEAN = (0.1307,)
            parameters.MNIST_STD = (0.3081,)
            transform = transforms.Compose([transforms.Resize(size=256),
                                            transforms.CenterCrop(224),
                                            transforms.Grayscale(3),
                                            transforms.ToTensor(), 
                                            transforms.Normalize(parameters.MNIST_MEAN, parameters.MNIST_STD)])
        else:
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Resize(256), 
                                            transforms.CenterCrop(224), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
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
         
#        parameters.all_classes = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4,
#        'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9,
#        'cow':10, 'diningtable':11, 'dog':12, 'horse':13,
#        'motorbike':14, 'person':15, 'pottedplant':16,
#        'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
#        
#        classes = parameters.bg_classes + parameters.target_classes
#        
#        parameters.NUM_CLASSES = len(classes)
#        
#        parameters.pascal_im_size = (224,224)
        
        trainset = datasets.MNIST(parameters.mnist_data_path, train=True,download=True,transform=transform)
        validset = datasets.MNIST(parameters.mnist_data_path, train=False,download=True,transform=transform)
        testset = datasets.MNIST(parameters.mnist_data_path, train=True,download=True,transform=transform)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False) 
    

    ## Define files to save epoch training/validation
    logfilename = parameters.outf + parameters.loss_file

    ######################################################################
    ####################### Initialize Network ###########################
    ######################################################################
    
    model = initialize_network.init(parameters)
    
#    ## Train entire model for IR data
#    if (parameters.DATASET == 'dsiac'):
#        for param in model.features.parameters():
#            param.requires_grad = True
    
    ## Save initial weights for further training
    temptrain = parameters.outf + parameters.parameter_path
    torch.save(model.state_dict(), temptrain)
    
    ## Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if (parameters.run_mode == 'train'):
        ######################################################################
        ######################## Fine Tune Model #############################
        ######################################################################
        
        print('================= Training =================\n')
            
        ################ Define Model and Training Criterion #############
         
        ## Define loss function (BCE for binary classification)
#        criterion = nn.BCELoss()
        criterion = nn.CrossEntropyLoss()
#        criterion = nn.BCEWithLogitsLoss()
        
        ## Define optimizer for weight updates
        optimizer = optim.Adamax(model.parameters(), lr=parameters.LR)
#        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        ## Set to GPU
        if parameters.cuda:
            criterion.cuda()     
            model.cuda()
        
        ###################### Train the Network ########################
        
        ## Initialize loss data structures
        valid_loss = torch.zeros(parameters.EPOCHS+1)
        train_loss = torch.zeros(parameters.EPOCHS+1) 
        
        ## Create file to display epoch loss
        f = open(logfilename, 'w')
        f.close()
        
        model.train()
        
        ## Train for the desired number of epochs
        print('Fine tuning model...')
        for epoch in range(0,parameters.EPOCHS+1):

            running_loss = 0.0
            for i, data in enumerate(tqdm(train_loader)):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                
            epoch_loss = running_loss/len(train_loader)       
            
            print(f'=========== Epoch: {epoch} ============')
            
            
            # Display training and validation loss to terminal and update loss file
            if not(epoch % parameters.update_on_epoch):
                model.eval()
                temptrain = parameters.outf + '/model_eoe_' + str(epoch) + '.pth'
                torch.save(model.state_dict(), temptrain)
                util.print_status(epoch, epoch_loss, train_loader, valid_loader, model, device, logfilename)
                model.train()
        
    elif (parameters.run_mode == 'test'):
        ######################################################################
        ########################### Test Model ###############################
        ######################################################################
        
        print('================== Testing ==================\n')
        
        model.eval()
        
        ############################# Total Accuracy #########################
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i,data in enumerate(tqdm(test_loader)):
                images, labels = data[0].to(device), data[1].to(device)
                
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
#                images = images.permute(2,3,1,0)
#                images = images.detach().cpu().numpy()
#                image = images[:,:,:,0]
#                plt.imshow(image)
#                plt.axis('off')
#                
#                if (labels.detach().cpu().numpy() == 1):
#                    savename = parameters.outf + '/dataimages/aeroplane_' + str(i) 
#                    
#                elif (labels.detach().cpu().numpy() == 0):
#                    savename = parameters.outf + '/dataimages/cat' + str(i)
#                
#                plt.savefig(savename)
#                plt.close()
        
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        
        ######################### Per-class Performance ######################

        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():
            for data in (tqdm(test_loader)):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
        
        ## Print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))


    
    elif (parameters.run_mode == 'cam'):
        ######################################################################
        ########################### Compute CAMs #############################
        ######################################################################
        
        import numpy as np
        from cam_functions import GradCAM, LayerCAM
        from cam_functions.utils.image import show_cam_on_image, preprocess_image

        ## Turn on gradients for CAM computation 
        for param in model.parameters():
            param.requires_grad = True
    
        model.eval()
        
        ## Define CAMs for ResNet18
        if (parameters.model == 'resnet18'):
#            cam0 = GradCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=parameters.cuda)
#            cam1 = GradCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=parameters.cuda)
#            cam2 = GradCAM(model=model, target_layers=[model.layer2[-1]], use_cuda=parameters.cuda)
#            cam3 = GradCAM(model=model, target_layers=[model.layer3[-1]], use_cuda=parameters.cuda)
#            cam4 = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=parameters.cuda)
            
            cam0 = LayerCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=parameters.cuda)
            cam1 = LayerCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=parameters.cuda)
            cam2 = LayerCAM(model=model, target_layers=[model.layer2[-1]], use_cuda=parameters.cuda)
            cam3 = LayerCAM(model=model, target_layers=[model.layer3[-1]], use_cuda=parameters.cuda)
            cam4 = LayerCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=parameters.cuda)
    
        elif (parameters.model == 'vgg16'):
#            cam0 = GradCAM(model=model, target_layers=[model.features[4]], use_cuda=parameters.cuda)
#            cam1 = GradCAM(model=model, target_layers=[model.features[9]], use_cuda=parameters.cuda)
#            cam2 = GradCAM(model=model, target_layers=[model.features[16]], use_cuda=parameters.cuda)
#            cam3 = GradCAM(model=model, target_layers=[model.features[23]], use_cuda=parameters.cuda)
#            cam4 = GradCAM(model=model, target_layers=[model.features[30]], use_cuda=parameters.cuda)
            
            cam0 = LayerCAM(model=model, target_layers=[model.features[4]], use_cuda=parameters.cuda)
            cam1 = LayerCAM(model=model, target_layers=[model.features[9]], use_cuda=parameters.cuda)
            cam2 = LayerCAM(model=model, target_layers=[model.features[16]], use_cuda=parameters.cuda)
            cam3 = LayerCAM(model=model, target_layers=[model.features[23]], use_cuda=parameters.cuda)
            cam4 = LayerCAM(model=model, target_layers=[model.features[30]], use_cuda=parameters.cuda)
        
        
        if (parameters.DATASET == 'mnist'):
            parameters.MNIST_MEAN = (0.1307,)
            parameters.MNIST_STD = (0.3081,)
            norm_image = transforms.Normalize(parameters.MNIST_MEAN, parameters.MNIST_STD)
            gt_transform = transforms.Grayscale(1)
            
        else:
            norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
        target_category = None
        img_idx = 0
        for data in tqdm(test_loader):
            
                if (parameters.DATASET == 'mnist'):
                    images, labels = data[0].to(device), data[1].to(device)
                else:
                    images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                ## Get predicted class label
                output = model(cam_input)
                pred_label = np.argmax(output.detach().cpu().numpy()[0,:])
                
#                pred_label = int(labels.detach().cpu().numpy())
#                if pred_label:
#                    pred_label = 0
#                elif not(pred_label):
#                    pred_label = 1
           
                ## Get CAMs
                grayscale_cam0 = cam0(input_tensor=cam_input,
                                target_category=int(pred_label))
                grayscale_cam1 = cam1(input_tensor=cam_input,
                                target_category=int(pred_label))
                grayscale_cam2 = cam2(input_tensor=cam_input,
                                target_category=int(pred_label))
                grayscale_cam3 = cam3(input_tensor=cam_input,
                                target_category=int(pred_label))
                grayscale_cam4 = cam4(input_tensor=cam_input,
                                target_category=int(pred_label))

                # Here grayscale_cam has only one image in the batch
                grayscale_cam0 = grayscale_cam0[0, :]
                grayscale_cam1 = grayscale_cam1[0, :]
                grayscale_cam2 = grayscale_cam2[0, :]
                grayscale_cam3 = grayscale_cam3[0, :]
                grayscale_cam4 = grayscale_cam4[0, :]
                
                ## Visualize input and CAMs
                images = images.permute(2,3,1,0)
                images = images.detach().cpu().numpy()
                image = images[:,:,:,0]
                img = image
                
                if (parameters.DATASET == 'mnist'):
                    gt_img = Image.fromarray(np.uint8(img*255))
                    gt_images = gt_transform(gt_img)
                    gt_image = np.asarray(gt_images)/255
                    
                    gt_image[np.where(gt_image>0.2)] = 1
                    gt_image[np.where(gt_image<=0.1)] = 0
                    
                else:
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
                
#                ## Show image and groundtruth
#                fig, axis = plt.subplots(nrows=1, ncols=2)
#                axis[0].imshow(img)
#                axis[1].imshow(gt_image, cmap='gray')
#                axis[0].axis('off')
#                axis[1].axis('off')

                ## Plot CAMs (VGG16)
                fig, axis = plt.subplots(nrows=1, ncols=6)
                
                axis[0].imshow(img) 
    
                cam_image0 = show_cam_on_image(img, grayscale_cam0, True)
                cam_image1 = show_cam_on_image(img, grayscale_cam1, True)
                cam_image2 = show_cam_on_image(img, grayscale_cam2, True)
                cam_image3 = show_cam_on_image(img, grayscale_cam3, True)
                cam_image4 = show_cam_on_image(img, grayscale_cam4, True)
                    
                axis[1].imshow(cam_image0, cmap='jet')
                axis[2].imshow(cam_image1, cmap='jet')
                axis[3].imshow(cam_image2, cmap='jet')
                axis[4].imshow(cam_image3, cmap='jet')
                axis[5].imshow(cam_image4, cmap='jet')
                
                axis[0].axis('off')
                axis[1].axis('off')
                axis[2].axis('off')
                axis[3].axis('off')
                axis[4].axis('off')
                axis[5].axis('off')
                
#                title = 'Pred: ' + classes[pred_label]
                title = 'Input'
                axis[0].set_title(title)
                axis[1].set_title('Layer 1')
                axis[2].set_title('Layer 2')
                axis[3].set_title('Layer 3')
                axis[4].set_title('Layer 4')
                axis[5].set_title('Layer 5')
                title2 = 'Pred: ' + classes[pred_label] + ' / Actual: ' + classes[int(labels[0].detach().cpu())]
                fig.suptitle(title2)
                
#                ## Plot CAMs (ResNet18)
#                fig, axis = plt.subplots(nrows=1, ncols=5)
#                
#                axis[0].imshow(img) 
#    
#                cam_image1 = show_cam_on_image(img, grayscale_cam1, True)
#                cam_image2 = show_cam_on_image(img, grayscale_cam2, True)
#                cam_image3 = show_cam_on_image(img, grayscale_cam3, True)
#                cam_image4 = show_cam_on_image(img, grayscale_cam4, True)
#                    
#                axis[1].imshow(cam_image1, cmap='jet')
#                axis[2].imshow(cam_image2, cmap='jet')
#                axis[3].imshow(cam_image3, cmap='jet')
#                axis[4].imshow(cam_image4, cmap='jet')
#                
#                axis[0].axis('off')
#                axis[1].axis('off')
#                axis[2].axis('off')
#                axis[3].axis('off')
#                axis[4].axis('off')
#                
##                title = 'Pred: ' + classes[pred_label]
#                title = 'Input'
#                axis[0].set_title(title)
#                axis[1].set_title('Layer 1')
#                axis[2].set_title('Layer 2')
#                axis[3].set_title('Layer 3')
#                axis[4].set_title('Layer 4')
#               
       
                savename = parameters.outf
                savename = savename.replace('output','cams/layercam/cams/')
                savename = savename + 'img_' + str(img_idx)
                fig.savefig(savename)
                
                plt.close()
                
                img_idx += 1
                
                if not(img_idx%25):
                    print(str(img_idx))
                    
                
    elif (parameters.run_mode == 'test-cams'):
        ######################################################################
        ########################### Compute CAMs #############################
        ######################################################################
        import numpy as np
        import matplotlib.pyplot as plt
        from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
        from cam_functions.utils.image import show_cam_on_image, preprocess_image
        
        results_dict = dict()
        visualization_results = dict()
        visualization_results['thresholds'] = parameters.CAM_SEG_THRESH
        
        norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "eigencam": EigenCAM,
         "layercam": LayerCAM}
        
        if (parameters.model == 'vgg16'):
            cam_layers = \
            {4:"layer1",
             9:"layer2",
             16:"layer3",
             23:"layer4",
             30:"layer5"}

        ## Turn on gradients for CAM computation 
        for param in model.parameters():
            param.requires_grad = True
    
        model.eval()
        
        for current_cam in parameters.cams:
            for layer in parameters.layers:      
                for thresh in parameters.CAM_SEG_THRESH:
                
                    n_samples = len(test_loader)
                    metric_iou = np.zeros(n_samples, dtype="float32")
                    metric_precision = np.zeros(n_samples, dtype="float32")
                    metric_recall = np.zeros(n_samples, dtype="float32")
                    metric_f1_score = np.zeros(n_samples, dtype="float32")
                    
                    current_model = current_cam + '_' + cam_layers[layer]
                    print(current_model+'_thresh_'+str(thresh))
                
                    cam_algorithm = methods[current_cam]
                
                    if (parameters.model == 'vgg16'):
                        cam = cam_algorithm(model=model, target_layers=[model.features[layer]], use_cuda=parameters.cuda)
        
                    target_category = None
                    
                    idx = 0
                    for data in tqdm(test_loader):
                  
                            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                            
                            ############# Convert groundtruth image into mask #############
                            gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
                            
                            ## Normalize input to the model
                            cam_input = norm_image(images)
                            
                            ##################### Get CAM Segmentaiton ####################
                            grayscale_cam = cam(input_tensor=cam_input, target_category=int(labels))
                            grayscale_cam = grayscale_cam[0, :]
                            
                            ## Binarize CAM for segmentation
                            pred_img = cam_img_to_seg_img(grayscale_cam, thresh)
                            
                            ##################### Evaluate segmentation ###################
                            intersection = np.logical_and(pred_img, gt_img)
                            union = np.logical_or(pred_img, gt_img)
                            
                            ## Catch divide by zero(union of prediction and groundtruth is empty)
                            try:
                                iou_score = np.sum(intersection) / np.sum(union)
                            except:
                                iou_score = 0
                                
                            metric_iou[idx] = round(iou_score,5)
                            
                            
#                            prec, rec, f1, _ = prfs(gt_img.reshape(gt_img.shape[0]*gt_img.shape[1]), 
#                                                    pred_img.reshape(pred_img.shape[0]*pred_img.shape[1]),
#                                                    pos_label=1,
#                                                    average='binary') 
                            prec = 0
                            rec = 0
                            f1 = 0
                            
                            metric_precision[idx], metric_recall[idx], metric_f1_score[idx] = round(prec,5), round(rec,5), round(f1,5)
                            
                            idx +=1
                            
                            images.detach()
                            labels.detach()
                            gt_images.detach()
                            cam_input.detach()
                            
                            
                    ## Compute statistics over the entire test set
                    metric_iou_mean = round(np.mean(metric_iou),3)
                    metric_iou_std = round(np.std(metric_iou),3)
                    metric_precision_mean = round(np.mean(metric_precision),3)
                    metric_precision_std = round(np.std(metric_precision),3)
                    metric_recall_mean = round(np.mean(metric_recall),3)
                    metric_recall_std = round(np.std(metric_recall),3)
                    metric_f1_score_mean = round(np.mean(metric_f1_score),3)
                    metric_f1_score_std = round(np.std(metric_f1_score),3)
                    
                    
#                    ## Amalgamate results into one dictionary
#                    details = {'method': current_model,
#                               'threshold': thresh,
#                               'iou mean':metric_iou_mean,
#                               'iou std':metric_iou_std,
#                               'precision mean':metric_precision_mean,
#                               'precision std':metric_precision_std,
#                               'recall mean':metric_recall_mean,
#                               'recall std':metric_recall_std,
#                               'f1 score mean':metric_f1_score_mean,
#                               'f1 score std':metric_f1_score_std}
                    
                    ## Amalgamate results into one dictionary
                    details = {'method': current_model,
                               'threshold': thresh,
                               'iou mean':metric_iou_mean,
                               'iou std':metric_iou_std}
                    
                    ## Save results to global dictionary of results
                    model_name = current_model + '_thresh_' + str(thresh)
                    results_dict[model_name] = details
                    
                    ## Write results to text file
                    with open(results_file, 'a+') as f:
                        for key, value in details.items():
                            f.write('%s:%s\n' % (key, value))
                        f.write('\n')
                        f.close()
                    
                    ## Clean up memory
                    del details
                    del cam
                    del grayscale_cam
                    del metric_iou_mean
                    del metric_iou_std
                    del metric_precision_mean
                    del metric_precision_std
                    del metric_recall_mean
                    del metric_recall_std
                    del metric_f1_score_mean
                    del metric_f1_score_std
                    del metric_iou 
                    del metric_precision 
                    del metric_recall 
                    del metric_f1_score 
                    del cam_algorithm
          
                    torch.cuda.empty_cache()
        
                ## Save parameters         
                with open(parameter_file, 'w') as f:
                    json.dump(parameters.__dict__, f, indent=2)
                f.close 
                    
                ## Save results
                results_savepath = parameters.outf.replace('output','baseline_experiments') + '/results_eigencam.npy'
                np.save(results_savepath, results_dict, allow_pickle=True)
    
#    results_savepath = parameters.outf.replace('output','baseline_experiments_t_aeroplane_b_cat') + '/visualization_results.npy'
#    np.save(results_savepath, visualization_results, allow_pickle=True)
    
    
        
#                ## Visualize input and CAMs
#                images = images.permute(2,3,1,0)
#                images = images.detach().cpu().numpy()
#                image = images[:,:,:,0]
#                img = image
                
#                #######################################
#                ## Show image and groundtruth
#                fig, axis = plt.subplots(nrows=1, ncols=2)
#                axis[0].imshow(img)
#                axis[1].imshow(gt_image, cmap='gray')
#                axis[0].axis('off')
#                axis[1].axis('off')
#
#                ## Plot CAMs
#                fig, axis = plt.subplots(nrows=1, ncols=2)
#                axis[0].imshow(img) 
#                cam_image = show_cam_on_image(img, grayscale_cam, True)
#                axis[1].imshow(cam_image, cmap='jet')
#                ########################################
                
#                plt.figure()
#                plt.imshow(gt_image,cmap='gray')
#                plt.title('GT Mask')
#                
#                plt.figure()
#                plt.imshow(norm_cam,cmap='gray')
#                plt.title('CAM Segmentation')
    
    