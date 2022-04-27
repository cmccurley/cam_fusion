#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:52:06 2022

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

######################################################################
######################### Import Packages ############################
######################################################################
import math
import numpy as np
import itertools
from scipy.special import erfinv, erfc
from copy import deepcopy

from joblib import Parallel, delayed
import multiprocessing as mp

### Import numba for GPU optimization 
#from numba import jit
#from numba import cuda
#from numba import njit, prange

######################################################################
####################### Function Definitions #########################
######################################################################

class MIChoquetIntegral:

    def __init__(self):
        """
        ===========================================================================
        Instantiation of a MI Choquet Integral.

        This sets up the ChI. To instatiate, use
        chi = MIChoquetIntegral()
        ===========================================================================
        """


    def train_chi_softmax(self, Bags, Labels, Parameters):
        """
        ===========================================================================
        This trains the instance of the MI Choquet Integral by obtimizing the 
        generalized-mean model.

        :param Bags: This is structure of the training bags of size 1 x B
        :param Labels: These are the bag-level training labels of size 1 x B
        ===========================================================================
        """
        self.type = 'generalized-mean'
        self.B = len(Bags) ## Number of training bags
        self.N = Bags[0].shape[1] ## Number of sources
        self.M = [Bags[k].shape[0] for k in range(len(Bags))]  ## Number of samples in each bag
        self.p = Parameters.p ## Parameters on softmax fitness function
        
        ## Create dictionary of sorts
        all_sources = list(np.arange(self.N)+1)
        all_sorts = []
        all_sorts_tuple = list(itertools.permutations(all_sources))
        for sort in all_sorts_tuple:
            all_sorts.append(list(sort))
        self.all_sorts = all_sorts
        self.nElements = (2**(self.N))-1 ## number of measure elements, including mu_all=1
        
        ## Initialize dictionary of fuzzy measure elements
        self.index_keys = self.get_keys_index()
        
        ## Get measure lattice information - Number of elements in each tier,
        ## the indices of the elements, and the cumulative sum of elements
        measureEach = []
        measureNumEach = np.zeros(self.N).astype('int64')
        
        vls = np.arange(1, self.N + 1)
        count = 0
        Lattice = {}
       
        for i in range(0, self.N):
            Lattice[str(np.array([vls[i]]))] = count
            count = count + 1
        measureEach.append(vls)
    
        measureNumEach[0] = int(count)
            
        for i in range(2, self.N):
            count = 0
            A = np.array(list(itertools.combinations(vls, i)))
            measureEach.append(A)
            for latt_pt in A:
                Lattice[str(latt_pt)] = count
                count = count + 1
            
            measureNumEach[i-1] = int(count)
        
        measureNumEach[-1] = 1
        measureEach.append(np.arange(1,self.N+1))
        
        measureNumCumSum = np.zeros(self.N)
        
        for i in range(self.N):
            measureNumCumSum[i] = np.sum(measureNumEach[:i])
            
        self.measureNumEach = measureNumEach ## Number in each tier of lattice, bottom-up
        self.measureNumCumSum = measureNumCumSum ## Cumulative sum of measures, bottom-up
        self.measureEach = measureEach ## The actual indices

        
        ## Estimate measure parameters using evolutionary algorithm
        print("Number Sources : ", self.N, "; Number Training Bags : ", self.B)
        measure, initialMeasure, Analysis = self.produce_lattice_softmax(Bags, Labels, Parameters)
        
        fm = dict()
        for key in self.index_keys:
            fm[key] = measure[self.index_keys[key]]
        
        self.measure = measure
        self.initialMeasure = initialMeasure
        self.Analysis = Analysis
        self.fm = fm

        return self

    def compute_chi(self, Bag, nBags, measure):
        """
        ===========================================================================
        This will produce an output of the ChI

        This will use the learned(or specified) Choquet integral to
        produce an output w.r.t. to the new input.

        :param Bag: testing bag of size (n_samples X n_sources)
        :return: output of the choquet integral.
        ===========================================================================
        """
        
        #######################################################################
        #################### Set up measure for look-up #######################
        #######################################################################
        index_keys = self.index_keys
        fm = dict()
        for key in index_keys:
            fm[key] = measure[index_keys[key]]

        #######################################################################
        ######################## Compute ChI Vectorized #######################
        #######################################################################
        
        ## Set up variables
        nPntsBags = self.M
        nSources = self.N
        measureNumEach = self.measureNumEach ## Number in each tier of lattice, bottom-up
        measureNumCumSum = self.measureNumCumSum ## Cumulative sum of measures, bottom-up
        measureEach = self.measureEach ## The actual indices
        
        ## Precompute differences and indices for each bag (diffM = difference in measures)
        diffM = []
        for i in range(nBags):
            
            if (nBags == 1):
                bag = deepcopy(Bag)
            else:
                bag = deepcopy(Bag[i])
                
            if not(i):
                tmp = np.zeros((nBags,bag.shape[1]-1),dtype=np.object)
            
            ## Sort values in descending order
            indx = (bag).argsort(axis=1)
            indx = indx[:,::-1]
            v = np.sort(bag,axis=1)[:,::-1]
            
            vz = np.concatenate((np.zeros((v.shape[0],1)), v), axis=1) - np.concatenate((v,np.zeros((v.shape[0],1))), axis=1)
            diffM.append(vz[:,1::])
            for j in range((diffM[i].shape[1]-1)): ## # of sources in the combination (e.g. j=1 for g_1, j=2 for g_12)
                
                tmp_arr = np.zeros((bag.shape[0],len(np.arange(0,j+1))))
                
                for n in range(bag.shape[0]):
                    tmp_arr[n,:] = np.sort(indx[n,0:j+1])

                tmp[i,j] = tmp_arr
        
        sec_start_inds = np.zeros(nSources, dtype='int')
        nElem_prev = 0
        for j in range(nSources-1):
            if not(j): ## singleton        
                sec_start_inds[j] = 0
            else:  ## non-singleton
                nElem_prev = nElem_prev + (measureNumEach[j-1]-1)
                sec_start_inds[j] = nElem_prev
      
        bag_row_ids = np.zeros((nBags,),dtype=np.object)
      
#        bag_row_ids = [] ## the row indices of measure used for each bag
        for i in range(nBags):
            nPnts1, nSources = diffM[i].shape
            tmp_row_ids = np.zeros((nPnts1,nSources-1), dtype='int16')
            
            for n in range(bag.shape[0]):             
                for j in range(nSources-1): 
                    if not(j):
                        tmp_row_ids[n,j] = tmp[i,j][n,0]
                    else:  ## non-singleton
                        elem = measureEach[j] - 1 ## the number of combinations, e.g., (1,2),(1,3),(2,3)
                        for k in range(elem.shape[0]):
                            if (np.sum(tmp[i,j][n,:].astype('int') == (elem[k,:])) == elem.shape[1]):
                                row_id = k+1
                    
                                tmp_row_ids[n,j] = sec_start_inds[j] + row_id
            
            if (nBags == 1):
                bag_row_ids = np.zeros((nBags,),dtype=np.object)
                bag_row_ids[0] = tmp_row_ids
            else:
                bag_row_ids[i] = tmp_row_ids

        ## Create oneV cell matrix
        oneV = []
        for i in range(nBags):
            oneV.append(np.ones((nPntsBags[i],1)))
 
        diffM_ns = deepcopy(diffM)
        bag_row_ids_ns = deepcopy(bag_row_ids)
        
        ## Compute CI for non-singleton bags
        for i in range(len(diffM_ns)):
            ci = np.sum(np.multiply(diffM_ns[i],np.concatenate((measure[bag_row_ids_ns[i]], oneV[i]),axis=1)),axis=1)
            
            if not(i):
                y = ci
            else:
                y = np.concatenate((y,ci))
   
#        #######################################################################
#        ##################### Compute ChI on Single Bag #######################
#        #######################################################################
#        if (nBags == 1): ## Single bag
#            ch = np.zeros(Bag.shape[0])
#        
#            for inst_idx in range(Bag.shape[0]):
#                x = Bag[inst_idx,:]
#                n = len(x)
#                pi_i = np.argsort(x)[::-1][:n] + 1
#                ch[inst_idx] = x[pi_i[0] - 1] * (fm[str(pi_i[:1])])
#                for i in range(1, n):
#                    latt_pti = np.sort(pi_i[:i + 1])
#                    latt_ptimin1 = np.sort(pi_i[:i])
#                    ch[inst_idx] = ch[inst_idx] + x[pi_i[i] - 1] * (fm[str(latt_pti)] - fm[str(latt_ptimin1)])
#            
#            y = ch
#        
#        #######################################################################
#        #################### Compute ChI on Multiple Bags #####################
#        #######################################################################
#        else: ## Multiple bags, concatenate outputs
#            for b in range(nBags):
#            
#                ch = np.zeros(Bag[b].shape[0])
#        
#                for inst_idx in range(Bag[b].shape[0]):
#                    x = Bag[b][inst_idx,:]
#                    n = len(x)
#                    pi_i = np.argsort(x)[::-1][:n] + 1
#                    ch[inst_idx] = x[pi_i[0] - 1] * (fm[str(pi_i[:1])])
#                    for i in range(1, n):
#                        latt_pti = np.sort(pi_i[:i + 1])
#                        latt_ptimin1 = np.sort(pi_i[:i])
#                        ch[inst_idx] = ch[inst_idx] + x[pi_i[i] - 1] * (fm[str(latt_pti)] - fm[str(latt_ptimin1)])
#                
#                if not(b):
#                    y = ch
#                else:
#                    y = np.concatenate((y,ch))
            
        return y
    
    def produce_lattice_softmax(self, Bags, Labels, Parameters, trueInitMeasure=None):
        
        """ 
        ===============================================================================
        %% learnCIMeasure_softmax()
        % This function learns a fuzzy measure for Multiple Instance Choquet Integral (MICI)
        % This function works with classifier fusion
        % This function uses a generalized-mean objective function and an evolutionary algorithm to optimize.
        %
        % INPUT
        %    InputBags        - 1xNumTrainBags cell    - inside each cell, NumPntsInBag x nSources double. Training bags data.
        %    InputLabels      - 1xNumTrainBags double  -  Bag-level training labels. 
        %                   If values of "1" and "0", transform into regression bags where each negative point forms a bag.
        %    Parameters                 - 1x1 struct - The parameters set by learnCIMeasureParams() function.
        %    trueInitMeasure (optional) - 1x(2^nSource-1) double - true initial Measureinput
        %
        % OUTPUT
        %    measure             - 1x(2^nSource-1) double - learned measure by MICI
        %    initialMeasure      - 1x(2^nSource-1) double - initial measure by MICI during random initialization
        %    Analysis (optional) - 1x1 struct - records all the intermediate results if Parameters.analysis == 1 
        %                 Analysis.ParentChildMeasure - the pool of both parent and child measures across iterations
        %                 Analysis.ParentChildFitness - the pool of both parent and child fitness values across iterations  
        %                 Analysis.ParentChildTmpIdx2 - the indices of remaining 25% child measures selected
        %                 Analysis.Idxrp - Indices after randomly change between the order of the population
        %                 Analysis.measurePop - measure population
        %                 Analysis.fitnessPop - fitness population
        %                 Analysis.measureiter - best measure for each iteration
        %                 Analysis.ratioa - min(1, exp(sum(fitnessPop)-sum(fitnessPopPrev)))
        %                 Analysis.ratioachild - Percentage of children measures are kepted
        %                 Analysis.ratiomVal - the maximum fitness value
        %                 Analysis.ratio - exp(sum(fitnessPop)-sum(fitnessPopPrev))
        %                 Analysis.JumpType - small-scale mutation or large-scale mutation
        %                 Analysis.ElemUpdated - the element updated in the small-scale mutation in every iteration
        %                 Analysis.subsetIntervalnPop - the intervals for each measure element for each measures in the population
        %
        ===============================================================================
        """
    
        #######################################################################
        ########################### Set Up Variables ##########################
        #######################################################################
        Analysis = dict() ## Dictionary storing updates of evolution
        nSources = self.N ## number of sources
        eta = Parameters.eta ## Rate of small-scale mutation in [0,1]
        p = self.p ## Hyper-parameters on generalized-mean objective
        nElements = self.nElements ## Number of elements in a measure, includining mu_all=1
        nPop = Parameters.nPop ## Number of measures in the population
        measureNumEach = self.measureNumEach
        measureEach = self.measureEach
        index_keys = self.index_keys
        sampleVar = Parameters.sampleVar
        
        #######################################################################
        ################## Compute Initial Measure Width Bounds ###############
        #######################################################################
        
        ## Get measure indices for lower and upper bounds on each measure element
        lowerindex, upperindex = self.compute_bounds()

        #######################################################################
        ###################### Initialize Measure Elements ####################
        #######################################################################
        
        ## Initialize measure population
        measurePop = np.zeros((nPop,nElements))
        
        ## Initialize fitness
        fitnessPop = -1e100*np.ones(nPop) ## Very small number, bad fitness
        
        ## Initialize measure population with input measures
        if (trueInitMeasure is not None): 
            for i in range(nPop):
                measurePop[i,:] = trueInitMeasure
                fitnessPop[i] = self.evalFitness_softmax(Bags, Labels, measurePop[i,:])
            
        else:  ## Initialize measure population randomly
            for i in range(nPop):
                measurePop[i,:] = self.sampleMeasure(nSources,lowerindex,upperindex);
                fitnessPop[i] = self.evalFitness_softmax(Bags, Labels, measurePop[i,:])
        
        #######################################################################
        #################### Iterate through Optimization #####################
        #######################################################################
        
        indx = np.argmax(fitnessPop)
        mVal = fitnessPop[indx]
        measure = measurePop[indx,:]  
        initialMeasure = measure
        mVal_before = -10000
        sampleVar = Parameters.sampleVar
        
        for t in range(Parameters.maxIterations):
            childMeasure = deepcopy(measurePop)
            childFitness = np.zeros(nPop) ## Pre-allocate array
            subsetIntervalnPop = np.zeros((nPop,nElements-1)) ## Dont need to sample mu_all=1 (last element)
            ElemIdxUpdated = np.zeros(nPop)
            JumpType = np.zeros(nPop)
            
            ###################################################################
            ######################## Sample Population ########################
            ###################################################################
            
            if (Parameters.use_parallel == True):
                num_cores = mp.cpu_count()
#                Parallel(n_jobs=num_cores)(delayed(self.sample_population)(Bags, Labels, childMeasure[i,:], lowerindex, upperindex, Parameters) for i in range(nPop)) 
                pool = mp.Pool(num_cores)
                res = [pool.apply_async(func=self.sample_population, args=(Bags, Labels, childMeasure[i,:], lowerindex, upperindex, Parameters)) for i in range(nPop)]

                for i in range(nPop):
                    childMeasure[i,:], childFitness[i], JumpType[i] = res[i].get(timeout=1)[0], res[i].get(timeout=1)[1], res[i].get(timeout=1)[2]

            else:
                for i in range(nPop):
                    childMeasure[i,:], childFitness[i], JumpType[i] = self.sample_population(Bags, Labels, childMeasure[i,:], lowerindex, upperindex, Parameters)
    
            ###################################################################
            ###################### Evolutionary Algorithm #####################
            ###################################################################
            """
            ===================================================================
            % Example: Say population size is P. Both P parent and P child are 
            % pooled together (size 2P) and sort by fitness, then, P/2 measures
            % with top 25% of the fitness are kept; The remaining P/2 child 
            % comes from the remaining 75% of the pool by sampling from a 
            % multinomial distribution).
            ===================================================================
            """
            
            """
            Create parent + child measure pool (2P)
            """
            fitnessPopPrev = deepcopy(fitnessPop)
            ParentChildMeasure = np.concatenate((measurePop,childMeasure),axis=0) ## total 2xNPop measures
            ParentChildFitness = np.concatenate((fitnessPop,childFitness))
            
            ## Sort fitness values in descending order
            ParentChildFitness_sortIdx = (-ParentChildFitness).argsort()
            ParentChildFitness_sortV = ParentChildFitness[ParentChildFitness_sortIdx]
            
            """
            Keep top 25% (P/2) from parent and child populations
            """
            measurePopNext = np.zeros((nPop,ParentChildMeasure.shape[1]))
            fitnessPopNext = np.zeros(nPop)
            ParentChildTmpIdx1 = ParentChildFitness_sortIdx[0:int(nPop/2)]
            measurePopNext[0:int(nPop/2),:] = deepcopy(ParentChildMeasure[ParentChildTmpIdx1,:])
            fitnessPopNext[0:int(nPop/2)] = deepcopy(ParentChildFitness[ParentChildTmpIdx1])
            
            """
            For the remaining (P/2) measures, sample according to multinomial 
            from remaining 75% of parent/child pool
            """
            ## Method 1: Random sample - Correct but not used in the paper
            # ParentChildTmpIdx2 = randsample(ParentChildFitness_sortIdx(Parameters.nPop/2+1:end),Parameters.nPop/2);
            
            ## Method 2: Sample by multinomial distribution based on fitness
            PDFdistr75 = deepcopy(ParentChildFitness_sortV[int(nPop/2)::])
            outputIndexccc = self.sampleMultinomial_mat(PDFdistr75, int(nPop/2), 'descend')

            ParentChildTmpIdx2 = deepcopy(ParentChildFitness_sortIdx[int(nPop/2)+outputIndexccc])
            measurePopNext[int(nPop/2):nPop,:] = deepcopy(ParentChildMeasure[ParentChildTmpIdx2,:])
            fitnessPopNext[int(nPop/2):nPop] = deepcopy(ParentChildFitness[ParentChildTmpIdx2])
            
            Idxrp = np.random.permutation(nPop) ## randomly change the order of the population
            measurePop = deepcopy(measurePopNext[Idxrp,:])
            fitnessPop = deepcopy(fitnessPopNext[Idxrp])
            
            a = np.minimum(1, np.exp(np.sum(fitnessPop)-np.sum(fitnessPopPrev))) ## fitness change ratio
            ParentChildTmpIdxall = np.concatenate((ParentChildTmpIdx1, ParentChildTmpIdx2))
            achild = np.sum(ParentChildTmpIdxall > nPop)/nPop ## Percentage of children were kept for the next iteration
            
            ## Update best answer - printed to the terminal
            if(np.max(fitnessPop) > mVal):
                mVal_before = deepcopy(mVal)
                mIdx = (-fitnessPop).argsort()[0]
                mVal = deepcopy(fitnessPop[mIdx])
                measure = deepcopy(measurePop[mIdx,:])
#                print(f'{mVal}')
#                print(measure)
            
            ###################################################################
            ##################### Update Analysis Tracker #####################
            ###################################################################
            if (Parameters.analysis == True): ## record all intermediate process (be aware of the possible memory limit!)
                if not(t):
                    Analysis['ParentChildMeasure'] = np.zeros((ParentChildMeasure.shape[0],ParentChildMeasure.shape[1],Parameters.maxIterations))
                    Analysis['ParentChildFitness'] = np.zeros((Parameters.maxIterations,len(ParentChildFitness)))
                    Analysis['ParentChildTmpIdx2'] = np.zeros((Parameters.maxIterations,len(ParentChildTmpIdx2)))
                    Analysis['Idxrp'] = np.zeros((Parameters.maxIterations,len(Idxrp)))
                    Analysis['measurePop'] = np.zeros((measurePop.shape[0],measurePop.shape[1],Parameters.maxIterations))
                    Analysis['fitnessPop'] = np.zeros((Parameters.maxIterations,len(fitnessPop)))
                    Analysis['measureiter'] = np.zeros((Parameters.maxIterations,len(measure)))
                    Analysis['ratioa'] = np.zeros((Parameters.maxIterations))
                    Analysis['ratioachild'] = np.zeros((Parameters.maxIterations))
                    Analysis['ratiomVal'] = np.zeros((Parameters.maxIterations))
                    Analysis['ratio'] = np.zeros((Parameters.maxIterations))
                    Analysis['JumpType'] = np.zeros((Parameters.maxIterations,len(JumpType))) 
#                    Analysis['ElemIdxUpdated'] = np.zeros((Parameters.maxIterations,len(ElemIdxUpdated))) 
#                    Analysis['subsetIntervalnPop'] = np.zeros((subsetIntervalnPop.shape[0],subsetIntervalnPop.shape[1],Parameters.maxIterations)) 
                else:
                    Analysis['ParentChildMeasure'][:,:,t] = deepcopy(ParentChildMeasure)
                    Analysis['ParentChildFitness'][t,:] = deepcopy(ParentChildFitness)
                    Analysis['ParentChildTmpIdx2'][t,:] = deepcopy(ParentChildTmpIdx2)
                    Analysis['Idxrp'][t,:] = deepcopy(Idxrp)
                    Analysis['measurePop'][:,:,t] = deepcopy(measurePop)
                    Analysis['fitnessPop'][t,:] = deepcopy(fitnessPop)
                    Analysis['measureiter'][t,:] = deepcopy(measure)
                    Analysis['ratioa'][t] = deepcopy(a)
                    Analysis['ratioachild'][t] = deepcopy(achild)
                    Analysis['ratiomVal'][t] = deepcopy(mVal)
                    Analysis['ratio'][t] = np.exp(np.sum(fitnessPop)-np.sum(fitnessPopPrev))
                    Analysis['JumpType'][t,:] = deepcopy(JumpType)
#                    Analysis['ElemIdxUpdated'][t,:] = ElemIdxUpdated
#                    Analysis['subsetIntervalnPop'][:,:,t] = subsetIntervalnPop  
            
            ## Update terminal 
            if(not(t % 10)):
                print('\n')
                print(f'Iteration: {str(t)}')
                print(f'Best fitness: {mVal.round(6)}')
                print(measure.round(4))
                
            del fitnessPopNext
            del measurePopNext
            del PDFdistr75
            del fitnessPopPrev
            del ParentChildMeasure
            del ParentChildFitness
            del childMeasure
            del childFitness
    
            ## Stop if we've found a measure meeting our desired level of fitness
            if (np.abs(mVal - mVal_before) <= Parameters.fitnessThresh):
                break
    
        return measure, initialMeasure, Analysis
    
    def evalFitness_softmax(self, Bags, Labels, measure):
    
        """
        ===========================================================================
        % Evaluate the fitness a measure, similar to evalFitness_minmax() for
        % classification but uses generalized mean (sometimes also named "softmax") model
        %
        % INPUT
        %    Labels         - 1xNumTrainBags double  - Training labels for each bag
        %    measure        - measure to be evaluated after update
        %    nPntsBags      - 1xNumTrainBags double    - number of points in each bag
        %    bag_row_ids    - the row indices of measure used for each bag
        %    diffM          - Precompute differences for each bag
        %
        % OUTPUT
        %   fitness         - the fitness value using min(sum(min((ci-d)^2))) for regression.
        %
        % Written by: X. Du 03/2018
        %
        ===========================================================================
        """
        
        p1 = self.p[0]
        p2 = self.p[1]
        
        fitness = 0
       
        ## Compute CI for non-singleton bags
        for b_idx in range(self.B):
            ci = self.compute_chi(Bags[b_idx], 1, measure)
            if(Labels[b_idx] == 0):  ## Negative bag, label = 0
                fitness = fitness - (np.mean(ci**(2*p1))**(1/p1))
            else: ## Positive bag, label=1
                fitness = fitness - (np.mean((ci-1)**(2*p2))**(1/p2))
    
            ## Sanity check
            if (np.isinf(fitness) or not(np.isreal(fitness)) or np.isnan(fitness)):
                fitness = np.real(fitness)
                fitness = -10000000
                
        return fitness
    
    def sample_population(self, Bags, Labels, childMeasure, lowerindex, upperindex, Parameters):
        """
        ===========================================================================
        % Evaluate the fitness a measure, similar to evalFitness_minmax() for
        % classification but uses generalized mean (sometimes also named "softmax") model
        %
        % INPUT
        %    Labels         - 1xNumTrainBags double  - Training labels for each bag
        %    measure        - measure to be evaluated after update
        %    nPntsBags      - 1xNumTrainBags double    - number of points in each bag
        %    bag_row_ids    - the row indices of measure used for each bag
        %    diffM          - Precompute differences for each bag
        %
        % OUTPUT
        %   fitness         - the fitness value using min(sum(min((ci-d)^2))) for regression.
        %
        % Written by: X. Du 03/2018
        %
        ===========================================================================
        """
        
        subsetInterval = self.evalInterval(childMeasure, lowerindex, upperindex)
        
        ## Sample new value(s)
        z = np.random.uniform(low=0,high=1)
        if(z < Parameters.eta): ## Small-scale mutation: update only one element of the measure
            iinterv = self.sampleMultinomial_mat(subsetInterval, 1, 'descend' ) ## Update the one interval according to multinomial
            childMeasure = self.sampleMeasure(self.N, lowerindex, upperindex, childMeasure, iinterv, Parameters.sampleVar)
            childFitness = self.evalFitness_softmax(Bags, Labels, childMeasure)
            JumpType = 1

        else: ## Large-scale mutation: update all measure elements sort by valid interval widths in descending order
            indx_subsetInterval = (-subsetInterval).argsort() ## sort the intervals in descending order
      
            for iinterv in range(len(indx_subsetInterval)): ## for all elements
                childMeasure =  self.sampleMeasure(self.N, lowerindex, upperindex, childMeasure, iinterv, Parameters.sampleVar)
                childFitness = self.evalFitness_softmax(Bags, Labels, childMeasure)
                JumpType = 2
        
        return childMeasure, childFitness, JumpType
    
    def sampleMeasure(self, *args):
        """
        ===========================================================================
        %sampleMeasure - sampling a new measure
        % If the number of inputs are two, sampling a brand new measure (only used during initialization)
        % If the number of inputs are all five, sampling a new value for only one element of the measure
        % This code samples a new measure value from truncated Gaussian distribution.
        %
        % INPUT
        %   nSources - number of sources
        %   lowerindex - the cell that stores all the corresponding subsets (lower index) of measure elements
        %   upperindex - the cell that stores all the corresponding supersets (upper index) of measure elements
        %   prevSample (optional)- previous measure, before update
        %   IndexsubsetToChange (optional)- the measure element index to be updated
        %   sampleVar (optional) - sampleVar set in Parameters
        % OUTPUT
        %   - measure - new measure after update
        %
        ===========================================================================
        """
        
        ## Extract function arguments
        nSources = args[0]
        lowerindex = args[1]
        upperindex = args[2]
    
        if (len(args) > 4):
            prevSample = args[3]
            IndexsubsetToChange = args[4]
            sampleVar = args[5]
    
        if(len(args) < 4):
            """
            ## Sample a brand new measure
            ## Flip a coin to decide whether to sample from above or sample from bottom
            ## Sample new measure, prior is uniform/no prior
            """
            z = np.random.uniform(low=0,high=1)
            if (z >= 0.5): ## sample from bottom-up
                measure = self.sampleMeasure_Bottom(nSources,lowerindex)
            else:   ## sample from top-down
                measure = self.sampleMeasure_Above(nSources,upperindex)
        else:
            ## Sample just one new element of measure
            
            nElements = self.nElements
            measure = prevSample
            if ((IndexsubsetToChange<=nSources-1) and (IndexsubsetToChange>=0)): ## singleton
                lowerBound = 0
                upperBound = np.amin(measure[upperindex[IndexsubsetToChange]])
            elif ((IndexsubsetToChange>=(nElements-nSources-1)) and (IndexsubsetToChange<=(nElements-2))): ## (nSources-1)-tuple
                lowerBound = np.amax(measure[lowerindex[IndexsubsetToChange]])
                upperBound = 1
            else:  ## remaining elements
                lowerBound = np.amax(measure[lowerindex[IndexsubsetToChange]]) 
                upperBound = np.amin(measure[upperindex[IndexsubsetToChange]]) 
            
            denom = upperBound - lowerBound
            v_bar = sampleVar/((denom**2)+1e-5) ## -changed
            x_bar = measure[IndexsubsetToChange] ## -changed
            
            ## Sample from a Truncated Gaussian
            sigma_bar = np.sqrt(v_bar)
            c2 = np.random.uniform(low=0,high=1) ## Randomly generate a value between [0,1]
            val = self.invcdf_TruncatedGaussian(c2,x_bar,sigma_bar,lowerBound,upperBound) ## New measure element value
            measure[IndexsubsetToChange] = val  ## New measure element value
    
        return measure
    
    def sampleMeasure_Above(self, nSources,upperindex):
        """
        =======================================================================
        %sampleMeasure_Above - sampling a new measure from "top-down"
        % The order of sampling a brand new measure is to first sample (nSource-1)-tuples (e.g. g_1234 for 5-source),
        % then (nSource-2)-tuples (e.g. g_123), and so on until singletons (g_1)
        % Notice it will satisfy monotonicity!
        %
        % INPUT
        %   nSources - number of sources
        %   upperindex - the cell that stores all the corresponding supersets (upper index) of measure elements
        %
        % OUTPUT
        %   - measure - new measure after update
        %
        =======================================================================
        """
        
        ## Sample new measure, prior is uniform/no prior
        nElements = self.nElements  ## total length of measure
        nSources = self.N ## Number of sources
        measure = np.zeros(nElements)
        measure[-1] = 1 ## mu_all
        measure[nElements-2:nElements-nSources-2:-1] = np.random.uniform(low=0,high=1,size=nSources) ## mu_1234,1235,1245,1345,2345 for 5 sources, for example (second to last tier)
        
        for j in range(nElements-nSources-2,-1,-1):
            upperBound = np.amin(measure[upperindex[j]]) ## upper bound
            measure[j] = 0 + (upperBound - 0)*np.random.uniform(low=0,high=1)
            
        return measure
    
    def sampleMeasure_Bottom(self,nSources,lowerindex):
        """
        =======================================================================
        %sampleMeasure_Bottom - sampling a new measure from "bottom-up"
        % The order of sampling a brand new measure is to first sample singleton (e.g. g_1..),
        % then duplets (e.g. g_12..), then triples (e.g. g_123..), etc..
        % Notice it will satisfy monotonicity!
        %
        % INPUT
        %   nSources - number of sources
        %   lowerindex - the cell that stores all the corresponding subsets (lower index) of measure elements
        %
        % OUTPUT
        %   - measure - new measure after update
        %
        =======================================================================
        """
        
        nElements = self.nElements  ## total length of measure
        nSources = self.N ## Number of sources
        measure = np.zeros(nElements)
        measure[0:nSources] = np.random.uniform(low=0,high=1,size=nSources) ## sample singleton densities
        measure[-1] = 1 ## mu_all
        
        for j in range(nSources,(len(measure)-1)):
            lowerBound = np.amax(measure[lowerindex[j]]) ## lower bound     
            measure[j] = lowerBound + (1-lowerBound)*np.random.uniform(low=0,high=1)
        
        return measure
    
    
    def evalInterval(self, measure, lowerindex, upperindex):
        """
        =======================================================================
        % Evaluate the valid interval width of a measure, then sort in descending order.
        %
        % INPUT
        %   measure -  measure to be evaluated after update
        %   lowerindex - the cell that stores all the corresponding subsets (lower index) of measure elements
        %   upperindex - the cell that stores all the corresponding supersets (upper index) of measure elements
        % OUTPUT
        %   subsetInterval - the valid interval widths for the measure elements,before sorting by value
        %
        =======================================================================
        """
        
        nElements = self.nElements 
        nSources = self.N
        lb = np.zeros(nElements-1)
        ub = np.zeros(nElements-1)
        
        for j in range(nSources):  ## singleton
            lb[j] = 0 ## lower bound
            ub[j] = np.amin(measure[upperindex[j]]) ## upper bound
    
        for j in range(nSources, nElements-nSources-1):
            lb[j] = np.amax(measure[lowerindex[j]]) ## lower bound
            ub[j] = np.amin(measure[upperindex[j]]) ## upper bound
        
        for j in range(nElements-nSources-1,nElements-1):
            lb[j] = np.max(measure[lowerindex[j]]) ## lower bound
            ub[j] = 1 ## upper bound
            
        subsetInterval = ub - lb   
        
        return subsetInterval
    
    def compute_bounds(self):
        
        """ 
        ===============================================================================
        %% compute_bounds()
        % This function learns a fuzzy measure for Multiple Instance Choquet Integral (MICI)
        % This function works with classifier fusion
        % This function uses a generalized-mean objective function and an evolutionary algorithm to optimize.
        %
        % INPUT
        %    Parameters                 - 1x1 struct - The parameters set by learnCIMeasureParams() function.
        %
        % OUTPUT
        %    measure             - 1x(2^nSource-1) double - learned measure by MICI
        %
        ===============================================================================
        """
        
        #######################################################################
        ########################### Set Up Variables ##########################
        #######################################################################
        nElements = self.nElements ## Number of elements in a measure, includining mu_all=1
        measureNumEach = self.measureNumEach
        measureEach = self.measureEach
        index_keys = self.index_keys
        nSources = self.N
        
        #######################################################################
        ################## Compute Initial Measure Width Bounds ###############
        #######################################################################
        ## Compute lower bound index (Lower bound is largest value of it's subsets)
        nElem_prev = 0
        nElem_prev_prev = 0
        
        lowerindex = [] ## Indices of elements used for lower bounds of each element
        
        ## Populate bounds singletons (lower-bounds are meaningless)
        for i in range(nSources):
            lowerindex.append([-1])
        
        for i in range(1,nSources-1): ## Lowest bound is 0, highest bound is 1
            nElem = measureNumEach[i] 
            elem = measureEach[i] 
            nElem_prev = nElem_prev + measureNumEach[i-1]
            if (i==1):
                nElem_prev_prev = 0
            elif (i>1):
                nElem_prev_prev  = nElem_prev_prev + measureNumEach[i-1]
     
            for j in range(nElem):
#                lowerindex[nElem_prev+j] = -1
                children = np.array(list(itertools.combinations(elem[j,:], i)))
                
                for idk, child in enumerate(children):
                    
                    if not(idk):
                        tmp_ind = [index_keys[str(child)]]
                    else:
                        tmp_ind = tmp_ind + [index_keys[str(child)]]
            
                lowerindex.append(tmp_ind) 
                
        lowerindex.append([-1])

        ## Compute upper bound index (Upper bound is smallest value of it's supersets)
        upperindex = [] ## Indices of elements used for upper bounds of each element
        tmp_ind = []
        
        for i in range(0,nSources-1): ## Loop through tiers
            nElem = measureNumEach[i] 
            elem = measureEach[i]
            elem_next_ind = i+1
            elem_next = measureEach[elem_next_ind]
            
            for j, child in enumerate(elem): ## work from last element to first
                tmp_ind = []
                
                if(not(i==(nSources-2))):
                
                    for idk, parent in enumerate(elem_next):
                        
                        ## Test if child is a subest of parent
                        mask = np.in1d(child,parent)
                        
                        if(i == 0): ## If singleton level
                            if (np.sum(mask) == 1):
                                
                                if not(len(tmp_ind)):
                                    tmp_ind = [index_keys[str(parent)]]
                                else:
                                    tmp_ind = tmp_ind + [index_keys[str(parent)]]
                        
                        elif(np.sum(mask) == len(child)):  ## If level above singletons
                            
                            if not(len(tmp_ind)):
                                tmp_ind = [index_keys[str(parent)]]
                            else:
                                tmp_ind = tmp_ind + [index_keys[str(parent)]]
                    
                    upperindex.append(tmp_ind) 
                
                else:
                    upperindex.append([nElements])
                
        upperindex.append([-1])
        
        return lowerindex, upperindex

    def invcdf_TruncatedGaussian(self,cdf,x_bar,sigma_bar,lowerBound,upperBound):
        """
        =======================================================================
        %
        % stats_TruncatedGaussian - stats for a truncated gaussian distribution
        %
        % INPUT
        %   - cdf: evaluated at the values at cdf
        %   - x_bar,sigma_bar,lowerBound,upperBound: suppose X~N(mu,sigma^2) has a normal distribution and lies within
        %   the interval lowerBound<X<upperBound
        %   *The size of cdfTG and pdfTG is the common size of X, MU and SIGMA.  A scalar input
        %   functions as a constant matrix of the same size as the other inputs.
        %
        % OUTPUT
        %   - val: the x corresponding to the cdf TG value
        %
        =======================================================================
        """
    
        term2 = (self.normcdf(upperBound,x_bar,sigma_bar) - self.normcdf(lowerBound,x_bar,sigma_bar))
        const2 =  cdf*term2 + self.normcdf(lowerBound,x_bar,sigma_bar)
        
        const3 = (const2*2)-1
        inner_temp = erfinv(const3)
        val = inner_temp*np.sqrt(2)*sigma_bar + x_bar
        
        return val
        
        
    def normcdf(self,x,mu,sigma):
        """
        =======================================================================
        % INPUT
        %   - x: the x to compute the cdf value
        %   - mu: mean of the Gaussian distribution
        %   - sigma: sigma of the Gaussian distribution
        % OUTPUT
        %   - val: the x corresponding to the cdf value
        % based on MATLAB normcdf() function.
        =======================================================================
        """
        
        z = np.divide((x-mu),sigma)
        p = 0.5 * erfc(np.divide(-z,np.sqrt(2)))
        
        return p


    def sampleMultinomial_mat(self, PDFdistr, NumSamples, MethodStr):
        """
        =======================================================================
        % Generate random samples from a multinomial. Return index of the sample.
        %
        % INPUT
        % PDFdistr - distribution, can be fitness or interval or any distribution
        %            input. Must be 1x#of distrbution values
        % NumSamples - Number of samples
        % MethodStr - ='descend', sort by descending order, the larger the distribution
        % the more likely it is to be sampled. ='ascend', sort by ascending order,
        % the smaller the distribution value the more likely it is to be sampled
        % (to be used in picking samples). Case insensitive!
        % OUTPUT
        % Indxccc - index of the samples
        %
        =======================================================================
        """
        
#        ## Method 1: correct! Vector form. Uses a for-loop. Slow.
#        Indxccc = []
#        PDFdistr_sortIdx = (-PDFdistr).argsort() ## Sort descending order
#        PDFdistr_resort = PDFdistr[PDFdistr_sortIdx]
#        PDFdistr_resort_norm = np.divide(PDFdistr_resort, np.sum(PDFdistr_resort))
#        PDFdistr_resort_normCDF = np.cumsum(PDFdistr_resort_norm)
#        ccc = np.random.uniform(low=0,high=1,size=NumSamples)
#        for pciter in range(NumSamples):
#            Indxccc.append(np.where(ccc[pciter]<=PDFdistr_resort_normCDF)) ## draw from Multinomial CDF, the first index/element that is larger than ccc =rand(1)
#
#        outputIndex = PDFdistr_sortIdx[Indxccc]
        
        ## Method 2: correct! Matrix form. Faster
        if (MethodStr == 'descend'):
            if (np.sum(PDFdistr<0)==0): ## All positive distribution values
                PDFdistr_sortIdx = (-PDFdistr).argsort()
                PDFdistr_resort = PDFdistr[PDFdistr_sortIdx]
            elif (np.sum(PDFdistr<0)==len(PDFdistr)): ## all negative
                PDFdistr = np.divide(-1,PDFdistr)
                PDFdistr_sortIdx = (-PDFdistr).argsort()
                PDFdistr_resort = PDFdistr[PDFdistr_sortIdx]
            else:
                PDFdistr = PDFdistr-np.amax(PDFdistr)-(1e-5);
                PDFdistr = np.divide(-1,PDFdistr)
                PDFdistr_sortIdx = (-PDFdistr).argsort()
                PDFdistr_resort = PDFdistr[PDFdistr_sortIdx]
                
        elif (MethodStr == 'ascend'):
            if  (np.sum(PDFdistr<0)==0): ## all positive distribution values
                PDFdistr = np.divide(-1,PDFdistr)
                PDFdistr_sortIdx = (-PDFdistr).argsort()
                PDFdistr_sortIdx = PDFdistr_sortIdx[::-1]
                PDFdistr_resort = PDFdistr[PDFdistr_sortIdx]
            elif (np.sum(PDFdistr<0)==len(PDFdistr)): ## all negative     
                PDFdistr_sortIdx = (-PDFdistr).argsort()
                PDFdistr_sortIdx = PDFdistr_sortIdx[::-1]
                PDFdistr_resort = PDFdistr[PDFdistr_sortIdx]
            else:
                PDFdistr = PDFdistr-np.amax(PDFdistr)-(1e-5)
                PDFdistr_sortIdx = (-PDFdistr).argsort()
                PDFdistr_sortIdx = PDFdistr_sortIdx[::-1]
                PDFdistr_resort = PDFdistr[PDFdistr_sortIdx]
        
        PDFdistr_resort_norm = np.divide(PDFdistr_resort, np.sum(PDFdistr_resort))
        PDFdistr_resort_normCDF = np.cumsum(PDFdistr_resort_norm)
        ccc = np.random.uniform(low=0,high=1,size=NumSamples)
        cccrepmat = np.tile(np.expand_dims(ccc,axis=1), (1,len(PDFdistr)))
        PDFdistr_resort_normCDFrepmat = np.tile(PDFdistr_resort_normCDF,(NumSamples,1))
        diffrepmat = cccrepmat - PDFdistr_resort_normCDFrepmat
        
        if (NumSamples == 1):
            Indxccc = np.where(diffrepmat<0)[1][0] 
        else:
            Indxccc = np.zeros(NumSamples)
            for j in range(NumSamples):
                Indxccc[j] = np.where(diffrepmat[j,:]<0)[0][0]
                
            Indxccc = Indxccc.astype('int64')
        
        outputIndex =  PDFdistr_sortIdx[Indxccc]
        
        return outputIndex

    def get_keys_index(self):
        """
        Sets up a dictionary for referencing FM.

        :return: The keys to the dictionary
        """

        vls = np.arange(1, self.N + 1)
        count = 0
        Lattice = {}
        for i in range(0, self.N):
            Lattice[str(np.array([vls[i]]))] = count
            count = count + 1
        for i in range(2, self.N + 1):
            A = np.array(list(itertools.combinations(vls, i)))
            for latt_pt in A:
                Lattice[str(latt_pt)] = count
                count = count + 1
        return Lattice
    
    
    def chi_by_sort(self, x2):
        """
        This 

        :param 
        :return: 
        """
        
        n = len(x2)
        pi_i = np.argsort(x2)[::-1][:n] + 1
        ch = x2[pi_i[0] - 1] * (self.fm[str(pi_i[:1])])
        for i in range(1, n):
            latt_pti = np.sort(pi_i[:i + 1])
            latt_ptimin1 = np.sort(pi_i[:i])
            ch = ch + x2[pi_i[i] - 1] * (self.fm[str(latt_pti)] - self.fm[str(latt_ptimin1)])
        
        sort_idx = [idx for idx, val in enumerate(self.all_sorts) if (sum(pi_i == val ) == n)]
        
        return ch, int(sort_idx[0]+1)
            
    def get_linear_eqts(self):
        """
        This 

        :param 
        :return: 
        """
        
        eqt_dict = dict()
        
        for idx, pi_i in enumerate(self.all_sorts):
            
            line = 'Sort: ' + str(pi_i) + ', Eq: ChI = ' 
            
            line = line + str(round(self.fm[str(pi_i[:1])],4)) + '[' + str(pi_i[0]) + ']'
            
            for i in range(1,len(pi_i)):
                
                latt_pti = np.sort(pi_i[:i + 1])
                latt_ptimin1 = np.sort(pi_i[:i])
                
                line = line + ' + ' + str(abs(round(self.fm[str(latt_pti)] - self.fm[str(latt_ptimin1)],4))) + '[' + str(pi_i[i]) + ']'
            
        
            eqt_dict[str(pi_i)] = line
        
        return eqt_dict
        
    
    