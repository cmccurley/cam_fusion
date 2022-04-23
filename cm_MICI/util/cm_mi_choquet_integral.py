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
        self.type='noisy-or'
        self.trainSamples, self.trainLabels = [], []
        self.testSamples, self.testLabels = [], []
        self.N, self.numberConstraints, self.M = 0, 0, 0
        self.g = 0
        self.fm = []
        self.type = []


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
        fm_len = 2 ** self.N - 1  # nc
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

        
        ## Estimate measure parameters
        print("Number Sources : ", self.N, "; Number Training Bags : ", self.B)
        self.fm = self.produce_lattice_softmax(Bags, Labels, Parameters)

        return self

    def compute_chi(self, Bag):
        """
        ===========================================================================
        This will produce an output of the ChI

        This will use the learned(or specified) Choquet integral to
        produce an output w.r.t. to the new input.

        :param Bag: testing bag of size (n_samples X n_sources)
        :return: output of the choquet integral.
        ===========================================================================
        """
        ch = np.zeros(Bag.shape[0])

        for inst_idx in range(Bag.shape[0]):
            x = Bag[inst_idx,:]
            n = len(x)
            pi_i = np.argsort(x)[::-1][:n] + 1
            ch[inst_idx] = x[pi_i[0] - 1] * (self.fm[str(pi_i[:1])])
            for i in range(1, n):
                latt_pti = np.sort(pi_i[:i + 1])
                latt_ptimin1 = np.sort(pi_i[:i])
                ch[inst_idx] = ch[inst_idx] + x[pi_i[i] - 1] * (self.fm[str(latt_pti)] - self.fm[str(latt_ptimin1)])
        return ch

#    def produce_lattice_softmax(self, Bags, Labels):
#        """
#        ===========================================================================
#        This method builds is where the lattice(or FM variables) will be learned.
#
#        The FM values can be found via a quadratic program, which is used here
#        after setting up constraint matrices. Refer to papers for complete overview.
#
#        :return: Lattice, the learned FM variables.
#        ===========================================================================
#        """
#
#        fm_len = 2 ** self.N - 1  # nc
#        E = np.zeros((fm_len, fm_len))  # D
#        L = np.zeros(fm_len)  # f
#        index_keys = self.get_keys_index()
#        for i in range(0, self.M):  # it's going through one sample at a time.
#            l = self.trainLabels[i]  # this is the labels
#            fm_coeff = self.get_fm_class_img_coeff(index_keys, self.trainSamples[:, i], fm_len)  # this is Hdiff
#            # print(fm_coeff)
#            L = L + (-2) * l * fm_coeff
#            E = E + np.matmul(fm_coeff.reshape((fm_len, 1)), fm_coeff.reshape((1, fm_len)))
#
#        solvers.options['show_progress'] = False
#        G, h, A, b = self.build_constraint_matrices(index_keys, fm_len)
#        sol = solvers.qp(matrix(2 * E, tc='d'), matrix(L.T, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'),
#                         matrix(A, tc='d'), matrix(b, tc='d'))
#
#        g = sol['x']
#        Lattice = {}
#        for key in index_keys.keys():
#            Lattice[key] = g[index_keys[key]]
#        return Lattice
    
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
        
        for t in range(Parameters.maxIterations):
            childMeasure = measurePop
            childFitness = np.zeros(nPop) ## Pre-allocate array
            subsetIntervalnPop = np.zeros((nPop,nElements-1)) ## Dont need to sample mu_all=1 (last element)
            ElemIdxUpdated = np.zeros(nPop)
            JumpType = np.zeros(nPop)
            for i in range(nPop):
                subsetInterval = self.evalInterval(measurePop[i,:], lowerindex, upperindex)
                subsetIntervalnPop[i,:] = subsetInterval
                
                ## Sample new value(s)
                z = np.random.uniform(low=0,high=1)
                if(z < eta): ## Small-scale mutation: update only one element of the measure
                    iinterv = self.sampleMultinomial_mat(subsetInterval, 1, 'descend' ) ## Update the one interval according to multinomial
                    childMeasure[i,:] = self.sampleMeasure(nSources, lowerindex, upperindex, childMeasure[i,:],iinterv, sampleVar)
                    childFitness[i] = self.evalFitness_softmax(Bags, Labels, measurePop[i,:])
                    JumpType[i] = 1
                    ElemIdxUpdated[i] = iinterv
                else: ## Large-scale mutation: update all measure elements sort by valid interval widths in descending order
                    indx_subsetInterval = (-subsetInterval).argsort(axis=1) ## sort the intervals in descending order
      
                    for iinterv in range(indx_subsetInterval.shape[1]): ## for all elements
                        childMeasure[i,:] =  self.sampleMeasure(nSources, lowerindex, upperindex, childMeasure[i,:],iinterv, sampleVar)
                        childFitness[i] = self.evalFitness_softmax(Bags, Labels, measurePop[i,:])
                        JumpType[i] = 2
    
#            ###################################################################
#            ###################### Evolutionary Algorithm #####################
#            ###################################################################
#            """
#            ===================================================================
#            % Example: Say population size is P. Both P parent and P child are 
#            % pooled together (size 2P) and sort by fitness, then, P/2 measures
#            % with top 25% of the fitness are kept; The remaining P/2 child 
#            % comes from the remaining 75% of the pool by sampling from a 
#            % multinomial distribution).
#            ===================================================================
#            """
#            
#            """
#            Create parent + child measure pool (2P)
#            """
#            fitnessPopPrev = fitnessPop
#            ParentChildMeasure = np.concatenate((measurePop,childMeasure),axis=0) ## total 2xNPop measures
#            ParentChildFitness = np.concatenate((fitnessPop,childFitness))
#            
#            ## Sort fitness values in descending order
#            ParentChildFitness_sortIdx = (-ParentChildFitness).argsort()
#            ParentChildFitness_sortV = ParentChildFitness[ParentChildFitness_sortIdx]
#            
#            """
#            Keep top 25% (P/2) from parent and child populations
#            """
#            measurePopNext = np.zeros((nPop,ParentChildMeasure.shape[1]))
#            fitnessPopNext = np.zeros(nPop)
#            ParentChildTmpIdx1 = ParentChildFitness_sortIdx[0:nPop/2] 
#            measurePopNext[0:nPop/2,:] = ParentChildMeasure[ParentChildTmpIdx1,:]
#            fitnessPopNext[0:nPop/2] = ParentChildFitness[ParentChildTmpIdx1]
#            
#            """
#            For the remaining (P/2) measures, sample according to multinomial 
#            from remaining 75% of parent/child pool
#            """
#            ## Method 1: Random sample - Correct but not used in the paper
#            # ParentChildTmpIdx2 = randsample(ParentChildFitness_sortIdx(Parameters.nPop/2+1:end),Parameters.nPop/2);
#            
#            ## Method 2: Sample by multinomial distribution based on fitness
#            PDFdistr75 = ParentChildFitness_sortV[nPop/2::]
#            outputIndexccc = self.sampleMultinomial_mat(PDFdistr75, nPop/2, 'descend')
#            
#            ParentChildTmpIdx2 = ParentChildFitness_sortIdx[nPop/2+outputIndexccc]
#            measurePopNext[nPop/2:nPop,:] = ParentChildMeasure[ParentChildTmpIdx2,:]
#            fitnessPopNext[nPop/2:nPop] = ParentChildFitness[ParentChildTmpIdx2]
#            
#            Idxrp = np.random.permutation(nPop); ## randomly change the order of the population
#            measurePop = measurePopNext[Idxrp,:]
#            fitnessPop = fitnessPopNext[Idxrp]
#            
#            a = np.minimum(1, np.exp(np.sum(fitnessPop)-np.sum(fitnessPopPrev))) ## fitness change ratio
#            ParentChildTmpIdxall = np.concatenate((ParentChildTmpIdx1, ParentChildTmpIdx2))
#            achild = np.sum(ParentChildTmpIdxall > nPop)/nPop ## Percentage of children were kept for the next iteration
#            
#            ## Update best answer - printed to the terminal
#            if(np.max(fitnessPop) > mVal):
#                mVal_before = mVal
#                mIdx = (-fitnessPop).argsort()[0]
#                mVal = fitnessPop[mIdx]
#                measure = measurePop[mIdx,:]
#                print(f'{mVal}')
#                print(measure)
#            
#            ###################################################################
#            ##################### Update Analysis Tracker #####################
#            ###################################################################
#            if (Parameters.analysis == True): ## record all intermediate process (be aware of the possible memory limit!)
#                Analysis['ParentChildMeasure'][:,:,t] = ParentChildMeasure
#                Analysis['ParentChildFitness'][t,:] = ParentChildFitness
#                Analysis['ParentChildTmpIdx2'][t,:] = ParentChildTmpIdx2
#                Analysis['Idxrp'][t,:] = Idxrp
#                Analysis['measurePop'][:,:,t] = measurePop
#                Analysis['fitnessPop'][t,:] = fitnessPop
#                Analysis['measureiter'][t,:] = measure
#                Analysis['ratioa'][t] = a
#                Analysis['ratioachild'][t] = achild
#                Analysis['ratiomVal'][t] = mVal
#                Analysis['ratio'][t] = exp(sum(fitnessPop)-sum(fitnessPopPrev))
#                Analysis['JumpType'][t,:] = JumpType
#                Analysis['ElemIdxUpdated'][t,:] = ElemIdxUpdated
#                Analysis['subsetIntervalnPop'][:,:,t] = subsetIntervalnPop  
#            
#            ## Update terminal 
#            if(not(t % 10)):
#                print(f'Iteration: {str(t)}')
#    
#                ## Stop if we've found a measure meeting our desired level of fitness
#                if (np.abs(mVal - mVal_before) <= Parameters.fitnessThresh):
#                    break
#    
#        return measure, initialMeasure, Analysis
    
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
            ci = self.compute_chi(Bags[b_idx])
            if(Labels[b_idx] == 0):  ## Negative bag, label = 0
                fitness = fitness - (np.mean(np.power(ci,(2*p1))))^(1/p1)
            else: ## Positive bag, label=1
                fitness = fitness - (np.mean(np.power((ci-1),(2*p2))))^(1/p2)
    
            ## Sanity check
            if (np.isinf(fitness) or not(np.isreal(fitness)) or np.isnan(fitness)):
                fitness = np.real(fitness)
                fitness = -10000000
                
        return fitness
    
    def sampleMeasure(self, nSources, lowerindex, upperindex,  prevSample=None, IndexsubsetToChange=None, sampleVar=None):
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
    
        if(prevSample == None):
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
            
            nElements = self.N
            measure = prevSample
            if ((IndexsubsetToChange<=nSources) and (IndexsubsetToChange>=0)): ## singleton
                lowerBound = 0
                upperBound = np.amin(measure[upperindex[IndexsubsetToChange]])
            elif ((IndexsubsetToChange>=(nElements-nSources)) and (IndexsubsetToChange<=(nElements-1))): ## (nSources-1)-tuple
                lowerBound = np.amax(measure[lowerindex[IndexsubsetToChange]])
                upperBound = 1
            else:  ## remaining elements
                lowerBound = np.amax(measure[lowerindex[IndexsubsetToChange]]) 
                upperBound = np.amin(measure[upperindex[IndexsubsetToChange]]) 
            
            denom = upperBound - lowerBound
            v_bar = sampleVar/((denom**2)+1e-5) ## -changed
            x_bar = measure[IndexsubsetToChange]; ## -changed
            
            ## Sample from a Truncated Gaussian
            sigma_bar = np.sqrt(v_bar)
            c2 = np.random.uniform(low=0,high=1) ## Randomly generate a value between [0,1]
            val = self.invcdf_TruncatedGaussian (c2,x_bar,sigma_bar,lowerBound,upperBound) ## New measure element value
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
    
    
#    def evalInterval(self, measure, lowerindex, upperindex):
#        """
#        =======================================================================
#        % Evaluate the valid interval width of a measure, then sort in descending order.
#        %
#        % INPUT
#        %   measure -  measure to be evaluated after update
#        %   lowerindex - the cell that stores all the corresponding subsets (lower index) of measure elements
#        %   upperindex - the cell that stores all the corresponding supersets (upper index) of measure elements
#        % OUTPUT
#        %   subsetInterval - the valid interval widths for the measure elements,before sorting by value
#        %
#        =======================================================================
#        """
#        
#        nElements = self.nElements 
#        nSources = self.N
#        lb = np.zeros(nElements-1)
#        ub = np.zeros(nElements-1)
#        
#        for j in range(nSources)  ## singleton
#            lb[j] = 0 ## lower bound
#            ub[j] = min(measure(upperindex{j})); ## upper bound
#    
#        for j in range(nSources, nElements-nSources-1):
#            lb[j] = max(measure(lowerindex{j})); ## lower bound
#            ub[j] = min(measure(upperindex{j})); ## upper bound
#        
#        for j in range(nElements-nSources,nElements-1)
#            lb[j] = max(measure(lowerindex{j})) ## lower bound
#            ub[j] = 1 ## upper bound
#            
#        subsetInterval = ub - lb   
#        
#        return subsetInterval
    
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

#    def get_fm_class_img_coeff(self, Lattice, h, fm_len):  # Lattice is FM_name_and_index, h is a sample, fm_len
#        """
#        This creates a FM map with the name as the key and the index as the value
#
#        :param Lattice: dictionary with FM
#        :param h: sample
#        :param fm_len: fm length
#        :return: the fm_coeff
#        """
#
#        n = len(h)  # len(h) is the number of the samples
#        fm_coeff = np.zeros(fm_len)
#        pi_i = np.argsort(h)[::-1][:n] + 1 ## sorted smallest to largest
#        for i in range(1, n):
#            fm_coeff[Lattice[str(np.sort(pi_i[:i]))]] = h[pi_i[i - 1] - 1] - h[pi_i[i] - 1]
#        fm_coeff[Lattice[str(np.sort(pi_i[:n]))]] = h[pi_i[n - 1] - 1]
#        np.matmul(fm_coeff, np.transpose(fm_coeff))
#        return fm_coeff

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
        
            
            
    
    