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

import numpy as np
import itertools
from cvxopt import solvers, matrix

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
        self.nElements = (2^(self.N))-1 ## number of measure elements, including mu_all=1
        
        ## Initialize dictionary of fuzzy measure elements
        fm_len = 2 ** self.N - 1  # nc
        index_keys = self.get_keys_index()
        
        ## Get measure information
        
        
        ## Estimate measure parameters
        print("Number Sources : ", self.N, "; Number Training Bags : ", self.B)
        self.fm = self.produce_lattice_softmax(Bags, Labels, Parameters, trueInitMeasure=None)

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
    
    def produce_lattice_softmax(Bags, Labels, Parameters, trueInitMeasure=None):
        
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
        
#        ## Compute lower bound index
#        nElem_prev = 0
#        nElem_prev_prev=0
#        
#        for i = 2:nSources-1 %sample rest
#            nElem = MeasureSection.NumEach(i);%the number of combinations, e.g.,3
#            elem = MeasureSection.Each{i};%the number of combinations, e.g., (1,2),(1,3),(2,3)
#            nElem_prev = nElem_prev+MeasureSection.NumEach(i-1);
#            if i==2
#                nElem_prev_prev = 0;
#            elseif i>2
#                nElem_prev_prev  = nElem_prev_prev + MeasureSection.NumEach(i-2);
#            end
#            for j = 1:nElem
#                Parameters.lowerindex{nElem_prev+j}  =[];
#                elemSub = nchoosek(elem(j,:), i-1);
#                for k = 1:length(elemSub) %it needs a length(elemSub) because it is taking subset of the element rows
#                    tindx = elemSub(k,:);
#                    [Locb] = ismember_findRow(tindx,MeasureSection.Each{i-1});
#                    Parameters.lowerindex{nElem_prev+j} = horzcat(Parameters.lowerindex{nElem_prev+j} , nElem_prev_prev+Locb);
#                end
#            end
#        end
#        
#        %compute upper bound index
#        nElem_nextsum = 2^nSources-1;%total length of measure
#        for i = nSources-2:-1:1 %sample rest
#            nElem = MeasureSection.NumEach(i);%the number of combinations, e.g.,3
#            elem = MeasureSection.Each{i};%the number of combinations, e.g., (1,2),(1,3),(2,3)
#            %nElem_next = MeasureSection.NumEach(i+1);
#            elem_next = MeasureSection.Each{i+1};
#            
#            nElem_nextsum = nElem_nextsum - MeasureSection.NumEach(i+1); %cumulative sum of how many elements in the next tier so far
#            for j = nElem:-1:1
#                Parameters.upperindex{nElem_nextsum-nElem+j-1}  =[];
#                elemSub = elem(j,:); 
#                [~,~,row_id1] = ismember_findrow_mex_my(elemSub,elem_next);
#                Parameters.upperindex{nElem_nextsum-nElem+j-1} = horzcat(Parameters.upperindex{nElem_nextsum-nElem+j-1} , nElem_nextsum+row_id1-1);
#            end
#        end
#        
#        %Create oneV cell matrix
#        oneV = cell(n_bags,1);
#        for i  = 1:n_bags
#            oneV{i} = ones(nPntsBags(i), 1);
#        end
#        
#        lowerindex = Parameters.lowerindex;
#        upperindex = Parameters.upperindex;
#        sampleVar = Parameters.sampleVar;
        
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
                measurePop[i,:] = sampleMeasure(nSources,lowerindex,upperindex);
                fitnessPop[i] = self.evalFitness_softmax(Bags, Labels, measurePop[i,:])
            
            
        indx = np.argmax(fitnessPop)
        mVal = fitnessPop[indx]
        measure = measurePop[indx,:]  
        initialMeasure = measure
        mVal_before = -10000
        
        #######################################################################
        #################### Iterate through Optimization #####################
        #######################################################################
        for t in range(Parameters.maxIterations):
            childMeasure = measurePop
            childFitness = np.zeros(nPop) ## Pre-allocate array
            subsetIntervalnPop = np.zeros((nPop,nElements-1)) ## Dont need to sample mu_all=1 (last element)
            ElemIdxUpdated = np.zeros(nPop)
            JumpType = np.zeros(nPop)
            for i = 1:Parameters.nPop
                [subsetInterval] = evalInterval(measurePop(i,:),nSources,lowerindex,upperindex);
                subsetIntervalnPop(i,:)=subsetInterval;
                %Sample new value
                z = rand(1);
                if(z < eta) %Small-scale mutation: update only one element of the measure
                    [iinterv] = sampleMultinomial_mat(subsetInterval, 1, 'descend' );%update the one interval according to multinomial
                    childMeasure(i,:) =  sampleMeasure(nSources, lowerindex, upperindex, childMeasure(i,:),iinterv, sampleVar);
                    childFitness(i) = evalFitness_softmax(Labels, childMeasure(i,:), nPntsBags, oneV, bag_row_ids, diffM,p);
                    JumpType(i) = 1;
                    ElemIdxUpdated(i) = iinterv;
                else %Large-scale mutation: update all measure elements sort by valid interval widths in descending order
                        [~, indx_subsetInterval] = sort(subsetInterval,'descend'); %sort the intervals through descending order
                    for iinterv = 1:size(indx_subsetInterval,2) %for all elements
                        childMeasure(i,:) =  sampleMeasure(nSources, lowerindex, upperindex, childMeasure(i,:),iinterv, sampleVar);
                        childFitness(i) = evalFitness_softmax(Labels, childMeasure(i,:), nPntsBags, oneV, bag_row_ids, diffM,p);
                        JumpType(i) = 2;
                    end
                end
            end  %in order to do par-for
            
            
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
            fitnessPopPrev = fitnessPop;
            ParentChildMeasure = vertcat(measurePop,childMeasure); %total 2xNPop measures
            ParentChildFitness = horzcat(fitnessPop,childFitness);
            [ParentChildFitness_sortV,ParentChildFitness_sortIdx] = sort(ParentChildFitness,'descend');
            
            measurePopNext = zeros(Parameters.nPop,size(ParentChildMeasure,2));
            fitnessPopNext = zeros(1,Parameters.nPop);
            ParentChildTmpIdx1 = ParentChildFitness_sortIdx(1:Parameters.nPop/2);
            measurePopNext(1:Parameters.nPop/2,:) = ParentChildMeasure(ParentChildTmpIdx1 , : );
            fitnessPopNext(1:Parameters.nPop/2) = ParentChildFitness(ParentChildTmpIdx1);
            
            %%% Method 1: Random sample - Correct but not used in the paper
            %          ParentChildTmpIdx2 = randsample(ParentChildFitness_sortIdx(Parameters.nPop/2+1:end),Parameters.nPop/2);
            
            %%% Method 2: Sample by multinomial distribution based on fitness
            PDFdistr75 = ParentChildFitness_sortV(Parameters.nPop/2+1:end);
            [outputIndexccc] = sampleMultinomial_mat(PDFdistr75, Parameters.nPop/2, 'descend' );
            
            ParentChildTmpIdx2  =   ParentChildFitness_sortIdx(Parameters.nPop/2+outputIndexccc);
            measurePopNext(Parameters.nPop/2+1 : Parameters.nPop,:) = ParentChildMeasure(ParentChildTmpIdx2 , : );
            fitnessPopNext(Parameters.nPop/2+1 : Parameters.nPop) = ParentChildFitness(ParentChildTmpIdx2);
            
            Idxrp = randperm(Parameters.nPop); %randomly change between the order of the population
            measurePop = measurePopNext(Idxrp,:);
            fitnessPop = fitnessPopNext(Idxrp);
            
            a = min(1, exp(sum(fitnessPop)-sum(fitnessPopPrev)));  %fitness change ratio
            ParentChildTmpIdxall = [ParentChildTmpIdx1 ParentChildTmpIdx2];
            achild = sum(ParentChildTmpIdxall>Parameters.nPop)/Parameters.nPop; %percentage of child that has been kept till the next iteration
            
            %update best answer - outputted to command window
            if(max(fitnessPop) > mVal)
                 mVal_before = mVal;
                [mVal, mIdx] = max(fitnessPop);
                measure = measurePop(mIdx,:);
                disp(mVal);
                disp(measure);
            end
            
            ###################################################################
            ##################### Update Analysis Tracker #####################
            ###################################################################
            if (Parameters.analysis == True): ## record all intermediate process (be aware of the possible memory limit!)
                Analysis['ParentChildMeasure'][:,:,t] = ParentChildMeasure
                Analysis['ParentChildFitness'][t,:] = ParentChildFitness
                Analysis['ParentChildTmpIdx2'][t,:] = ParentChildTmpIdx2
                Analysis['Idxrp'][t,:] = Idxrp
                Analysis['measurePop'][:,:,t] = measurePop
                Analysis['fitnessPop'][t,:] = fitnessPop
                Analysis['measureiter'][t,:] = measure
                Analysis['ratioa'][t] = a
                Analysis['ratioachild'][t] = achild
                Analysis['ratiomVal'][t] = mVal
                Analysis['ratio'][t] = exp(sum(fitnessPop)-sum(fitnessPopPrev))
                Analysis['JumpType'][t,:] = JumpType
                Analysis['ElemIdxUpdated'][t,:] = ElemIdxUpdated
                Analysis['subsetIntervalnPop'][:,:,t] = subsetIntervalnPop  
            
            ## Update terminal 
            if(not(t % 10)):
                print(f'Iteration: {str(t)}')
    
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
    
def sampleMeasure(nSources, lowerindex, upperindex,  prevSample, IndexsubsetToChange, sampleVar):
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
    % Written by: X. Du 03/2018
    %
    ===========================================================================
    """

    if(nargin < 4):
        
        % sample a brand new measure
        % Flip a coin to decide whether to sample from above or sample from bottom
        %sample new measure, prior is uniform/no prior
        coin = rand(1);
        if coin >= 0.5 % sample from bottom
            [measure] = sampleMeasure_Bottom(nSources,lowerindex);
        else   % sample from above
            [measure] = sampleMeasure_Above(nSources,upperindex);
        end
    else
        %sample just one new element of measure
        
        Nmeasure = 2^nSources-1;
        measure = prevSample;
        if (IndexsubsetToChange<=nSources) && (IndexsubsetToChange>=1) %singleton
            lowerBound = 0; 
            upperBound = min(measure(upperindex{IndexsubsetToChange})); 
        elseif (IndexsubsetToChange>=(Nmeasure-nSources)) && (IndexsubsetToChange<=(Nmeasure-1)) %(nSources-1)-tuple
            lowerBound = max(measure(lowerindex{IndexsubsetToChange})); 
            upperBound = 1;
        else  %remaining elements
            lowerBound = max(measure(lowerindex{IndexsubsetToChange})); 
            upperBound = min(measure(upperindex{IndexsubsetToChange})); 
        end
        
        denom = upperBound - lowerBound;
        v_bar = sampleVar/(denom^2+eps);%-changed
        x_bar = measure(IndexsubsetToChange); %-changed
        
        %% sample from a Truncated Gaussian
        sigma_bar = sqrt(v_bar);
        c2 = rand(1);       % randomly generate a value between [0,1]
        [val] = invcdf_TruncatedGaussian (c2,x_bar,sigma_bar,lowerBound,upperBound); % new measure element value
        measure(IndexsubsetToChange) = val;  % new measure element value
    end

    return measure

    def build_constraint_matrices(self, index_keys, fm_len):
        """
        This method builds the necessary constraint matrices.

        :param index_keys: map to reference lattice components
        :param fm_len: length of the fuzzy measure
        :return: the constraint matrices
        """

        vls = np.arange(1, self.N + 1)
        line = np.zeros(fm_len)
        G = line
        line[index_keys[str(np.array([1]))]] = -1.
        h = np.array([0])
        for i in range(2, self.N + 1):
            line = np.zeros(fm_len)
            line[index_keys[str(np.array([i]))]] = -1.
            G = np.vstack((G, line))
            h = np.vstack((h, np.array([0])))
        for i in range(2, self.N + 1):
            parent = np.array(list(itertools.combinations(vls, i)))
            for latt_pt in parent:
                for j in range(len(latt_pt) - 1, len(latt_pt)):
                    children = np.array(list(itertools.combinations(latt_pt, j)))
                    for latt_ch in children:
                        line = np.zeros(fm_len)
                        line[index_keys[str(latt_ch)]] = 1.
                        line[index_keys[str(latt_pt)]] = -1.
                        G = np.vstack((G, line))
                        h = np.vstack((h, np.array([0])))

        line = np.zeros(fm_len)
        line[index_keys[str(vls)]] = 1.
        G = np.vstack((G, line))
        h = np.vstack((h, np.array([1])))

        # equality constraints
        A = np.zeros((1, fm_len))
        A[0, -1] = 1
        b = np.array([1]);

        return G, h, A, b


    def get_fm_class_img_coeff(self, Lattice, h, fm_len):  # Lattice is FM_name_and_index, h is a sample, fm_len
        """
        This creates a FM map with the name as the key and the index as the value

        :param Lattice: dictionary with FM
        :param h: sample
        :param fm_len: fm length
        :return: the fm_coeff
        """

        n = len(h)  # len(h) is the number of the samples
        fm_coeff = np.zeros(fm_len)
        pi_i = np.argsort(h)[::-1][:n] + 1 ## sorted smallest to largest
        for i in range(1, n):
            fm_coeff[Lattice[str(np.sort(pi_i[:i]))]] = h[pi_i[i - 1] - 1] - h[pi_i[i] - 1]
        fm_coeff[Lattice[str(np.sort(pi_i[:n]))]] = h[pi_i[n - 1] - 1]
        np.matmul(fm_coeff, np.transpose(fm_coeff))
        return fm_coeff


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
        
        if self.type == 'quad':
            n = len(x2)
            pi_i = np.argsort(x2)[::-1][:n] + 1
            ch = x2[pi_i[0] - 1] * (self.fm[str(pi_i[:1])])
            for i in range(1, n):
                latt_pti = np.sort(pi_i[:i + 1])
                latt_ptimin1 = np.sort(pi_i[:i])
                ch = ch + x2[pi_i[i] - 1] * (self.fm[str(latt_pti)] - self.fm[str(latt_ptimin1)])
            
            sort_idx = [idx for idx, val in enumerate(self.all_sorts) if (sum(pi_i == val ) == n)]
            
            return ch, int(sort_idx[0]+1)
        else:
            print("If using sugeno measure, you need to use chi_sugeno.")
            
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
        
            
            
    
    