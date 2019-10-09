#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:18:33 2019

@author: shreyasaxena
"""

import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)
from ssm.models import HMM
import os
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

K=2 # number of discrete states
filename = 'neural'
numiters=40 # Probably need higher number of iterations - look at log likelihoods to see if it converged
kappa=10
savedir = '/Users/shreyasaxena/Dropbox/Documents/Labwork/Columbia/Data/AnneChurchland/AllData/'
resultsdir = '/Users/shreyasaxena/Dropbox/Documents/Labwork/Columbia/Code/Packages/ssm_glaser/ssm_v2/AnneChurchland/results_pca_arhmm/'
animals=['mSM30','mSM30','mSM34','mSM34','mSM36','mSM36','mSM46','mSM46','mSM57','mSM57',
         'mSM43','mSM43','mSM44','mSM44','mSM53','mSM53','mSM55','mSM55','mSM56','mSM56']
sessions=['10-Oct-2017','12-Oct-2017','01-Dec-2017','02-Dec-2017','05-Dec-2017','07-Dec-2017','01-Dec-2017','13-Jan-2018',
          '02-Feb-2018','08-Feb-2018',
          '21-Nov-2017','23-Nov-2017','21-Nov-2017','29-Nov-2017','14-Mar-2018','21-Mar-2018','13-Feb-2018',
          '16-Feb-2018','22-Feb-2018','27-Feb-2018']

if not os.path.isdir(resultsdir): os.mkdir(resultsdir)
for animaliter in np.arange(len(animals)):
    # Parameters
    animal=animals[animaliter]
    session=sessions[animaliter]
    print(animal); print(session)
    if not os.path.isdir(os.path.join(resultsdir,animal)): os.mkdir(os.path.join(resultsdir,animal))

    resultsfolder=os.path.join(resultsdir,animal,session)
    if not os.path.isdir(resultsfolder): os.mkdir(resultsfolder)
    
    print('Number of latents per area: ', 1)
    print('filename: ',filename)
    print('K: ',K)
    datafilepath=os.path.join(savedir,animal,session,filename+'.mat')
    if not os.path.exists(datafilepath): continue

    data=sio.loadmat(datafilepath)
    neur_data=data['neural']

    reg_indxs=data['reg_indxs']
    areanames=reg_indxs.dtype.fields.keys(); areas=[None]*len(areanames)
    for i,area in enumerate(areanames):
        areas[i]=area

    # Do PCA on all components for each region / area
    areas=np.sort(areas)
    (numtrials,numtimepoints,ncomps)=neur_data.shape
    neur_data=np.reshape(neur_data,(neur_data.shape[0]*neur_data.shape[1],neur_data.shape[2]))
    pcdata=np.zeros((neur_data.shape[0],len(areas)))
    for i,area in enumerate(areas):
        pca = PCA(n_components=1,whiten=True)
        indstokeep=reg_indxs[area][0][0][0]-1
        pcdata[:,i]=np.squeeze(pca.fit_transform(neur_data[:,indstokeep.flatten()]))
    pcdata=np.reshape(pcdata,(numtrials,numtimepoints,len(areas)))
    print(pcdata.shape)
    
    # Set the parameters of the HMM
    T = pcdata.shape[1]    # number of time bins
    K = K       # number of discrete states
    N = pcdata.shape[2]      # number of observed dimensions
    
    print('T:',T,', K:',K,', N:',N)
    
    # Put data in list format to feed into the HMM
    trialdata=[pcdata[i,:,:] for i in np.arange(pcdata.shape[0])]
    
    # shuffle the trials
    sequence = [i for i in range(len(trialdata))]
    npr.shuffle(sequence)
    sequence=np.array(sequence)
    
    # Divide into training and testing (I didn't really end up using the testing - but can check log likelihoods to decide K)
    traintrials=[trialdata[j] for j in sequence[:int(np.ceil(0.8*len(trialdata)))]]
    testtrials=[trialdata[j] for j in sequence[int(np.ceil(0.8*len(trialdata))):]]
    print(len(traintrials)); print(len(testtrials))              
    
    # Run the ARHMM
    arhmm = HMM(K,N, observations="ar",
                transitions="sticky",
                transition_kwargs=dict(kappa=kappa))
    
    arhmm_em_lls = arhmm.fit(traintrials, method="em", num_em_iters=numiters)
    
    # Get the inferred states for train and test trials
    traintrials_z=[arhmm.most_likely_states(traintrial) for traintrial in traintrials]
    traintrials_z=np.asarray(traintrials_z)
    testtrials_z=[arhmm.most_likely_states(testtrial) for testtrial in testtrials]
    testtrials_z=np.asarray(testtrials_z)


    As=[None]*K; maxvals=[None]*K
    for k in np.arange(K):
        As[k]=arhmm.params[2][0][k,:,:]; 
        maxvals[k]=np.var(As[k])    # Tried to permute the states so that it would be 'no movement' --> 'movement', based on the variance of the values in the A matrix (didn't really work)
    
    # permute the states
    sortorder=np.argsort(maxvals)
    sortedmaxvals=np.sort(maxvals)
    print(sortorder); print(sortedmaxvals)
    As=[As[i] for i in sortorder]

    traintrials_sorted=traintrials_z.copy(); testtrials_sorted=testtrials_z.copy()
    for k in np.arange(K):
        traintrials_sorted[traintrials_z==sortorder[k]]=k
        testtrials_sorted[testtrials_z==sortorder[k]]=k
    
    # Plot states of all trials
    fig = plt.figure(figsize=(8, 10))
    plt.imshow(traintrials_sorted, aspect="auto", vmin=0, vmax=K)
    fig.savefig(os.path.join(resultsfolder,'traintrials_K'+str(K)+'.png'))
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(testtrials_sorted, aspect="auto", vmin=0, vmax=K)
    fig.savefig(os.path.join(resultsfolder,'testtrials_K'+str(K)+'.png'))
    
    # Plot the A matrices
    fig = plt.figure(figsize=(8*K,8))
    for k in np.arange(K):
        ax = plt.subplot(1,K,k+1)
        plt.imshow(As[k],cmap='jet'); 
        plt.clim([-0.2,0.2]); 
        plt.xticks(np.arange(len(areas)),areas,rotation=45,fontsize=10)
        plt.yticks(np.arange(len(areas)),areas,fontsize=10)
        plt.title('State'+str(k)+':'+'Var='+"%8.4f" % sortedmaxvals[k]); 
    fig.savefig(os.path.join(resultsfolder,'As_K'+str(K)+'.png'))

    mydict={'As':As,'traintrials_sorted':traintrials_sorted,
            'testtrials_sorted':testtrials_sorted,
            'sortedmaxvals':sortedmaxvals,'arhmm':arhmm,
            'arhmm_em_lls':arhmm_em_lls,
            'traintrials_z':traintrials_z, 'testtrials_z':testtrials_z,
            'sequence':sequence, 'trialdata':trialdata, 'traintrials':traintrials,
            'testtrials':testtrials}
    savefile = os.path.join(resultsfolder,'pca_arhmm_K'+str(K)+'.p')
    f = open(savefile, 'wb')
    pickle.dump(mydict, f)          # dump data to f
    f.close()   
    print('File saved at ',savefile)
