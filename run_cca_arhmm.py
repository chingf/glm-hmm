#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:47:38 2019

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
from sklearn.cross_decomposition import CCA

K=2 # number of discrete states
do_cca=1
filename = 'neural'
numiters=40
kappa=10

concatenate_sessions=0
whiten=1
zscore=1
regularize=0; lambda_reg=1000

constring='_separate'; 
if concatenate_sessions: constring='_concatenated'


savedir = '/Users/shreyasaxena/Dropbox/Documents/Labwork/Columbia/Data/AnneChurchland/AllData/'
resultsdir = '/Users/shreyasaxena/Dropbox/Documents/Labwork/Columbia/Code/Packages/ssm_glaser/ssm_v2/AnneChurchland/results_pca_arhmm/'
animals=['mSM30','mSM30','mSM34','mSM34','mSM36','mSM36','mSM46','mSM46','mSM57','mSM57',
         'mSM43','mSM43','mSM44','mSM44','mSM53','mSM53','mSM55','mSM55','mSM56','mSM56']
sessions=['10-Oct-2017','12-Oct-2017','01-Dec-2017','02-Dec-2017','05-Dec-2017','07-Dec-2017','01-Dec-2017','13-Jan-2018',
          '02-Feb-2018','08-Feb-2018',
          '21-Nov-2017','23-Nov-2017','21-Nov-2017','29-Nov-2017','14-Mar-2018','21-Mar-2018','13-Feb-2018',
          '16-Feb-2018','22-Feb-2018','27-Feb-2018']

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    X_white = np.dot(U, Vt)

    return X_white

if not os.path.isdir(resultsdir): os.mkdir(resultsdir)
for animaliter in np.arange(0,len(animals),2):
    # Parameters
    animal=animals[animaliter]
    session1=sessions[animaliter]
    session2=sessions[animaliter+1]
    print(animal); print(session1); print(session2)
    if not os.path.isdir(os.path.join(resultsdir,animal)): os.mkdir(os.path.join(resultsdir,animal))

    resultsfolder=os.path.join(resultsdir,animal)
    if not os.path.isdir(resultsfolder): os.mkdir(resultsfolder)
    
    print('Number of latents per area: ', 1)
    print('filename: ',filename)
    print('K: ',K)
    whitenstring=''; 
    if whiten: whitenstring='_whitened'
    
    data_ses1=sio.loadmat(savedir+animal+'/'+session1+'/neural.mat')
    data_ses2=sio.loadmat(savedir+animal+'/'+session2+'/neural.mat')
    
    neur_data_ses1=data_ses1['neural']
    print(neur_data_ses1.shape)
    reg_indxs_ses1=data_ses1['reg_indxs']
    areanames_ses1=reg_indxs_ses1.dtype.fields.keys(); 
    areas_ses1=[None]*len(areanames_ses1)
    for i,area in enumerate(areanames_ses1):
        areas_ses1[i]=area
    
    areas_ses1=np.sort(areas_ses1).tolist()
    
    neur_data_ses2=data_ses2['neural']
    print(neur_data_ses2.shape)
    reg_indxs_ses2=data_ses2['reg_indxs']
    areanames_ses2=reg_indxs_ses2.dtype.fields.keys(); 
    areas_ses2=[None]*len(areanames_ses2)
    for i,area in enumerate(areanames_ses2):
        areas_ses2[i]=area
    
    areas_ses2=np.sort(areas_ses2).tolist()
    numtrials_ses1=neur_data_ses1.shape[0]
    numtrials_ses2=neur_data_ses2.shape[0]
    
    numtimepoints=neur_data_ses1.shape[1]
    
    for area in areas_ses1: 
        if area not in areas_ses2: 
            print(area)
            areas_ses1.remove(area)
    for area in areas_ses2: 
        if area not in areas_ses1: 
            print(area)
            areas_ses2.remove(area)
    while len(areas_ses1) != len(areas_ses2):
        for area in areas_ses1: 
            if area not in areas_ses2: 
                print(area)
                areas_ses1.remove(area)
        for area in areas_ses2: 
            if area not in areas_ses1: 
                print(area)
                areas_ses2.remove(area)

    neur_mean_ses1=np.squeeze(np.mean(neur_data_ses1,axis=0))
    neur_mean_ses2=np.squeeze(np.mean(neur_data_ses2,axis=0))
    latentmean_ses1=np.zeros((neur_mean_ses1.shape[0],len(areas_ses1)))
    latentmean_ses2=np.zeros((neur_mean_ses2.shape[0],len(areas_ses2)))
    neur_data_ses1=np.reshape(neur_data_ses1,(neur_data_ses1.shape[0]*numtimepoints,neur_data_ses1.shape[2]))
    neur_data_ses2=np.reshape(neur_data_ses2,(neur_data_ses2.shape[0]*numtimepoints,neur_data_ses2.shape[2]))
    
    latents_ses1=np.zeros((neur_data_ses1.shape[0],len(areas_ses1)))
    latents_ses2=np.zeros((neur_data_ses2.shape[0],len(areas_ses2)))
    scores_mean_area=np.zeros(len(areas_ses1))
    
    for i,area in enumerate(areas_ses2):
        indstokeep_ses1=reg_indxs_ses1[area][0][0][0]-1
        neur_mean_ses1_area=neur_mean_ses1[:,indstokeep_ses1.flatten()]
        
        indstokeep_ses2=reg_indxs_ses2[area][0][0][0]-1
        neur_mean_ses2_area=neur_mean_ses2[:,indstokeep_ses2.flatten()]
        
        if do_cca:
            # Do CCA for all components in one region across sessions, to get the most correlated signals
            cca=CCA(n_components=1)
            
            Xcm, Ycm = cca.fit_transform(neur_mean_ses1_area, neur_mean_ses2_area)
            
            latentmean_ses1[:,i] = np.squeeze(Xcm) 
            latentmean_ses2[:,i] = np.squeeze(Ycm)
            
            Xc,Yc=cca.transform(X=neur_data_ses1[:,indstokeep_ses1.flatten()],
                                Y=neur_data_ses2[:,indstokeep_ses2.flatten()])
            
            latents_ses1[:,i]=np.squeeze(Xc)
            latents_ses2[:,i]=np.squeeze(Yc)
            cca_str='_cca'
        else:
            # Do PCA for all components in one region across sessions, to get most correlated signals
            pca = PCA(n_components=1)
            latents_ses1[:,i]=np.squeeze(pca.fit_transform(neur_data_ses1[:,indstokeep_ses1.flatten()]))
            latentmean_ses1[:,i] = np.squeeze(pca.transform(neur_mean_ses1_area))
            
            pca = PCA(n_components=1)
            latents_ses2[:,i]=np.squeeze(pca.fit_transform(neur_data_ses2[:,indstokeep_ses2.flatten()]))
            latentmean_ses2[:,i] = np.squeeze(pca.transform(neur_mean_ses2_area))
            cca_str='_pca'
        
        scores_mean_area[i] = np.corrcoef(latentmean_ses1[:,i], latentmean_ses2[:,i])[0,1]
    
    if whiten:
        # Whiten the latents
        latents_ses1=svd_whiten(latents_ses1)
        latents_ses2=svd_whiten(latents_ses2)
    
    if zscore:
        # Z-score the latents
        latents_ses1=(latents_ses1-np.mean(latents_ses1,axis=0))/np.std(latents_ses1,axis=0)
        latents_ses2=(latents_ses2-np.mean(latents_ses2,axis=0))/np.std(latents_ses2,axis=0)
    
    latents_ses1=np.reshape(latents_ses1,(numtrials_ses1,numtimepoints,len(areas_ses1)))
    latents_ses2=np.reshape(latents_ses2,(numtrials_ses2,numtimepoints,len(areas_ses2)))
    
        # Set the parameters of the HMM
    T = latents_ses1.shape[1]    # number of time bins
    K = K       # number of discrete states
    N = latents_ses1.shape[2]      # number of observed dimensions
    
    print('T:',T,', K:',K,', N:',N)
    
    trialdata_ses1=[latents_ses1[i,:,:] for i in np.arange(latents_ses1.shape[0])]
    
    if concatenate_sessions:
        trialdata_ses1=trialdata_ses1+[latents_ses2[i,:,:] for i in np.arange(latents_ses2.shape[0])]
        
    else:
        trialdata_ses2=[latents_ses2[i,:,:] for i in np.arange(latents_ses2.shape[0])]
    
    if regularize:
        arhmm_ses1 = HMM(K,N, observations="ar",
                    transitions="sticky",
                    transition_kwargs=dict(kappa=10),
                    observation_kwargs=dict(regularization_params=dict(type='l2',lambda_A=lambda_reg)))
        regularizestring='_regl2'
    else:
        arhmm_ses1 = HMM(K,N, observations="ar",
                    transitions="sticky",
                    transition_kwargs=dict(kappa=10))
        regularizestring=''

    arhmm_em_lls_ses1 = arhmm_ses1.fit(trialdata_ses1, method="em", num_em_iters=numiters)
    
    # Get the inferred states for session 1
    trialdata_ses1_z=[arhmm_ses1.most_likely_states(trial) for trial in trialdata_ses1]
    trialdata_ses1_z=np.asarray(trialdata_ses1_z)
    As_ses1=[None]*K; 
    for k in np.arange(K):
        As_ses1[k]=arhmm_ses1.params[2][0][k,:,:]; 
    
    fig = plt.figure(figsize=(8, 10))
    plt.imshow(trialdata_ses1_z, aspect="auto", vmin=0, vmax=K)
    plt.plot([54,54],[0,trialdata_ses1_z.shape[0]],'-r')
    plt.ylim([0,trialdata_ses1_z.shape[0]])
    fig.savefig(os.path.join(resultsfolder,'ses1'+cca_str+constring+'K'+str(K)+whitenstring+regularizestring+'.png'))
    
    fig = plt.figure(figsize=(8*K,8))
    for k in np.arange(K):
        ax = plt.subplot(1,K,k+1)
        plt.imshow(As_ses1[k],cmap='jet'); 
        plt.clim([-0.2,0.2]); 
        plt.xticks(np.arange(len(areas_ses1)),areas_ses1,rotation=45,fontsize=10)
        plt.yticks(np.arange(len(areas_ses1)),areas_ses1,fontsize=10)
        plt.title('State'+str(k)); 
    fig.savefig(os.path.join(resultsfolder,'As_ses1'+cca_str+constring+'K'+str(K)+whitenstring+regularizestring+'.png'))
    
    if not concatenate_sessions:
        N = latents_ses2.shape[2]      # number of observed dimensions
        
        print('T:',T,', K:',K,', N:',N)
        
        if regularize:
            arhmm_ses2 = HMM(K,N, observations="ar",
                        transitions="sticky",
                        transition_kwargs=dict(kappa=10),
                        observation_kwargs=dict(regularization_params=dict(type='l2',lambda_A=lambda_reg)))
        else:
            arhmm_ses2 = HMM(K,N, observations="ar",
                        transitions="sticky",
                        transition_kwargs=dict(kappa=10))
        
        arhmm_em_lls_ses2 = arhmm_ses2.fit(trialdata_ses2, method="em", num_em_iters=numiters)

        # Get the inferred states for session 2
        trialdata_ses2_z=[arhmm_ses2.most_likely_states(trial) for trial in trialdata_ses2]
        trialdata_ses2_z=np.asarray(trialdata_ses2_z)

        As_ses2=[None]*K; 
        for k in np.arange(K):
            As_ses2[k]=arhmm_ses2.params[2][0][k,:,:]; 
    
        fig = plt.figure(figsize=(8, 10))
        plt.imshow(trialdata_ses2_z, aspect="auto", vmin=0, vmax=K)
        plt.plot([54,54],[0,trialdata_ses2_z.shape[0]],'-r')
        plt.ylim([0,trialdata_ses2_z.shape[0]])
        fig.savefig(os.path.join(resultsfolder,'ses2'+cca_str+constring+'K'+str(K)+whitenstring+regularizestring+'.png'))
        fig = plt.figure(figsize=(8*K,8))
        for k in np.arange(K):
            ax = plt.subplot(1,K,k+1)
            plt.imshow(As_ses2[k],cmap='jet'); 
            plt.clim([-0.2,0.2]); 
            plt.xticks(np.arange(len(areas_ses1)),areas_ses1,rotation=45,fontsize=10)
            plt.yticks(np.arange(len(areas_ses1)),areas_ses1,fontsize=10)
            plt.title('State'+str(k)); 
        fig.savefig(os.path.join(resultsfolder,'As_ses2'+cca_str+constring+'K'+str(K)+whitenstring+regularizestring+'.png'))


    mydict={'As_ses1':As_ses1,'trialdata_ses1':trialdata_ses1,
            'arhmm_ses1':arhmm_ses1,'arhmm_em_lls_ses1':arhmm_em_lls_ses1,
            'trialdata_ses1_z':trialdata_ses1_z,
            'areas_ses1':areas_ses1,'scores_mean_area':scores_mean_area}
    if not concatenate_sessions:
        mydict['As_ses2']=As_ses2; mydict['trialdata_ses2']=trialdata_ses2; mydict['arhmm_ses2']=arhmm_ses2;mydict['arhmm_em_lls_ses2']=arhmm_em_lls_ses2;
        mydict['trialdata_ses2_z']=trialdata_ses2_z; mydict['areas_ses2']=areas_ses2;
    savefile = os.path.join(resultsfolder,'arhmm'+cca_str+constring+'K'+str(K)+whitenstring+regularizestring+'.p')
    f = open(savefile, 'wb')
    pickle.dump(mydict, f)          # dump data to f
    f.close()   
    print('File saved at ',savefile)
