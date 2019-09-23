import os
import pickle
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

class LVM_PCA(object):
    data = None # Assumed to be (trials, bins, components) in size
    pca = None 

    def __init__(self, data):
        """
        Args
            Data: a (samples x features) numpy array
        """
        self.data = data
        self.pca = None 

    def generate_latents(self):
        """
        Gets the principal components of the data.

        Returns
            a (components x components) numpy array
        """

        data = self.data.reshape((-1, self.data.shape[2]))
        self.pca = PCA()
        self.pca.fit(data)
        return self.pca.components_ 

    def get_latent_projections(self, X):
        """
        Projects the given data onto the principal components

        Args
            X: a (n_samples, components) numpy array 
        Returns
            A (n_samples, components) numpy array
        Raises
            ValueError: If PCA has not been fit yet to the data
        """

        if self.pca is None:
            raise ValueError("Please generate the latents first.")
        projection = self.pca.transform(X)
        return projection
        
class LVM_SLDS(object):
    data = None # Assumed to be (samples, features) in size

    def __init__(self):
        """
        """

        pass

    def generate_latents(self):
        pass

    def get_latent_trajectories(self):
        pass

