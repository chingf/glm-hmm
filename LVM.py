import os
import pickle
import numpy as np
from abc import ABC
from scipy.io import loadmat

class LVM(ABC):
    
    def __init__(self):
        """
        Takes in whatever you need to take in for this thing
        """

    @abstractmethod
    def generate_latents(self):
        pass

    @abstractmethod
    def get_latent_trajectories(self):
        pass

class LVM_PCA(LVM):
    data = None # Assumed to be (samples, features) in size

    def __init__(self):
        """
        """

        pass

    def generate_latents(self):
        pass

    def get_latent_trajectories(self):

class LVM_SLDS(LVM):
    data = None # Assumed to be (samples, features) in size

    def __init__(self):
        """
        """

        pass

    def generate_latents(self):
        pass

    def get_latent_trajectories(self):

