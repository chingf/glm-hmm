import os
import pickle
import csv
import h5py
import re
import numpy as np
from copy import deepcopy
from scipy.io import loadmat

class LearningSession(object):
    """
    Represents one task session.
    """
    
    def __init__(
            self, mouse, date, access_engram=False, load_Vc=True
            ):
        """
        Args
            mouse: String; the name of the mouse
            date: String; the date of the session
        Raises
            ValueError: if the inputs do not specify a valid filepath.
        """
        
        self.num_trials = 0
        self.num_bins = 0
        self.num_components = 0
        self.Vc = {}
        self.trialmarkers = {}
        self.spatialdisc = {}
        self.is_aud_trial = []
        self.mouse = mouse
        self.date = date
        if access_engram:
            self.datadir = "/home/chingf/engram/data/musall/learning/neural/"
            self.analysisdir = "/home/chingf/engram/analysis/musall/learning/"
        else:
            self.datadir = "/home/chingf/Code/Widefield/data/musall/learning/neural/"
            self.analysisdir = "/home/chingf/Code/Widefield/analysis/musall/learning/"
        self.dirpath = self.datadir + mouse + "/" + date + "/"
        if not os.path.isdir(self.dirpath):
            raise ValueError("Invalid path: " + self.dirpath)
        self._load_trialmarkers()
        self._load_spatialdisc()
        if load_Vc:
            self._load_Vc()
        else:
            self.num_trials = self.trialmarkers['ResponseSide'].size
        
    def get_trial_indices(self):
        """
        Extracts the indices corresponding relevant trial events.

        Args
            include_stim: boolean; whether or not to include stimulus
                presentation time periods
        Returns
            A (trials, 5) numpy array. For some trial, the 5-length array
            contains the indices for the following events:
            Value 1: Lnum_trialsevers in, inclusive
            Value 2: First stimulus, inclusive
            Value 3: Last stimulus, inclusive
            Value 4: Spouts in, inclusive
            Value 5: Spouts out, inclusive
        """

        trial_indices = []
        for trial in range(self.num_trials):
            t_i = self._get_delay_period_trial(trial)
            trial_indices.append(t_i)
        return np.array(trial_indices)

    def _load_Vc(self):
        """
        Loads the `Vc.mat` file and saves its data structures.
        """

        filepath = self.dirpath + "Vc.mat"
        matfile = h5py.File(filepath, 'r')
        for key in matfile:
            self.Vc[key] = np.array(matfile[key]).squeeze()
        self.num_trials, self.num_bins, self.num_components =\
            self.Vc['Vc'].shape

    def _load_trialmarkers(self):
        """
        Loads the `trialmarkers.mat` file and saves its data structures.
        """

        filepath = self.dirpath + "trialmarkers.mat"
        matfile = loadmat(filepath)
        for key in [key for key in matfile if not key.startswith("__")]:
            self.trialmarkers[key] = np.array(matfile[key]).squeeze()
        num_trials = self.trialmarkers['CorrectSide'].size
        for trial in range(num_trials):
            if self.trialmarkers['audStimL'][trial].size > 0 or\
                self.trialmarkers['audStimR'][trial].size > 0:
                self.is_aud_trial.append(True)
            else:
                self.is_aud_trial.append(False)

    def _load_spatialdisc(self):
        """
        Loads the relevant data structures from Spatial Disc.
        """

        all_files = os.listdir(self.dirpath)
        spatialdisc_file = [f for f in all_files if 'SpatialDisc' in f][0]
        filepath = self.dirpath + spatialdisc_file 
        matfile = loadmat(filepath)
        sessiondata = np.array(matfile['SessionData'])[0,0]
        self.spatialdisc['Unassisted'] = sessiondata['Assisted'].squeeze()

    def _get_delay_period_trial(self, trial_num):
        """
        Extracts the indices of relevant events in a given trial.

        Args
            trial_num: integer; the trial to process.
        Returns
            A length-5 array corresponding to the time bin boundaries of the
            relevant trial events: lever in, first stim, last stim, spouts in,
            spouts out.
        """
      
        trial_indices = []

        # Get Lever In
        lever_in = self.trialmarkers['stimGrab'][trial_num] - 1
        if lever_in < 0:
            trial_indices.append(0)
        else:
            trial_indices.append(lever_in)

        # Get Stim in and Stim out
        if self.is_aud_trial[trial_num]:
            l_stims = self.trialmarkers['audStimL'][trial_num].squeeze()
            r_stims = self.trialmarkers['audStimR'][trial_num].squeeze()
        else:
            l_stims = self.trialmarkers['tacStimL'][trial_num].squeeze()
            r_stims = self.trialmarkers['tacStimR'][trial_num].squeeze()
        all_stims = []
        if (l_stims.size > 0) and (l_stims.shape != ()):
            all_stims.extend([s - 1 for s in l_stims])
        if (r_stims.size > 0) and (r_stims.shape != ()):
            all_stims.extend([s - 1 for s in r_stims])
        if len(all_stims) == 0:
            trial_indices.append(np.nan)
            trial_indices.append(np.nan)
        else:
            trial_indices.append(min(all_stims))
            trial_indices.append(max(all_stims))

        # Get Spout In and Spout Out
        trial_indices.append(self.trialmarkers['spoutTime'][trial_num] - 1)
        spout_out_time = self.trialmarkers['spoutOutTime'][trial_num] - 1
        if np.isnan(spout_out_time) or spout_out_time > (self.num_bins - 2):
            trial_indices.append(
                min(trial_indices[-1] + 10, self.num_bins - 2)
                )
        else:
            trial_indices.append(spout_out_time)
        tt = np.array(trial_indices)
        return trial_indices 
