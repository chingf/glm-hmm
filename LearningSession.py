import os
import pickle
import csv
import h5py
import re
import numpy as np
from copy import deepcopy
from scipy.io import loadmat

class Session(object):
    """
    Represents one task session.
    """
    
    datadir = "/home/chingf/Code/Widefield/data/musall/learning/neural"
    analysisdir = "/home/chingf/Code/Widefield/analysis/musall/learning/"
    dirpath = None
    task = None
    mouse = None
    date = None
    num_trials = 0
    num_bins = 0
    num_components = 0
    Vc = {}
    trialmarkers = {}
    spatialdisc = {}
    
    def __init__(
            self, task, mouse, date, access_engram=False
            ):
        """
        Args
            mouse: String; the name of the mouse
            date: String; the date of the session
        Raises
            ValueError: if the inputs do not specify a valid filepath.
        """
        
        self.mouse = mouse
        self.date = date
        if access_engram:
            self.datadir = "/home/chingf/engram/data/musall/learning/neural"
            self.analysisdir = "/home/chingf/engram/analysis/musall/learning"
        self.dirpath = self.datadir + task + "/" + mouse + "/" + date + "/"
        if not os.path.isdir(self.dirpath):
            raise ValueError("Invalid path: " + self.dirpath)
        self._load_Vc()
        self._load_trialmarkers()
        self._load_spatialdisc()
        
    def get_delay_period(self, include_stim=False):
        """
        Extracts the indices corresponding to the delay period over all trials.
        See _get_delay_period_trial for more details on arguments and return.

        Args
            include_stim: boolean; whether or not to include stimulus
                presentation time periods
        Returns
            A (trials, 2) numpy array. Each entry is the time bin boundaries of
            the delay period (first value inclusive, second value exclusive). 
        """

        delay_periods = []
        for trial in range(self.num_trials):
            delay_start, delay_end = self._get_delay_period_trial(
                trial, include_stim
                )
            delay_periods.append([delay_start, delay_end])
        return np.array(delay_periods)

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

    def _load_spatialdisc(self):
        """
        Loads the relevant data structures from Spatial Disc.
        """

        pattern = '(\d{2})-(\w{3})-(\d{4})'
        date_pattern = re.search(pattern, self.date)
        day = date_pattern.group(1)
        month = date_pattern.group(2)
        year = date_pattern.group(2)
        filename = self.mouse + '_SpatialDisc_' + month + day +\
            '_' + year + '_Session2.mat'
        filepath = self.dirpath + filename
        matfile = loadmat(filepath)
        sessiondata = np.array(matfile['SessionData'])[0,0]
        self.spatialdisc['Unassisted'] = sessiondata['Assisted']

    def _get_delay_period_trial(self, trial_num, include_stim=False):
        """
        Extracts the indices corresponding to the delay period in a given trial.
        If we stimulus presentation time isn't included, then this captures the
        period between the second stimulus and 'spouts in'; the period
        between the first and second stimulus is ignored.

        Args
            trial_num: integer; the trial to process.
            include_stim: boolean; whether or not to include stimulus
                presentation time periods
        Returns
            A tuple corresponding to the time bin boundaries of the delay period.
            The first value is inclusive, the second value is exclusive.
        """
       
        if self.trialmarkers['CorrectSide'] == 1: # Left
            if self.trialmarkers['audStimL'][0,trial_num].size > 0: # Auditory 
                stims = self.trialmarkers['audStimL'][0,trial_num]
            else: # Tactile
                stims = self.trialmarkers['tacStimL'][0,trial_num]
        elif self.trialmarkers['CorrectSide'] == 2: # Right
            if self.trialmarkers['audStimR'][0,trial_num].size > 0: # Auditory 
                stims = self.trialmarkers['audStimR'][0,trial_num]
            else: # Tactile
                stims = self.trialmarkers['tacStimR'][0,trial_num]
        if include_stim:
            delay_start = np.min(stims) - 1
        else:
            delay_start = np.max(stims) - 1
        delay_end = self.trialmarkers['spoutTime'][0,trial_num]
        return (delay_start, delay_end)
