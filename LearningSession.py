import os
import pickle
import csv
import numpy as np
from copy import deepcopy
from scipy.io import loadmat

class Session(object):
    """
    Represents one task session.
    """
    
    datadir = "/home/chingf/Code/Widefield/data/musall/"
    analysisdir = "/home/chingf/Code/Widefield/analysis/musall/"
    dirpath = None
    task = None
    mouse = None
    date = None
    num_trials = 0
    num_bins = 0
    num_components = 0
    neural = {}
    trialmarkers = {}
    
    def __init__(
            self, task, mouse, date, access_engram=False
            ):
        """
        Args
            task: String; "vistrained" or "audiotrained"
            mouse: String; the name of the mouse
            date: String; the date of the session
        Raises
            ValueError: if the inputs do not specify a valid filepath.
        """
        
        self.task = task
        self.mouse = mouse
        self.date = date
        if access_engram:
            self.datadir = "/home/chingf/engram/data/musall/"
            self.analysisdir = "/home/chingf/engram/analysis/behavenet/musall/"
        self.dirpath = self.datadir + task + "/" + mouse + "/" + date + "/"
        if not os.path.isdir(self.dirpath):
            raise ValueError("Invalid path: " + self.dirpath)
        self._load_neural()
        self._load_trialmarkers()
        
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

    def get_lever_grab_activity(self):
        """ Extracts the activity centered around the lever grab initializing the
        trial. This will always be at frame 54. Activity will be extracted from
        half a second before the lever grab (15 frames) to a second after the
        lever grab (30 frames)

        Returns
            A (trials, 15+30, components) numpy array.
        """

        levergrab_frame = 54
        start_frame = levergrab_frame - 0 
        end_frame = levergrab_frame + 30
        return self.neural['neural'][:,start_frame:end_frame,:]

    def _load_neural(self):
        """
        Loads the `neural.mat` file and saves its data structures.
        """

        filepath = self.dirpath + "neural.mat"
        matfile = loadmat(filepath)
        for key in [key for key in matfile if not key.startswith("__")]:
            self.neural[key] = np.array(matfile[key]).squeeze()
        self.num_trials, self.num_bins, self.num_components =\
            self.neural['neural'].shape

    def _load_trialmarkers(self):
        """
        Loads the `trialmarkers.mat` file and saves its data structures.
        """

        filepath = self.dirpath + "trialmarkers.mat"
        matfile = loadmat(filepath)
        for key in [key for key in matfile if not key.startswith("__")]:
            self.trialmarkers[key] = np.array(matfile[key]).squeeze()

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
        
        delay_start = self.trialmarkers['stimTime'][trial_num]
        if not include_stim:
            delay_start += 51
        delay_end = self.trialmarkers['spoutTime'][trial_num]
        return (delay_start, delay_end)
