import os
import pickle
import numpy as np
from scipy.io import loadmat

class Session(object):
    """
    Represents one task session.
    """
    
    datadir = "/home/chingf/engram/data/musall/"
    #datadir = "/home/chingf/Code/LVM-Widefield/Data/10-Oct-2017/"
    analysisdir = "/home/chingf/engram/analysis/behavenet/musall/"
    dirpath = None
    task = None
    mouse = None
    date = None
    num_trials = 0
    num_bins = 0
    num_components = 0
    neural = {}
    trialmarkers = {}
    behavenet_latents = None
    
    def __init__(self, task, mouse, date):
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
        self.dirpath = self.datadir + task + "/" + mouse + "/" + date + "/"
        if not os.path.isdir(self.dirpath):
            raise ValueError("Invalid path.")
        self._load_neural()
        self._load_trialmarkers()
        self._load_behavioral_latents()
        
    def get_quiescent_activity(self):
        """
        Extracts the neural activity during the quiescent period of each trial.
        For each trial, we will look at the delay period, and extract the
        largest continuous portion of the delay period where there seems to be
        a period of inactivity. Inactivity is defined by looking at the
        behavioral latents and thresholding the total change in latents from
        one time step to the next.

        Returns
            A trials-length array, where each element of the array is a
            (bins x components) numpy array.
        """

        quiescent_activity = []
        neural_activity = self.neural['neural']
        for trial in range(self.num_trials):
            start_delay, end_delay = self.get_delay_period(trial)
            states = self._label_nonmovement(trial, start_delay, end_delay)
            
            # Find the longest consecutive non-movement chunk
            longest_segment_idxs = (0,0) # Stores the best sequence so far
            longest_segment_length = 0
            current_idx = 0 # Initialize the index to start at
            start_idx = 0 # The index that the current segment starts at
            prev_state = False # Initialize the previous state as movement 
            while current_idx < states.size:
                current_state = states[current_idx]
                # If we are at the end of the current segment
                if current_state == False and prev_state == True:
                    current_segment_length = current_idx - start_idx
                    if current_segment_length > longest_segment_length:
                        longest_segment_idxs = (start_idx, current_idx)
                        longest_segment_length = current_segment_length
                # If we have not yet found a nonmovement segment
                elif current_state == False and prev_state == False:
                    pass
                # If we have found the start of a new nonmovement segment
                elif current_state == True and prev_state == False:
                    start_idx = current_idx
                # If we are in the middle of a nonmovement segment
                else:
                    pass
                    
                prev_state = current_state
                current_idx += 1
            nonmovement_start = start_delay + longest_segment_idxs[0]
            nonmovement_end = start_delay + longest_segment_idxs[1]
            quiescent_activity.append(
                neural_activity[trial, nonmovement_start:nonmovement_end, :]
                )
        return quiescent_activity

    def get_delay_period(self, trial_num, include_stim=False):
        """
        Extracts the trial indices corresponding to the delay period. If we
        choose to ignore stimulus presentation times, then this captures the
        period between the second stimulus and 'spouts in' and the period
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

    def get_change_in_latents(self, trial, start_bin):
        """
        Returns the vector of the change in behavioral latents from timestep
        START_BIN to the next timestep at trial TRIAL.

        Returns
            A (behavelatent_dim,) numpy array
        """

        start_behavelatent = self.behave_latents[trial, start_bin, :]
        next_behavelatent = self.behave_latents[trial, start_bin + 1, :]
        return next_behavelatent - start_behavelatent

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

    def _load_behavioral_latents(self):
        """
        Loads the BehaveNet continuous behavioral latents.
        """

        sessionpath = self.analysisdir +\
            self.task + "/" + self.mouse + "/" + self.date + "/"
        latentpath = "ae/conv/16_latents/test_tube_data/test_pt/version_0/"
        pklpath = sessionpath + latentpath + "latents.pkl"
        with open(pklpath, "rb") as pkl:
            latentdata = pickle.load(pkl)
        behavenet_latents = np.array(latentdata['latents'])
        self.behavenet_latents = behavenet_latents

    def _label_nonmovement(self, trial, start_delay, end_delay):
        """
        For the given trial and delay period indices, the function will label
        each time bins as either movement (False) or non-movement(True).

        Returns
            A (end_delay - start_delay,) boolean numpy array
        """

        # First, see if the k=4 BehaveNet labeling is available
        # If so, we need only use state 1 as a nonmovement label
        sessionpath = self.analysisdir +\
            self.task + "/" + self.mouse + "/" + self.date + "/"
        labeldir = "arhmm/16_latents/04_states/0e+00_kappa/gaussian/" +\
            "test_tube_data/diff_init_grid_search/version_0/"
        statepkl = sessionpath + labeldir + "states.pkl"
        if os.path.isfile(statepkl):
            with open(statepkl, "rb") as pkl:
                states = np.array(pickle.load(pkl)['states'])
            states = states[trial, start_delay:end_delay]
            nonmovement_labels = (states == 1)
            return nonmovement_labels

        # If the BehaveNet labeling is unavailable, use some threshold on
        # the change in behavioral latents at each time step to determine
        # if movement has occurred.
        print("Using heuristic for nonmovement labeling")
        nonmovement_labels = []
        for i in range(start_delay, end_delay):
            change_in_latents = self.get_change_in_latents(trial, start_delay)
            change_in_latents = np.abs(change_in_latents) 
            if np.sum(change_in_latents) < 0.07:
                nonmovement_labels.append(True)
            else:
                nonmovement_labels.append(False)
        return np.array(nonmovement_labels)