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
    
    def __init__(
            self, task, mouse, date,
            load_behavenet=True, load_reconstructions=False, access_engram=False
            ):
        """
        Args
            task: String; "vistrained" or "audiotrained"
            mouse: String; the name of the mouse
            date: String; the date of the session
        Raises
            ValueError: if the inputs do not specify a valid filepath.
        """
        
        self.num_trials = 0
        self.num_bins = 0
        self.num_components = 0
        self.neural = {}
        self.trialmarkers = {}
        self.behavenet_latents = None
        self.reconstruction_model = None
        self.reconstruction_model_table = None
        self.reconstructions = None
        self.reconstructions_trials = None
        self.task = task
        self.mouse = mouse
        self.date = date
        if access_engram:
            self.datadir = "/home/chingf/engram/data/musall/"
            self.analysisdir = "/home/chingf/engram/analysis/behavenet/musall/"
        else:
            self.datadir = "/home/chingf/Code/Widefield/data/musall/"
            self.analysisdir = "/home/chingf/Code/Widefield/analysis/musall/"
        self.dirpath = self.datadir + task + "/" + mouse + "/" + date + "/"
        if not os.path.isdir(self.dirpath):
            raise ValueError("Invalid path: " + self.dirpath)
        self._load_neural()
        self._load_trialmarkers()
        if load_behavenet:
            self._load_behavioral_latents()
        if load_reconstructions:
            self._load_reconstructions()
        
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

    def _load_behavioral_latents(self):
        """
        Loads the BehaveNet continuous behavioral latents.
        """

        sessionpath = self.analysisdir +\
            self.task + "/" + self.mouse + "/" + self.date + "/"
        latentpath = "ae/conv/32_latents/test/version_0/"
        pklpath = sessionpath + latentpath + "latents.pkl"
        with open(pklpath, "rb") as pkl:
            latentdata = pickle.load(pkl)
        behavenet_latents = np.array(latentdata['latents'])
        self.behavenet_latents = behavenet_latents

    def _load_reconstructions(self):
        """
        Loads the best BehaveNet reconstruction.
        """

        def generate_models_lookuptable():
            """
            Helper function to make a models lookuptable for reconstructions.
            """
            
            l2_dict = {}
            nl_dict = {'8': deepcopy(l2_dict), '16': deepcopy(l2_dict)}
            lr_dict = {
                '0.0001': deepcopy(nl_dict), '0.001': deepcopy(nl_dict),
                '0.01': deepcopy(nl_dict)
                }
            hl_dict = {
                '0': deepcopy(lr_dict), '1': deepcopy(lr_dict),
                '2': deepcopy(lr_dict), '3': deepcopy(lr_dict)
                }
            return hl_dict

        ffdatadir = "/home/chingf/Code/Widefield/analysis/musall/vistrained/"\
            + self.mouse + "/" + self.date + "/ae-neural/16_latents/ff/all/test/"
        models = []
        models_lookuptable = generate_models_lookuptable()
        min_test_loss = 1
        best_model = None
        for version in os.listdir(ffdatadir):
            versiondir = ffdatadir + version + '/'
            metatag_file = 'meta_tags.pkl'
            metrics_file = 'metrics.csv'
            val_loss = []
            epoch = []
            test_loss = []
            
            with open(versiondir + metatag_file, 'rb') as f:
                metatag = pickle.load(f) # Give hyperparameters in a dictionary
            with open(versiondir + metrics_file) as f:
                csvreader = csv.DictReader(f)
                for row in csvreader:
                    if row['val_loss'] != '':
                        val_loss.append(float(row['val_loss']))
                        epoch.append(int(row['epoch']))
                    if row['test_loss'] != '':
                        test_loss.append(float(row['test_loss']))
                        
            # Save model to look at later
            model = {}
            n_hid_layers = str(metatag['n_hid_layers'])
            learning_rate = str(metatag['learning_rate'])
            n_lags = str(metatag['n_lags'])
            l2_reg = str(metatag['l2_reg'])
            model['n_hid_layers'] = n_hid_layers
            model['learning_rate'] = learning_rate
            model['n_lags'] = n_lags
            model['l2_reg'] = l2_reg
            model['epoch'] = epoch
            model['val_loss'] = np.array(val_loss)
            model['test_loss'] = np.array(test_loss)
            model['version'] = version
            models.append(model)
            models_lookuptable[n_hid_layers][learning_rate][n_lags][l2_reg] = model
            
            # See if this is a better model than the last found
            mean_test_loss = np.mean(test_loss)
            if mean_test_loss < min_test_loss:
                min_test_loss = mean_test_loss
                best_model = model

        # Save the results of looking over all the NN models 
        self.reconstruction_model_table = models_lookuptable
        self.reconstruction_model = best_model

        # Load the reconstruction neural activity of the best model
        bn_version = best_model['version']
        predictions_file = ffdatadir + bn_version + "/" + "predictions.pkl"
        with open(predictions_file, "rb") as f:
            predictions_data = pickle.load(f)
        predictions = np.nan_to_num(
            predictions_data['predictions']
            )
        self.reconstructions = self._undo_zscore(self.neural['neural'], predictions)
        self.reconstructions_trials = predictions_data['trials']

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

    def _undo_zscore(self, neural, predictions):
        """
        Assumes data is trial x time x components
        
        Args:
            neural: original neural activity
            predictions: predicted z-scored activity
        Returns:
            predictions with the z-scoring undone
        """
    
        predictions = predictions.copy()
        predictions *= np.std(neural, axis=(0,1))
        predictions += np.mean(neural, axis=(0,1))
        return predictions
