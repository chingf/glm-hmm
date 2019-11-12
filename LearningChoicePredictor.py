import numpy as np
from scipy.io import loadmat
from LearningSession import *
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class LearningPredictor():
    """
    Super class for choice predictors. Predictors will predict choice from lever
    in to spouts out. Since each trial has variable delay periods, the time
    periods will be normalized. From lever in to the first stimulus will take
    time 0-1. From first stimulus to last stimulus will be time 1-2. From last
    stimulus to spout in will be time 2-3. From spout in to spout out will be
    time 3-4.

    Attributes
        session: The Session object of the session of interest.
    """

    session = None
    datatype = None
    data = None
    trial_choices = None
    results = {} 
    loo_results = {}

    def __init__(self, session, shuffle=False):
        self.session = session
        self.trial_choices = session.trialmarkers['ResponseSide']
        if shuffle:
            np.random.shuffle(self.trial_choices)
        self.data = session.Vc['Vc']
        self.trial_indices = session.get_trial_indices()

    def fit_all(self):
        """
        Does a grid search over each index of the trial, using all the loaded
        data. Returns a dictionary of scores and fitted models.
        """

        results = self._fit_data(self.data)
        self.results = results
        return results

    def _fit_data(self, data):
        """
        Does a grid search over relevant trial events. Returns a list of
        dictionary of trained models, their scores, and the trial indices
        corresponding to each model. Model will be fit in three different trial
        events: pre-stim, during stim, and post-stim. Specifically, we look at
        10 frames before stims in, 20 frames after stims in, and 10 frames
        after stims out.

        Trials are thrown of if there are nans in the activity or if the
        trial indices found are too close to the end of the trial.
        """

        scores = []
        models = []
        all_test_indices = []
        all_correct_test_indices = []

        # Fit choice decoder for EVENT_LENGTH frames of the pre-stim duration.
        event_length = 10
        for frames_pre_stim in range(event_length, 0, -1):
            window_activity = []
            choices = []
            for trial in range(self.session.num_trials):
                stim_on_frame = self.trial_indices[trial, 1]
                start_frame = stim_on_frame - frames_pre_stim
                if (np.sum(np.isnan(self.trial_indices[trial,:])) > 0) or\
                    (start_frame > (self.session.num_bins - 2)) or\
                    (np.isnan(self.trial_choices[trial])):
                    continue
                start_frame = int(start_frame)
                window_activity.append(
                    data[trial, start_frame:start_frame+2, :]
                    )
                choices.append(self.trial_choices[trial])
            choices = np.array(choices)
            score, model, test_indices, correct_test_indices = \
                self._fit_window(window_activity, choices)
            scores.append(score)
            models.append(model)
            all_test_indices.append(test_indices)
            all_correct_test_indices.append(correct_test_indices)

        # Fit choice decoder for EVENT_LENGTH frames of the stim duration.
        event_length = 20
        for frames_into_stim in range(event_length):
            window_activity = []
            choices = []
            for trial in range(self.session.num_trials):
                stim_on_frame = self.trial_indices[trial, 1]
                start_frame = stim_on_frame + frames_into_stim 
                if (np.sum(np.isnan(self.trial_indices[trial,:])) > 0) or\
                    (start_frame > (self.session.num_bins - 2)) or\
                    (np.isnan(self.trial_choices[trial])):
                    continue
                start_frame = int(start_frame)
                window_activity.append(
                    data[trial, start_frame:start_frame+2, :]
                    )
                choices.append(self.trial_choices[trial])
            choices = np.array(choices)
            score, model, test_indices, correct_test_indices = \
                self._fit_window(window_activity, choices)
            scores.append(score)
            models.append(model)
            all_test_indices.append(test_indices)
            all_correct_test_indices.append(correct_test_indices)

        # Fit choice decoder for EVENT_LENGTH frames post-stim.
        event_length = 10
        for frames_post_stim in range(event_length):
            window_activity = []
            choices = []
            for trial in range(self.session.num_trials):
                stim_off_frame = self.trial_indices[trial, 2]
                start_frame = stim_off_frame + frames_post_stim 
                if (np.sum(np.isnan(self.trial_indices[trial,:])) > 0) or\
                    (start_frame > (self.session.num_bins - 2)) or\
                    (np.isnan(self.trial_choices[trial])):
                    continue
                start_frame = int(start_frame)
                window_activity.append(
                    data[trial, start_frame:start_frame+2, :]
                    )
                choices.append(self.trial_choices[trial])
            choices = np.array(choices)
            score, model, test_indices, correct_test_indices = \
                self._fit_window(window_activity, choices)
            scores.append(score)
            models.append(model)
            all_test_indices.append(test_indices)
            all_correct_test_indices.append(correct_test_indices)

        results = {
            "scores": scores, "models": models,
            "test_indices": all_test_indices,
            "correct_test_indices": all_correct_test_indices
            }
        return results

class LRChoice(LearningPredictor):
    """
    Logistic regression predictor. Looks one frame into the future. Regularized
    by L2 norm.
    """

    def __init__(self, session):
        super(LRChoice, self).__init__(session)

    def _fit_window(self, window_data):
        """
        Fits a L2-regularized logistic regression model, predicting
        left/right licking choice.
        
        Args
            start_idx: index in delay period to start extracting a window
                of activity.
            window_length: size of the window of activity to extract
        """
        
        X = []
        y = []
        # Extracting training and test data
        assert(len(window_data) == choices.size)
        for trial in range(choices.size):
            choice = self.trial_choices[trial]
            activity = window_data[trial].flatten()
            if np.isnan(choice) or np.sum(np.isnan(activity)) > 0:
                continue
            X.append(activity.flatten())
            y.append(int(choice-1))
        X = np.array(X)
        y = np.array(y)
        indices = np.arange(y.size)
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            train_test_split(
                X, y, indices, test_size = 0.20, stratify=y
                )
        
        # Training the model with cross validation
        log_reg = LogisticRegressionCV(
            Cs=5, cv=5, scoring='accuracy', max_iter=500
            )
        log_reg.fit(X_train, y)
        correct_test_indices = (y_test == log_reg.predict(X_test))
        test_score = np.sum(correct_test_indices)/np.size(y_test)
        return test_score, log_reg, test_indices, correct_test_indices

class SVCChoice(LearningPredictor):
    """
    SVM classifier with polynomial kernel. Looks one frame into the future.
    """

    results = []

    def __init__(self, session):
        super(SVCChoice, self).__init__(session)

    def get_trial_index_map(self):
        """
        Returns a map of how SVC data indices correspond to session trial
        indices. Specifically, for N SVC data trials, this function returns a
        size (N,) numpy array. The value at the ith index indicates the session
        trial index corresponding to the ith SVC data index.
        """

        num_intervals = 4
        all_index_maps = []
        for trial_index in range(self.trial_indices.shape[1] - 1):
            # Collect usable trials for this particular trial event.
            # Trial will be thrown out if there are nans in the activity or if
            # the trial event starts at a frame too close to the end of the
            # trial.
            event_index_map = [[] for _ in range(num_intervals)]
            for trial in range(self.session.num_trials):
                start_frame = self.trial_indices[trial,trial_index]
                end_frame = self.trial_indices[trial,trial_index + 1]
                if np.sum(np.isnan(self.trial_indices[trial,:])) > 0:
                    continue
                elif start_frame > (self.session.num_bins - 2):
                    continue
                elif np.isnan(self.trial_choices[trial]):
                    continue
                frames = np.linspace(
                     start_frame, end_frame - 1, num_intervals
                     ).astype(int)
                for interval, frame in enumerate(frames):
                    activity = self.data[trial, frame:frame+2, :]
                    if np.sum(np.isnan(activity)) > 0:
                        continue
                    event_index_map[interval].append(trial)
            event_index_map = [np.array(m) for m in event_index_map]
            all_index_maps.extend(event_index_map)
        return all_index_maps

    def _fit_window(self, window_data, choices):
        """
        Fits SVC over the data given in window_data 
        """

        X = []
        y = []
        # Extracting training and test data
        assert(len(window_data) == choices.size)
        for trial in range(choices.size):
            choice = self.trial_choices[trial]
            activity = window_data[trial].flatten()
            if np.isnan(choice) or np.sum(np.isnan(activity)) > 0:
                continue
            X.append(activity.flatten())
            y.append(int(choice-1))
        X = np.array(X)
        y = np.array(y)
        indices = np.arange(y.size)
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            train_test_split(
                X, y, indices, test_size = 0.20, stratify=y
                )
        
        best_score = 0
        best_model = None
        best_correct_test_indices = []
        
        for C in [2**i for i in range(-3,3)]:
            for gamma in [2**i for i in range(-4,4)]:
                for degree in [1,2]:
                    svclassifier = SVC(
                        kernel='poly', degree=degree, C=C, gamma=gamma
                        )
                    svclassifier.fit(X_train, y_train)
                    score = svclassifier.score(X_test, y_test)
                    correct_test_indices = \
                        svclassifier.predict(X_test) == y_test
                    
                    if best_score < score:
                        best_score = score
                        best_model = svclassifier
                        best_correct_test_indices = correct_test_indices
        return best_score, best_model, test_indices, best_correct_test_indices

