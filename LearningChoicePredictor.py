import numpy as np
from scipy.io import loadmat
from LearningSession import *
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
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

    def __init__(self, session):
        self.session = session
        self.trial_choices = session.trialmarkers['ResponseSide']
        self.data = session.Vc['Vc']
        self.trial_indices = session.get_trial_indices()

class LRChoice(LearningPredictor):
    """
    Logistic regression predictor. Looks one frame into the future. Regularized
    by L2 norm.
    """

    def __init__(self, session):
        super(LRChoice, self).__init__(session)

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
        Does a grid search over each index of the trial for the given data.
        Returns a list of logistic regression models. 
        """

        scores = []
        models = []
        indices = []
        index_start = 0
        num_intervals = 4 
      
        for trial_index in range(self.trial_indices.shape[1] - 1):
            all_window_activity = [[] for _ in range(num_intervals)]
            for trial in range(self.session.num_trials):
                start_frame = self.trial_indices[trial,trial_index]
                end_frame = self.trial_indices[trial,trial_index + 1]
                frames = np.linspace(
                    start_frame, end_frame - 1, num_intervals
                    ).astype(int)
                for interval, frame in enumerate(frames):
                    all_window_activity[interval].append(
                        data[trial,frame:frame+2,:]
                        )
            for interval, window_activity in enumerate(all_window_activity):
                window_activity = np.array(window_activity)
                model = self._fit_window(window_activity)
                models.append(model)
                scores.append(np.max(np.mean(model.scores_[1], axis=0)))
                indices.append(
                    index_start + interval/(num_intervals*1.0)
                    )
            index_start += 1
        results = {"scores": scores, "models": models, "indices": indices}
        return results

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
        for trial in range(self.trial_choices.size):
            choice = self.trial_choices[trial]
            if np.isnan(choice):
                continue
            activity = window_data[trial,:,:]
            X.append(activity.flatten())
            y.append(int(choice-1))
        X = np.array(X)
        y = np.array(y)
        
        # Training the model with cross validation
        log_reg = LogisticRegressionCV(
            Cs=5, cv=5, scoring='accuracy', max_iter=500
            )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        log_reg.fit(X, y)
        return log_reg

class SVCChoice(LearningPredictor):
    """
    SVM classifier with polynomial kernel. Looks one frame into the future.
    """

    results = []

    def __init__(self, session):
        super(SVCChoice, self).__init__(session)

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
        Does a grid search over each index of the trial. Returns a list of
        dictionary of trained models, their scores, and the trial indices
        corresponding to each model.
        """

        scores = []
        models = []
        indices = []
        index_start = 0
        num_intervals = 4
      
        for trial_index in range(self.trial_indices.shape[1] - 1):
            all_window_activity = [[] for _ in range(num_intervals)]
            choices = []
            for trial in range(self.session.num_trials):
                start_frame = self.trial_indices[trial,trial_index]
                end_frame = self.trial_indices[trial,trial_index + 1]
                if np.sum(np.isnan(self.trial_indices[trial,:])) > 0:
                    continue
                if start_frame > (self.session.num_bins - 2):
                    continue
                frames = np.linspace(
                    start_frame, end_frame - 1, num_intervals
                    ).astype(int)
                for interval, frame in enumerate(frames):
                    all_window_activity[interval].append(
                        data[trial,frame:frame+2,:]
                        )
                    if data[trial, frame:frame+2,:].size != 400:
                        import pdb; pdb.set_trace()
                choices.append(self.trial_choices[trial])

            for interval, window_activity in enumerate(all_window_activity):
                window_activity = np.array(window_activity)
                choices = np.array(choices)
                score, model = self._fit_window(window_activity, choices)
                models.append(model)
                scores.append(np.mean(score))
                indices.append(
                    index_start + interval/(num_intervals*1.0)
                    )
            index_start += 1
        results = {"scores": scores, "models": models, "indices": indices}
        return results

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
            if activity.size == 0:
                continue
            if np.isnan(choice) or np.sum(np.isnan(activity)) > 0:
                continue
            X.append(activity.flatten())
            y.append(int(choice-1))
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.20
            )
        
        best_score = 0
        best_model = None
        
        for C in [2**i for i in range(-3,3)]:
            for gamma in [2**i for i in range(-4,4)]:
                for degree in [1,2]:
                    svclassifier = SVC(
                        kernel='poly', degree=degree, C=C, gamma=gamma
                        )
                    svclassifier.fit(X_train, y_train)
                    score = svclassifier.score(X_test, y_test)
                    
                    if best_score < score:
                        best_score = score
                        best_model = svclassifier
        return best_score, best_model

