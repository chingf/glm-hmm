import numpy as np
from scipy.io import loadmat
from Session import *
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class Predictor():
    """
    Super class for choice predictors. Predictors will predict choice over the
    course of a 120-frame segment of each trial. 

    Attributes
        session: The Session object of the session of interest.
        datatype: a String. Options are ['full', 'reconstructions', 'residuals',
            'behavenet'].
        data: A (trial x bins x components) array, constructed from
            datatype
        reg_indices: grouping of component indices into their respective brain
            regions.
        reg_names: list of brain regions
    """

    session = None
    datatype = None
    data = None
    trial_choices = None
    results = {} 
    loo_results = {}

    def __init__(self, session, datatype, truncate=True, shuffle=False):
        self.session = session
        self.reg_indices = session.neural['reg_indxs_consolidate'].item()
        self.reg_names = session.neural['reg_indxs_consolidate'].dtype.names
        self.trial_choices = session.trialmarkers['ResponseSide']
        self.datatype = datatype
        if datatype == 'full':
            self.data = session.neural['neural']
        elif datatype == 'reconstructions':
            self.data = session.reconstructions
        elif datatype == "residuals":
            residuals = session.neural['neural'] - session.reconstructions
            self.data = residuals
        elif datatype == "behavenet":
            behavenet_nogap = []
            choices_nogap = []
            for trial in range(self.session.num_trials):
                behavenet_activity = session.behavenet_latents[trial,:,:]
                if np.sum(np.isnan(behavenet_activity)) > 0:
                    continue
                behavenet_nogap.append(behavenet_activity)
                choices_nogap.append(self.trial_choices[trial])
            self.data = np.array(behavenet_nogap)
            self.trial_choices = np.array(choices_nogap)
            if shuffle:
                np.random.shuffle(self.trial_choices)
        if truncate:
            self._truncate_data()

    def _truncate_data(self):
        """
        Aligns trials by the stimulus onset. Then cuts trials to capture
        30 frames before the stimulus onset and 90 frames after the stimulus
        onset, for a total of 120 frames.
        """

        delay_period_indices = self.session.get_delay_period(include_stim=True)
        excerpt_indices = []
        for trial in range(delay_period_indices.shape[0]):
            start = delay_period_indices[trial,:][0] - 30
            end = delay_period_indices[trial,:][0] + 90
            excerpt_indices.append([start, end])

        data_excerpt = []
        for trial in range(self.data.shape[0]):
            activity = self.data[trial,:,:]
            indices = excerpt_indices[trial]
            data_excerpt.append(activity[indices[0]:indices[1],:])
        self.data = np.array(data_excerpt)

class LRChoice(Predictor):
    """
    Logistic regression predictor. Looks one frame into the future. Regularized
    by L2 norm.
    """

    def __init__(self, session, datatype):
        super(LRChoice, self).__init__(session, datatype)

    def fit_all(self):
        """
        Does a grid search over each index of the trial, using all the loaded
        data. Returns a dictionary of scores and fitted models.
        """

        results = self._fit_data(self.data)
        self.results = results
        return results

    def fit_loo(self):
        """
        Does a grid search over each index of the trial, in a leave-one-out
        manner over each brain region. Returns a dictionary of
        the dropped-out brain region and the corresponding dictionary of
        scores and fitted models.
        """

        if self.datatype == 'behavenet':
            raise Exception(
                "Neural region leave-one-out not applicable to BehaveNet data."
                )
            return None

        loo_results = {}
        for idx, reg_name in enumerate(self.reg_names):
            components = reg_indxs[idx].squeeze() - 1
            loo_indices = [
                i for i in range(self.session.num_components) if i not in components
                ]
            loo_data = self.data[:,:,loo_indices]
            results = self._fit_data(loo_data)
            loo_results[reg_name] = loo_results
        self.loo_results = loo_results
        return loo_results

    def _fit_data(self, data):
        """
        Does a grid search over each index of the trial for the given data.
        Returns a list of logistic regression models. 
        """

        window_length = 2
        scores = []
        models = []
        
        start_idxs = range(0, 120, 2)
        for start_idx in start_idxs:
            log_reg = self._fit_window(
                start_idx, window_length, data,
                )
            models.append(log_reg)
            scores.append(np.max(np.mean(log_reg.scores_[1], axis=0)))
        results = {"scores": scores, "models": models}
        return results

    def _fit_window(self, start_idx, window_length, data):
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
            activity = data[trial,start_idx:start_idx+window_length,:]
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

class SVCChoice(Predictor):
    """
    SVM classifier with polynomial kernel. Looks one frame into the future.
    """

    results = []

    def __init__(self, session, datatype):
        super(SVCChoice, self).__init__(session, datatype)

    def fit_all(self):
        """
        Does a grid search over each index of the trial, using all the loaded
        data. Returns a dictionary of scores and fitted models.
        """

        results = self._fit_data(self.data)
        self.results = results
        return results

    def fit_loo(self):
        """
        Does a grid search over each index of the trial, in a leave-one-out
        manner over each brain region. Returns a dictionary of
        the dropped-out brain region and the corresponding dictionary of
        scores and fitted models.
        """

        if self.datatype == 'behavenet':
            raise Exception(
                "Neural region leave-one-out not applicable to BehaveNet data."
                )
            return None

        loo_results = {}
        for idx, reg_name in enumerate(self.reg_names):
            components = reg_indxs[idx].squeeze() - 1
            loo_indices = [
                i for i in range(self.session.num_components) if i not in components
                ]
            loo_data = self.data[:,:,loo_indices]
            results = self._fit_data(loo_data)
            loo_results[reg_name] = loo_results
        self.loo_results = loo_results
        return loo_results

    def _fit_data(self, data):
        """
        Does a grid search over each index of the trial. Returns a list of
        logistic regression models.
        """

        window_length = 2
        scores = []
        models = []
        
        start_idxs = np.arange(0, data.shape[1], 2)
        for start_idx in start_idxs:
            score, model = self._fit_window(start_idx, window_length, data)
            scores.append(np.mean(score))
            models.append(model)
        results = {"scores": scores, "models": models}
        return results 

    def _fit_window(self, start_idx, window_length, data):
        """
        Fits SVC over the data from the start_idx for a framesize of window_length.
        """

        X = []
        y = []
        # Extracting training and test data
        for trial in range(self.trial_choices.size):
            choice = self.trial_choices[trial]
            if np.isnan(choice):
                continue
            activity = data[trial,start_idx:start_idx+window_length,:]
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

class LDAChoice(Predictor):
    """
    LDA classifier. Looks one frame into the future.
    """

    results = []

    def __init__(self, session, datatype):
        super(LDAChoice, self).__init__(session, datatype)

    def fit_all(self):
        """
        Does a grid search over each index of the trial, using all the loaded
        data. Returns a dictionary of scores and fitted models.
        """

        results = self._fit_data(self.data)
        self.results = results
        return results

    def fit_loo(self):
        """
        Does a grid search over each index of the trial, in a leave-one-out
        manner over each brain region. Returns a dictionary of
        the dropped-out brain region and the corresponding dictionary of
        scores and fitted models.
        """

        if self.datatype == 'behavenet':
            raise Exception(
                "Neural region leave-one-out not applicable to BehaveNet data."
                )
            return None

        loo_results = {}
        for idx, reg_name in enumerate(self.reg_names):
            components = reg_indxs[idx].squeeze() - 1
            loo_indices = [
                i for i in range(self.session.num_components) if i not in components
                ]
            loo_data = self.data[:,:,loo_indices]
            results = self._fit_data(loo_data)
            loo_results[reg_name] = loo_results
        self.loo_results = loo_results
        return loo_results

    def _fit_data(self, data):
        """
        Does a grid search over each index of the trial. Returns a list of
        LDA models.
        """

        window_length = 2 
        scores = []
        models = []
        
        start_idxs = np.arange(0, data.shape[1], 2)
        for start_idx in start_idxs:
            score, model = self._fit_window(start_idx, window_length, data)
            scores.append(np.mean(score))
            models.append(model)
        results = {"scores": scores, "models": models}
        return results 

    def _fit_window(self, start_idx, window_length, data):
        """
        Fits LDA over the data from the start_idx for a framesize of window_length.
        """

        X = []
        y = []
        # Extracting training and test data
        for trial in range(self.trial_choices.size):
            choice = self.trial_choices[trial]
            activity = data[trial,start_idx:start_idx+window_length,:]
            if np.isnan(choice) or np.sum(np.isnan(activity)) > 0:
                continue
            activity = activity.flatten()
            X.append(activity)
            y.append(int(choice-1))
        X = np.array(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.20
            )
        
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = np.sum(predictions == y_test)/(1.0*y_test.size) 
        return score, model

