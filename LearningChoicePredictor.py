import numpy as np
from scipy.io import loadmat
from LearningSession import *
from sklearn.decomposition import PCA
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

    def __init__(self, session, reduce_dim=None, mode=None, shuffle=False):
        self.session = session
        self.reg_indices = session.neural['reg_indxs'].item()
        self.reg_names = session.neural['reg_indxs'].dtype.names
        self.trial_choices = session.trialmarkers['ResponseSide']
        self.to_reduce = False
        if not reduce_dim is None:
            desired_regions = [
                'MOB', 'MOs1', 'SSp_bfd1', 'SSp_m1', 'SSp_n1', 'SSs1'
                #"FRP1", "MOp", "MOs", "SSp", "SSs1", "PL1", "MOB"
                ]
            self.desired_regions = desired_regions
            self.pc_var_threshold = reduce_dim
            self.to_reduce = True
        if shuffle:
            np.random.shuffle(self.trial_choices)
        self.data = session.neural['neural']
        self.trial_indices = session.get_trial_indices()
        if mode == "LOO":
            if not reduce_dim is None:
                raise ValueError("Cannot do LOO decoding and reduce dim.")
            self.loo = True
            self.loi = False
        elif mode == "LOI":
            if not reduce_dim is None:
                raise ValueError("Cannot do LOO decoding and reduce dim.")
            self.loo = False
            self.loi = True
        else:
            self.loo = False
            self.loi = False

    def fit_all(self):
        """
        Does a grid search over each index of the trial, using all the loaded
        data. Returns a dictionary of dictionaries over scores and fitted models.
        """

        if self.loo:
            results = {}
            for idx, reg_name in enumerate(self.reg_names):
                components = self.reg_indices[idx].squeeze() - 1
                num_components = self.session.num_components
                loo_indices = [
                    i for i in range(num_components) if i not in components
                    ]
                loo_data = self.data[:,:,loo_indices]
                loo_result = self._fit_data(loo_data)
                results[reg_name] = loo_result
        elif self.loi:
            results = {}
            reg_names_rl = [r[:-2] for r in self.reg_names]
            reg_names_rl = np.unique(reg_names_rl)
            for reg_name in reg_names_rl:
                components = []
                reg_name_r = reg_name + "_R"
                reg_name_l = reg_name + "_L"
                if reg_name_r in self.reg_names:
                    r_idx = self.reg_names.index(reg_name_r)
                    r_components = self.reg_indices[r_idx].squeeze() - 1
                    components.append(r_components)
                if reg_name_l in self.reg_names:
                    l_idx = self.reg_names.index(reg_name_l)
                    l_components = self.reg_indices[l_idx].squeeze() - 1
                    components.append(l_components)
                components = np.hstack(components)
                num_components = self.session.num_components
                loi_indices = [
                    i for i in range(num_components) if i in components
                    ]
                loi_data = self.data[:,:,loi_indices]
                loi_result = self._fit_data(loi_data)
                results[reg_name] = loi_result
        else:
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
        test_indices = []
        trial_labels = []
        predict_prob = []

        # Fit choice decoder for EVENT_LENGTH frames of the pre-stim duration.
        event_length = 10
        event_activity = []
        choices = []
        event_trial_labels = []
        for trial in range(self.session.num_trials):
            if (np.sum(np.isnan(self.trial_indices[trial,:])) > 0) or\
                (np.isnan(self.trial_choices[trial])):
                continue
            stim_on_frame = int(self.trial_indices[trial, 1])
            start_frame = int(stim_on_frame - event_length)
            trial_activity = data[trial, start_frame:stim_on_frame+1, :]
            trial_choice = self.trial_choices[trial]
            if np.isnan(trial_choice) or np.sum(np.isnan(trial_activity)) > 0:
                continue
            event_activity.append(trial_activity)
            choices.append(trial_choice)
            event_trial_labels.append(trial)
        choices = np.array(choices) - 1
        event_trial_labels = np.array(event_trial_labels)
        if self.to_reduce:
            event_activity = self._reduce_dim(event_activity)
        else:
            event_activity = np.array(event_activity)
        score, model, event_test_indices, event_predict_prob = \
                self._fit_window(event_activity, choices, event_trial_labels)
        scores.extend(score)
        models.extend(model)
        test_indices.append(event_test_indices)
        trial_labels.append(event_trial_labels)
        predict_prob.extend(event_predict_prob)

        # Fit choice decoder for EVENT_LENGTH frames of the stim duration.
        event_length = 20
        event_activity = []
        choices = []
        event_trial_labels = []
        for trial in range(self.session.num_trials):
            if (np.sum(np.isnan(self.trial_indices[trial,:])) > 0) or\
                (np.isnan(self.trial_choices[trial])):
                continue
            stim_on_frame = int(self.trial_indices[trial, 1])
            end_frame = int(stim_on_frame + event_length)
            trial_activity = data[trial, stim_on_frame:end_frame+1, :]
            trial_choice = self.trial_choices[trial]
            if np.isnan(trial_choice) or np.sum(np.isnan(trial_activity)) > 0:
                continue
            event_activity.append(trial_activity)
            choices.append(trial_choice)
            event_trial_labels.append(trial)
        choices = np.array(choices) - 1
        event_trial_labels = np.array(event_trial_labels)
        if self.to_reduce:
            event_activity = self._reduce_dim(event_activity)
        else:
            event_activity = np.array(event_activity)
        score, model, event_test_indices, event_predict_prob = \
                self._fit_window(event_activity, choices, event_trial_labels)
        scores.extend(score)
        models.extend(model)
        test_indices.append(event_test_indices)
        trial_labels.append(event_trial_labels)
        predict_prob.extend(event_predict_prob)

        # Fit choice decoder for EVENT_LENGTH frames post-stim.
        event_length = 10
        event_activity = []
        choices = []
        event_trial_labels = []
        for trial in range(self.session.num_trials):
            if (np.sum(np.isnan(self.trial_indices[trial,:])) > 0) or\
                (np.isnan(self.trial_choices[trial])):
                continue
            stim_off_frame = int(self.trial_indices[trial, 2])
            end_frame = int(stim_off_frame + event_length)
            trial_activity = data[trial, stim_off_frame:end_frame+1, :]
            trial_choice = self.trial_choices[trial]
            if np.isnan(trial_choice) or np.sum(np.isnan(trial_activity)) > 0:
                continue
            event_activity.append(trial_activity)
            choices.append(trial_choice)
            event_trial_labels.append(trial)
        choices = np.array(choices) - 1
        event_trial_labels = np.array(event_trial_labels)
        if self.to_reduce:
            event_activity = self._reduce_dim(event_activity)
        else:
            event_activity = np.array(event_activity)
        score, model, event_test_indices, event_predict_prob = \
                self._fit_window(event_activity, choices, event_trial_labels)
        scores.extend(score)
        models.extend(model)
        test_indices.append(event_test_indices)
        trial_labels.append(event_trial_labels)
        predict_prob.extend(event_predict_prob)

        results = {
            "scores": scores, "models": models,
            "test_indices": test_indices,
            "trial_labels": trial_labels,
            "predic_prob": predict_prob
            }
        return results

    def _reduce_dim(self, window_activity):
        """
        Reduces data by running PCA on the desired brain regions. Activity from
        other brain regions are dropped.
        """

        window_activity = np.array(window_activity)
        num_trials, num_bins, _ = window_activity.shape
        reg_names = self.reg_names
        reg_indices = self.reg_indices

        # Loop through desired regions and run PCA trial-by-trial
        pc_data = []
        for idx, reg_rl in enumerate(reg_names):
            # Extract data if valid (not all nans and is in desired region)
            reg = reg_rl[:-2]
            if not reg in self.desired_regions:
                continue
            reg_indx = reg_indices[idx] - 1
            reg_data = window_activity[:, :, reg_indx.flatten()]
            num_regs = reg_indx.flatten().size
            if np.sum(np.isnan(reg_data)) == reg_data.size:
                continue

            # Ignore NaN sections
            nans = np.argwhere(np.isnan(reg_data))[:,0].flatten()
            beginning_nans = nans[nans < num_bins//2]
            ending_nans = nans[nans > num_bins - num_bins//2]
            if beginning_nans.size > 0:
                start = max(beginning_nans) + 1
            else:
                start = 0 
            if ending_nans.size > 0:
                end = min(ending_nans)
            else:
                end = num_bins

            # Transform with PCS
            data_to_reduce = reg_data[:,start:end,:]
            num_nonnan_bins = data_to_reduce.shape[1]
            data_to_reduce = data_to_reduce.reshape(
                (num_trials*num_nonnan_bins, num_regs)
                )
            for n_c in range(1, reg_indx.size):
                pca = PCA(n_components=n_c, whiten=True)
                transformed_data = pca.fit_transform(data_to_reduce)
                if np.sum(pca.explained_variance_ratio_) > self.pc_var_threshold:
                    break
            transformed_data = transformed_data.reshape(
                (num_trials, num_nonnan_bins, n_c)
                )
            pc_data.append(transformed_data)
        pc_data = np.concatenate(pc_data, axis=2)
        return pc_data
    
class LRChoice(LearningPredictor):
    """
    Logistic regression predictor. Looks one frame into the future. Regularized
    by specified norm.
    """

    def __init__(
        self, session, reduce_dim=None,
        mode=None, shuffle=False, penalty='l2'
        ):
        super(LRChoice, self).__init__(session, reduce_dim, mode, shuffle)
        self.penalty = penalty

    def _fit_window(self, X, y, indices):
        """
        Fits a regularized logistic regression model, predicting
        left/right licking choice.
        Only fits a model if the number of trials is at least twice the number
        of covariates.
        
        Args
            start_idx: index in delay period to start extracting a window
                of activity.
            window_length: size of the window of activity to extract
        """

        assert(X.shape[0] == y.size)
        assert(indices.size == y.size)
        test_size = 0.40
        train_size = 1. - test_size
        event_length = X.shape[1]
        if X.shape[2] > (y.size*train_size)*2:
            return [np.nan]*event_length, [None]*event_length,\
                None, [np.nan]*event_length
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            train_test_split(
                X, y, indices, test_size=test_size, stratify=y
                )
        scores = []
        models = []
        predict_probs = []
        for frame in range(event_length):
            window_X_train = X_train[:, frame, :].squeeze()
            window_X_test = X_test[:, frame, :].squeeze()
            test_score, log_reg = self._train_model(
                window_X_train, window_X_test, y_train, y_test
                )
            predict_prob = log_reg.predict_proba(X[:,frame,:].squeeze())
            scores.append(test_score)
            models.append(log_reg)
            predict_probs.append(predict_prob)
        return scores, models, test_indices, predict_probs

    def _train_model(self, X_train, X_test, y_train, y_test):
        """
        Trains LR model over the input data
        """
            
        if self.penalty == "l1":
            solver="liblinear"
        else:
            solver="lbfgs"
        log_reg = LogisticRegressionCV(
            Cs=5, cv=5, scoring='accuracy', max_iter=500,
            penalty=self.penalty, solver=solver
            )
        log_reg.fit(X_train, y_train)
        correct_test_indices = (y_test == log_reg.predict(X_test))
        test_score = np.sum(correct_test_indices)/np.size(y_test)
        return test_score, log_reg
