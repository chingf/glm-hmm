import numpy as np
from scipy.io import loadmat
from LearningSession import *
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class PsychometricPredictor():
    """
    Trialmarker/Behavioral predictors will predict choice on a trial
    by trial basis, using trial events: discrimination difference, previous
    choice, and a bias term.
    Attributes
        session: The Session object of the session of interest.
    """

    session = None
    data = None
    trial_choices = None

    def __init__(self, session):
        self.session = session
        self.data = self._form_data_matrix()
        self.trial_choices = session.trialmarkers['ResponseSide'][1:]
        self.session.num_trials = self.session.num_trials - 1

    def _form_data_matrix(self):
        """
        Collects the covariates for the predictor: previous choice and
        discrimination difference The bias term is added automatically in the
        sklearn Logistic Regression model.
        Order of covariates:
        [prev choice, stim side, prev rewarded]
        """

        data = []
        for trial in range(1, self.session.num_trials):
            trial_data = []
            trial_data.append(
                self.session.trialmarkers['ResponseSide'][trial - 1] - 1
                )
            trial_data.append(
                self.session.trialmarkers['CorrectSide'][trial] - 1
                )
            trial_data.append(self.session.trialmarkers['Rewarded'][trial - 1])
            data.append(trial_data)
        return np.array(data)

class LRPsychometric(PsychometricPredictor):
    """
    Logistic regression predictor. Looks one frame into the future. Regularized
    by L2 norm.
    """

    def __init__(self, session):
        super(LRPsychometric, self).__init__(session)

    def fit(self):
        """
        Fits a cross-validated, L2-regularized logistic regression model
        """

        X = []
        y = []
        for trial in range(self.trial_choices.size):
            if np.isnan(self.trial_choices[trial]):
                continue
            if np.sum(np.isnan(self.data[trial,:])) > 0:
                continue
            X.append(self.data[trial,:])
            y.append(self.trial_choices[trial] - 1)
        X = np.array(X)
        y = np.array(y)
       
        # Training the model with cross validation
        log_reg = LogisticRegressionCV(
            Cs=5, cv=5, scoring='accuracy', max_iter=500
            )
        log_reg.fit(X, y)
        self.result = {
            "model": log_reg,
            "score": np.max(np.mean(log_reg.scores_[1], axis=0)),
            "data_size": y.size
            }
        return self.result
