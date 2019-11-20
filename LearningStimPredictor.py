import numpy as np
from scipy.io import loadmat
from LearningSession import *
from LearningChoicePredictor import *
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class LRStim(LRChoice):
    """
    Logistic regression predictor. Looks one frame into the future. Regularized
    by L2 norm. To allow subclassing of Choice predicting class, the trial
    stim array is saved under trial_choices.
    """

    session = None
    datatype = None
    data = None
    trial_choices = None
    results = {} 
    loo_results = {}

    def __init__(self, session, predict_previous=False, shuffle=False):
        self.session = session
        self.trial_choices = session.trialmarkers['CorrectSide']
        
        if shuffle:
            np.random.shuffle(self.trial_choices)
        self.data = session.Vc['Vc']
        self.trial_indices = session.get_trial_indices()
        self.predict_previous = predict_previous
        
        if predict_previous:
            self.trial_choices = self.trial_choices[:-1]
            self.data = self.data[1:,:,:] 
            self.trial_indices = self.trial_indices[1:,:]
            self.session.num_trials = self.session.num_trials - 1

class SVCStim(SVCChoice):
    """
    SVM classifier with polynomial kernel. Looks one frame into the future.
    To allow subclassing of Choice predicting class, the trial stim array is
    saved under trial_choices.
    """

    session = None
    datatype = None
    data = None
    trial_choices = None
    results = {} 
    loo_results = {}

    def __init__(self, session, predict_previous=False, shuffle=False):
        self.session = session
        self.trial_choices = session.trialmarkers['CorrectSide']
        
        if shuffle:
            np.random.shuffle(self.trial_choices)
        self.data = session.Vc['Vc']
        self.trial_indices = session.get_trial_indices()
        self.predict_previous = predict_previous
        
        if predict_previous:
            self.trial_choices = self.trial_choices[:-1]
            self.data = self.data[1:,:,:] 
            self.trial_indices = self.trial_indices[1:,:]
            self.session.num_trials = self.session.num_trials - 1

