#!/usr/bin/env python
# coding: utf-8

# # GLM HMM with EM on Simulated Data
# Import statements
import os
import traceback
import pickle
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from SyntheticHMM import *
import ssm
import pickle

D = 1

def main(datatype, transitions):
    np.random.seed(0)
    glmhmm_file = datatype + "_glmhmm_data.p"
    data = pickle.load(open(glmhmm_file, "rb"))

    # Grid Search over regularization weights
    prior_weights = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

    # Grid Search over Ks
    Ks = [1,2,3,4,5]
    results_K = {}
    for K in Ks:
        results_K[K] = {}
        for prior_weight in prior_weights:
            results = _fit_hmm_Ks(data, K, prior_weight, transitions)
            results_K[K][prior_weight] = results

    pickle.dump(results_K, open(
        datatype + "_glmhmm_results.p","wb"
        ))

def _fit_hmm(
    K, X_train, y_train,
    X_test, y_test, prior_weight, transitions
    ):
    """
    Fits HMM with given parameters 20 times and returns all of them in a list
    of dictionaries
    """

    input_size = X_train.shape[1]
    hmm = ssm.HMM(
        K, D, M=input_size, observations="logistic",
        transitions=transitions,
        observation_kwargs={
            "input_size":input_size, "prior_weight": prior_weight
            }
        )
    lls = hmm.fit(
        y_train,
        inputs=X_train,
        method="em"
        )
    test_ll = hmm.log_likelihood(y_test, inputs=X_test)
    result = {}
    result['hmm'] = hmm
    result['lls'] = lls
    result['test_ll'] = test_ll
    return result

def _fit_hmm_Ks(data, K, prior_weight, transitions):
    results = []
    for _ in range(20):
        X_train = data["X_train"] 
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]
        result = _fit_hmm(
            K, X_train, y_train,
            X_test, y_test, prior_weight, transitions
            )
        results.append(result)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify GLMHMM")
    parser.add_argument("d") # Datatype
    parser.add_argument("t") # Transition type
    args = parser.parse_args()
    main(args.d, args.t)
