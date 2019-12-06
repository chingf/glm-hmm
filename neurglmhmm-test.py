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
from sklearn.model_selection import train_test_split
from SyntheticHMM import *
import ssm
import pickle

# Suppress std out
import sys
sys.stdout = open(os.devnull, "w")

D = 1

def main(datatype, transitions):
    glmhmm_file = "habanero_data/" + datatype + "_neurglmhmm_testdata.p"
    data = pickle.load(open(glmhmm_file, "rb"))

    # Grid Search over regularization weights
    prior_weights = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

    # Grid Search over Ks
    Ks = [1,2,3]
    results_K = {}
    for K in Ks:
        results_K[K] = {}

        # No initialization
        results_K[K]["noinit"] = {}
        for prior_weight in prior_weights:
            results = _fit_hmm_Ks(
                data, K, prior_weight, transitions, None
                )
            results_K[K]["noinit"][prior_weight] = results
        # Hard initialization 
        results_K[K]["init"] = {}
        for prior_weight in prior_weights:
            init_weight = 0.01
            results = _fit_hmm_Ks(
                data, K, prior_weight, transitions, init_weight
                )
            results_K[K]["init"][prior_weight] = results
    pickle.dump(results_K, open(
        "habanero_results/" + datatype + "_neurglmhmm_testresults.p","wb"
        ))

def _fit_hmm(
    K, X_train, y_train,
    X_test, y_test, prior_weight, transitions, init_weight
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
            "input_size":input_size, "prior_weight": prior_weight,
            "init_weight":init_weight
            }
        )
    if init_weight is None:
        initialize = False
    else:
        initialize = True
    lls = hmm.fit(
        y_train,
        inputs=X_train,
        method="em", initialize=initialize
        )
    test_ll = hmm.log_likelihood(y_test, inputs=X_test)
    result = {}
    result['hmm'] = hmm
    result['lls'] = lls
    result['test_ll'] = test_ll
    return result

def _fit_hmm_Ks(data, K, prior_weight, transitions, init_weight):
    results = []
    for _ in range(30):
        X = data["X"]
        y = data["y"]
        indices = data["indices"]
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            train_test_split(
                X, y, indices, test_size=0.20, stratify=y
                )
        result = _fit_hmm(
            K, X_train, y_train,
            X_test, y_test, prior_weight, transitions, init_weight
            )
        result['train_indices'] = train_indices
        result['test_indices'] = test_indices
        results.append(result)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify GLMHMM")
    parser.add_argument("d")
    parser.add_argument("t")
    sys.stdout = open(os.devnull, "w")
    args = parser.parse_args()
    main(args.d, args.t)
