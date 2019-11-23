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

# Coefficient options
easy = np.array(
    [[0.0, 1.5, 0.0],
     [-0.0, -1.5, -0.0]]
    )
med = np.array(
    [[0.01, 0.08, 0.02],
     [0.5, 0.12, 2]]
    )
hard = np.array(
    [[0.01, 0.08, 0.02],
     [1., 0.04, -0.50]]
    )
prevbiased = np.array(
    [[0.01, 0.08, 0.02],
     [3., 0.1, -1.5]]
    )
true_K = 2
D = 1

def main(coef_mode, trans_mode):
    np.random.seed(0)
    if coef_mode == "easy":
        true_coef = easy
    elif coef_mode == "med":
        true_coef = med
    elif coef_mode == "hard":
        true_coef = hard
    elif coef_mode == "prevbiased":
        true_coef = prevbiased

    # Intialize true model
    if trans_mode == "std":
        transition_coefs = np.zeros((true_K, true_coef.shape[1]))
        transitions = "standard"
    elif trans_mode == "indEasy":
        transition_coefs = np.random.rand(true_K, true_coef.shape[1])
        transitions = "inputdriven"
    elif trans_mode == "indHard":
        transition_coefs = np.random.rand(true_K, true_coef.shape[1])*2
        transitions = "inputdriven"
    true_hmm = SyntheticInputDrivenGLMHMM(
        true_K, true_coef, transition_coefs
        )

    # Grid Search over regularization weights
    prior_weights = [1e-4, 1e-2, 1e-1, 1, 2]

    # Grid Search over Ks
    Ks = [1,2,3,4,5]
    results_K = {}
#    for K in Ks: #TODO: remove
#        results_K[K] = {}
#        for prior_weight in prior_weights:
#            results = _fit_hmm_Ks(
#                true_hmm, K, transitions, prior_weight
#                )
#            results_K[K][prior_weight] = results

    # Grid search over datasize
    dsizes = [
        100, 200, 400, 600, 800, 1000
#        1200, 1500, 2000, 2500, 3000, #TODO: remove
#        4000, 4500, 5000
        ]
    results_dsize = {}
    for dsize in dsizes:
        results_dsize[dsize] = {}
        for prior_weight in prior_weights:
            results = _fit_hmm_datasize(
                true_hmm, dsize, transitions, prior_weight
                )
            results_dsize[dsize][prior_weight] = results

    all_results = {}
    all_results['true_hmm'] = true_hmm
    all_results['results_K'] = results_K
    all_results['results_dsize'] = results_dsize
    pickle.dump(all_results, open(
        trans_mode + "_" + coef_mode + str(true_K) + "state_results.p","wb"
        ))

def _fit_hmm(
    K, X_train, y_train,
    X_test, y_test, transitions, prior_weight
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

def _fit_hmm_Ks(true_hmm, K, transitions, prior_weight):
    # Generate simulated data
    train_size = 700
    test_size = 500 
    results = []
    for _ in range(20):
        X_train = np.random.choice(
            np.arange(-30,31,1), size=train_size).reshape((-1,1))
        X_test = np.random.choice(
            np.arange(-30,31,1), size=test_size).reshape((-1,1))
        sim_states_train, sim_y_train = true_hmm.sample(X_train)
        sim_states_test, sim_y_test = true_hmm.sample(X_test)
        sim_y_train = sim_y_train.astype(int)
        sim_y_test = sim_y_test.astype(int)
        sim_X_test = np.hstack((sim_y_test[:-1], X_test[1:].copy()))
        sim_X_train = np.hstack((sim_y_train[:-1], X_train[1:].copy()))
        sim_y_test = sim_y_test[1:]
        sim_y_train = sim_y_train[1:]
        result = _fit_hmm(
            K, sim_X_train, sim_y_train,
            sim_X_test, sim_y_test, transitions, prior_weight
            )
        results.append(result)
    return results

def _fit_hmm_datasize(true_hmm, train_size, transitions, prior_weight):
    test_size = 500 
    results = []
    for _ in range(20):
        X_train = np.random.choice(
            np.arange(-30,31,1), size=train_size).reshape((-1,1))
        X_test = np.random.choice(
            np.arange(-30,31,1), size=test_size).reshape((-1,1))
        sim_states_train, sim_y_train = true_hmm.sample(X_train)
        sim_states_test, sim_y_test = true_hmm.sample(X_test)
        sim_y_train = sim_y_train.astype(int)
        sim_y_test = sim_y_test.astype(int)
        sim_X_test = np.hstack((sim_y_test[:-1], X_test[1:].copy()))
        sim_X_train = np.hstack((sim_y_train[:-1], X_train[1:].copy()))
        sim_y_test = sim_y_test[1:]
        sim_y_train = sim_y_train[1:]
        result = _fit_hmm(
            true_K, sim_X_train, sim_y_train,
            sim_X_test, sim_y_test, transitions, prior_weight)
        results.append(result)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify GLMHMM")
    parser.add_argument("c")
    parser.add_argument("t")
    args = parser.parse_args()
    main(args.c, args.t)
