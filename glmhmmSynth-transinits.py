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
true_coef = None
D = 1

def main(coef_mode, diag_mean):
    global true_coef
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
    transition_coefs = np.zeros((true_K, true_coef.shape[1]))
    transitions = "initialized"
    true_hmm = SyntheticInputDrivenGLMHMM(
        true_K, true_coef, transition_coefs
        )

    # Grid Search over regularization weights
    prior_weights = [1e-4, 1e-2, 1e-1, 1, 2]

    # Grid Search over Ks
    Ks = [1,2,3,4,5]
    results_K = {}

    # Grid search over datasize
    dsizes = [
        100, 200, 400, 600, 800, 1000,
        1200, 1500, 2000, 2500, 3000,
        4000, 4500, 5000, 6000, 7000
        ]
    results_dsize = {}
    for dsize in dsizes:
        results_dsize[dsize] = {}
        for prior_weight in prior_weights:
            results = _fit_hmm_datasize(
                true_hmm, dsize, transitions, prior_weight, diag_mean
                )
            results_dsize[dsize][prior_weight] = results

    all_results = {}
    all_results['true_hmm'] = true_hmm
    all_results['results_K'] = results_K
    all_results['results_dsize'] = results_dsize
    pickle.dump(all_results, open(
        "inittrans" + str(diag_mean) + "_" + coef_mode + str(true_K) + "state_results.p","wb"
        ))

def _fit_hmm(
    K, X_train, y_train, X_test, y_test,
    transitions, prior_weight, true_hmm, diag_mean
    ):
    """
    Fits HMM with given parameters 20 times and returns all of them in a list
    of dictionaries
    """

    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    input_size = X_train.shape[1]
    hmm = ssm.HMM(
        K, D, M=input_size, observations="logistic",
        transitions=transitions,
        transition_kwargs={"diagonal_mean":diag_mean},
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

def _fit_hmm_datasize(true_hmm, train_size, transitions, prior_weight, diag_mean):
    test_size = 500 
    results = []
    for _ in range(20):
        X_train = np.random.choice(
            np.arange(-30,31,1), size=train_size
            ).reshape((-1,1))
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
            true_K, sim_X_train, sim_y_train, sim_X_test, sim_y_test,
            transitions, prior_weight, true_hmm, diag_mean)
        result['X_train'] = sim_X_train
        result['y_train'] = sim_y_train
        result['X_test'] = sim_X_test
        result['y_test'] = sim_y_test
        result['z_train'] = sim_states_train
        result['z_test'] = sim_states_test
        results.append(result)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Specify GLMHMM")
    parser.add_argument("c")
    args = parser.parse_args()
    for diag_mean in [0.5, 0.6, 0.7, 0.8]:
        main(args.c, diag_mean)
