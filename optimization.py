import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import random
from config import *
from dataset_creation import *
from operator import itemgetter


# Creating Sigma matrix
def create_return_and_sigma_matrix(df, sigma, factor=True):
    if not factor:
        df1 = df[[symbol + "_return" for symbol in symbols]]
    else:
        df1 = df[[factor + "_return" for factor in factors]]
    returns = df1.mean().to_numpy()
    P = df1.corr().to_numpy()
    vol = np.array(sigma)
    Sigma = np.dot(P, np.dot(vol, vol.T))
    return Sigma, returns, P


def mean_variance_optimizing(Sigma, returns, lbd, low_bound=0,
                             high_bound=1, factor=True):
    # constraints : low_bound < weights < high_bound and sum(weights) = 1
    def fun(x):
        return 0.5*lbd*np.dot(x.T, np.dot(Sigma, x)) - np.dot(returns.T, x)

    size = len(Sigma)

    bnds = tuple([(low_bound, high_bound) for _ in range(size)])  # Boundaries
    cons = ({'type': 'eq', 'fun': lambda w:  sum(w) - 1})   # Sum(weight)=1
    if not factor:
        initial_weights = [random.random() for _ in range(size)]  # initial weights
        sum_weights = sum(initial_weights)
        initial_weights = tuple([x / sum_weights for x in initial_weights])
    else:
        last_weight = 0
        initial_weights = []
        while (low_bound > last_weight) or (last_weight > high_bound):
            initial_weights = [((random.random()*(high_bound - low_bound)) + low_bound) for _ in range(size-1)]
            last_weight = 1 - sum(initial_weights)
        initial_weights.append(last_weight)

    mvo = minimize(fun, initial_weights, bounds=bnds,
                   constraints=cons)

    return mvo


def min_Vol(Sigma, returns, min_return, low_bound=0, high_bound=1, factor=True):
    # constraints : low_bound < weights < high_bound and sum(weights) = 1
    def fun(x):
        return np.dot(x.T, np.dot(Sigma, x))

    size = len(Sigma)

    bnds = tuple([(low_bound, high_bound) for _ in range(size)])  # Boundaries
    cons = ({'type': 'eq', 'fun': lambda w:  sum(w) - 1}, # Sum(weight)=1
            {'type': 'ineq', 'fun': lambda w:  np.dot(returns.T, w) - min_return})

    if not factor:
        initial_weights = [random.random() for _ in range(size)]  # initial weights
        sum_weights = sum(initial_weights)
        initial_weights = tuple([x / sum_weights for x in initial_weights])
    else:
        last_weight = 0
        initial_weights = []
        while (low_bound > last_weight) or (last_weight > high_bound):
            initial_weights = [((random.random()*(high_bound - low_bound)) + low_bound) for _ in range(size-1)]
            last_weight = 1 - sum(initial_weights)
        initial_weights.append(last_weight)

    mvo = minimize(fun, initial_weights, bounds=bnds,
                   constraints=cons)

    return mvo


def sharp_ratio_optimizing(Sigma, returns, low_bound=0,
                           high_bound=1, factor=True):
    # constraints : low_bound < weights < high_bound and sum(weights) = 1
    def fun(x):
        return np.dot(returns.T, x)/(np.sqrt(np.dot(x.T, np.dot(Sigma, x))))

    size = len(Sigma)

    bnds = tuple([(low_bound, high_bound) for _ in range(size)])  # Boundaries
    cons = ({'type': 'eq', 'fun': lambda w:  sum(w) - 1})   # Sum(weight)=1
    if not factor:
        initial_weights = [random.random() for _ in range(size)]  # initial weights
        sum_weights = sum(initial_weights)
        initial_weights = tuple([x / sum_weights for x in initial_weights])
    else:
        last_weight = 0
        initial_weights = []
        while (low_bound > last_weight) or (last_weight > high_bound):
            initial_weights = [((random.random()*(high_bound - low_bound)) + low_bound) for _ in range(size-1)]
            last_weight = 1 - sum(initial_weights)
        initial_weights.append(last_weight)

    mvo = minimize(fun, initial_weights, bounds=bnds,
                   constraints=cons)

    return mvo


def max_diversification(P, returns, low_bound=0,
                        high_bound=1, factor=True):
    # constraints : low_bound < weights < high_bound and sum(weights) = 1
    def fun(x):
        return np.sqrt(np.dot(x.T, np.dot(P, x)))

    size = len(P)

    bnds = tuple([(low_bound, high_bound) for _ in range(size)])  # Boundaries
    cons = ({'type': 'eq', 'fun': lambda w:  sum(w) - 1})   # Sum(weight)=1
    if not factor:
        initial_weights = [random.random() for _ in range(size)]  # initial weights
        sum_weights = sum(initial_weights)
        initial_weights = tuple([x / sum_weights for x in initial_weights])
    else:
        last_weight = 0
        initial_weights = []
        while (low_bound > last_weight) or (last_weight > high_bound):
            initial_weights = [((random.random()*(high_bound - low_bound)) + low_bound) for _ in range(size-1)]
            last_weight = 1 - sum(initial_weights)
        initial_weights.append(last_weight)

    mvo = minimize(fun, initial_weights, bounds=bnds,
                   constraints=cons)

    return mvo


def min_correlation(P, sigma, low_bound=0,
                    high_bound=1, factor=True):
    # constraints : low_bound < weights < high_bound and sum(weights) = 1
    size = len(P)
    #vol = np.sqrt(np.diag(Sigma))
    initial_weights = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if j == i:
                pass
            else:
                initial_weights[i] += P[i, j]/(size) # computing initial portfolio weight estimates
    mu_initial = np.mean(initial_weights) # computing mean of the initial portfolio weight estimates
    sigma_initial = np.std(initial_weights) # computing standard deviation of the initial portfolio weight estimates

    w_T = np.zeros(size)
    for i in range(size):
        w_T[i] = 1 - norm.cdf(initial_weights[i],mu_initial,sigma_initial) # Scaling correlations between 0 and 1
 
    rank_in = np.array([k for k in range(0,size)])
    w_rank, rank_out = zip(*sorted(zip(w_T, rank_in)))
    w_rank += np.array([1 for _ in range(size)])
    w_rank = w_rank/(size*(size + 1)/2)

    weights = np.dot(w_rank, np.ones((size,size)) - P) # combining rank portfolio weight estimates with correlation matrix
    weights = weights/sum(weights) 

    for i in range(size):
        weights[i] = weights[i]/sigma[i] #scaling portfolio weights by assets standard deviations

    return weights/sum(weights)


def equal_risk_contribution(Sigma, returns, low_bound=0, high_bound=1,
                            factor=True):
    size = len(Sigma)

    def fun(x):
        result = 0
        for i in range(size):
            for j in range(size):
                result += (x[i] * ((np.dot(Sigma, x))[i]) - x[j] * ((np.dot(Sigma, x))[j]) )**2
        return result

    bnds = tuple([(low_bound, high_bound) for _ in range(size)])  # Boundaries
    cons = ({'type': 'eq', 'fun': lambda w:  sum(w) - 1})   # Sum(weight)=1
    if not factor:
        initial_weights = [random.random() for _ in range(size)]  # initial weights
        sum_weights = sum(initial_weights)
        initial_weights = tuple([x / sum_weights for x in initial_weights])
    else:
        last_weight = 0
        initial_weights = []
        while (low_bound > last_weight) or (last_weight > high_bound):
            initial_weights = [((random.random()*(high_bound - low_bound)) + low_bound) for _ in range(size-1)]
            last_weight = 1 - sum(initial_weights)
        initial_weights.append(last_weight)

    mvo = minimize(fun, initial_weights, bounds=bnds,
                   constraints=cons)

    return mvo


def russel(df):
    return_list = []
    for factor in factors:
        return_list.append([factor, 0])
    df1 = df[[factor + "_return" for factor in factors]]
    returns = df1.mean().to_numpy()  # metric 1

    # Excess return
    for i in range(len(factors)):
        return_list[i][1] = returns[i]

    # Active semi-deviation and Correlation of excess returns
    Corr_matrix = df1.corr().mean().to_numpy()   # metric 2
    for j in range(len(factors)):
        result = 0
        valeurs = df[factors[j]].to_numpy()
        size = len(valeurs)
        mean = df[factors[j]].mean()
        cpt = 0
        for i in range(size):
            if valeurs[i] < mean:
                result += (mean - valeurs[i])**2  # metric 3
                cpt += 1
        if cpt != 0:
            result = ( 1/cpt * result )**(0.5)
        return_list[j] += [result, Corr_matrix[j]]

    factors_rank = {}
    cond = True
    for factor in factors:
        factors_rank[factor] = 0
    for j in range(1, 4):
        if j == 2 or j ==3:
            cond = False
        return_list = sorted(return_list, key=itemgetter(j), reverse=cond)
        for i in range(len(return_list)):
            factors_rank[return_list[i][0]] += i + 1

    factors_rank_to_list = []
    for factor in factors:
        factors_rank_to_list.append([factor, factors_rank[factor]])

    factors_rank_to_list = sorted(factors_rank_to_list, key=itemgetter(1))
    for i in range(len(factors_rank_to_list)):
        factors_rank[factors_rank_to_list[i][0]] = i + 1

    #mvo = [(6-factors_rank[factor])/15 for factor in factors]
    w_russel = [0.3, 0.25, 0.2, 0.15, 0.1]
    mvo = [w_russel[factors_rank[factor] - 1] for factor in factors]
    sum_mvo = sum(mvo)
    mvo = [elm/sum_mvo for elm in mvo]

    return mvo


def risk_parity(df, Sigma, returns):
    std_matrix = df.std()
    weights = [std_matrix[6+i] for i in range(len(factors))]
    sum_weights = sum(weights)
    return weights/sum_weights
()