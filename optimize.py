import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

def fame_equition(params,X,q_weeks,y_actual):
    params_mu = params[:5]
    params_sigma = params[5:]
    w_mu = params_mu[:-1]
    b_mu = params_mu[-1]
    w_s = params_sigma[:-1]
    b_s = params_sigma[-1]
    mu = np.dot(X, w_mu) + b_mu
    y2 = np.dot(X, w_s) + b_s
    sigma = np.log(1+np.exp(y2))
    vote = mu+