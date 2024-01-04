import numpy as np
import pandas as pd

#Random generator
rg = np.random.default_rng()

#Random generator of features and weights.
def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    weights = rg.random((1, n_values))[0]
    targets = np.random.choice([0,1], n_features)

    #Organise data in Pandas dataframe
    data = pd.DataFrame(features, columns=["x0","x1","x2"])
    data["targets"] = targets
    return data, weights

def get_weighted_sum(feature, weights, bias):
    #Product of two arrays
    return np.dot(feature, weights) + bias

#Sigmoid function for gradient descent
def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))

#Individual loss, cross entropy calculation
def cross_entropy(target, prediction):
    return -(target*np.log10(prediction) + (1-target)*np.log10(1-prediction))

#Gradient descent for new weights.
def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x,w in zip(feature, weights):
        new_w = w + l_rate*(target-prediction)*x
        new_weights.append(new_w)
    return new_weights

#new bias
def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate*(target-prediction)