import numpy as np
import pandas as pd
from 02_activate_functions import generate_data, get_weighted_sum, sigmoid, cross_entropy,update_weights, update_bias

#Initial variables
bias = 0.5 
l_rate = 0.01
epochs = 30
epoch_loss = []
data, weights = generate_data(50,3)

def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs):
        individual_loss = []
        #Calculate output function for features and weights to set prediction value
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature,weights,bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            individual_loss.append(loss)
            # Gracient descent
            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
        average_loss = sum(individual_loss)/len(individual_loss)
        epoch_loss.append(average_loss)
        print("*****************************")
        print("epoch", e)
        print(average_loss)

train_model(data, weights, bias, l_rate, epochs)

# lot the average loss
df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid=True).get_figure()
df_plot.savefig("Training_Loss.pdf")