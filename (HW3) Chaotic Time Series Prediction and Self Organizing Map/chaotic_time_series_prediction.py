import numpy as np
import pandas as pd

# Parameters
N = 3       # Input neurons
M = 500     # Reservoir neurons
k = 0.01    # Ridge regression parameter

# Load Data
training_set_df = pd.read_csv('training-set.csv', header=None)
training_set = training_set_df.values.T
test_set_df = pd.read_csv('test-set.csv', header=None)
test_set = test_set_df.values.T

# Initialize Weights
w_in = np.random.normal(0, np.sqrt(0.002), size=(M, N))     # Input weights
w = np.random.normal(0, np.sqrt(2/M), size=(M, M))          # Reservoir weights

# Functions. Ridge regression, Update rule, etc.
def ridge_regression(X, y, k, M):
    I = np.identity(M)
    X_T_X = np.dot(X.T, X)
    return np.dot(np.dot(y.T, X), np.linalg.inv(X_T_X + k * I))

def update_reservoir(r, x_t, w_in, w):
    return np.tanh(np.dot(w, r) + np.dot(w_in, x_t))

def train_output_weights(x, w_in, w, M, k):
    rows = x.shape[0] - 1
    r = np.zeros(M)
    X = np.zeros((rows, M))

    for i in range(rows):
        r = update_reservoir(r, x[i,:], w_in, w)
        X[i,:] = r
    y = x[1::,:]
    w_out = ridge_regression(X, y, k, M)
    
    return w_out

def predict_with_reservoir(x, w_in, w, w_out, M, prediction_steps):
    r = np.zeros(M)
    predictions  = np.zeros((prediction_steps, x.shape[1]))

    for dp_idx in range(x.shape[0]):
        r = update_reservoir(r, x[dp_idx,:], w_in, w)
    for step in range(prediction_steps):
        predictions[step,:] = np.dot(w_out, r)
        r = update_reservoir(r, predictions [step,:], w_in, w)

    return predictions

# Main Logic
w_out = train_output_weights(training_set, w_in, w, M, k)

# Predict 500 timesteps (10s = 500dt) from test data
test_pred = predict_with_reservoir(test_set, w_in, w, w_out, M, 500)

# Saving Predictions y values
predicted_y_values = test_pred[:,1]
df_predictions = pd.DataFrame(predicted_y_values)
df_predictions.to_csv('prediction.csv', index=False, header=False, sep=',')



# Extra part:
import matplotlib.pyplot as plt

# Plot results for prediction on training set
train_pred = predict_with_reservoir(training_set[0:1000], w_in, w, w_out, M, 500)
plt.plot(np.arange(500), training_set[1000:1500,1], label='Actual data')
plt.plot(np.arange(500), train_pred[:,1], 'r--', label='Prediction')
plt.xlabel("Time steps")
plt.legend()
plt.show()