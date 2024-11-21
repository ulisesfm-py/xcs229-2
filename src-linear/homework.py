import numpy as np
import util
import importlib.util

train_path = 'ds1_train.csv'

# Load dataset
x_train, y_train = util.load_dataset(train_path, add_intercept=True)

use_submission = importlib.util.find_spec('submission') is not None
if use_submission:
    from submission import GDA, LogisticRegression

# *** START CODE HERE ***
n_examples, dim = x_train.shape
theta = np.zeros(dim)


# First derivative
z = np.dot(x_train, theta)
h = 1/(1 + np.exp(-z))
gradient = np.dot(x_train.T, (h-y_train)) / n_examples

# Second derivative or Hessian
H = (x_train.T * (h*(1-h)) @ x_train)/n_examples
# H has a shape of (dim, dim)

# Update theta
theta -= np.linalg.solve(H, gradient)

# *** END CODE HERE ***
