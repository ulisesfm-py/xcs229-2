import numpy as np


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, theta_0=None, verbose=True):
        """
        Args:
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        phi = np.mean(y)
        mu0 = np.mean(x[y == 0], axis=0)
        mu1 = np.mean(x[y == 1], axis=0)
        sigma = np.cov(x[y == 0], rowvar=False)*(1-phi) + \
            np.cov(x[y == 1], rowvar=False) * phi

        theta_0 = (np.log((1 - phi) / phi)
                   - 0.5 * (mu1.T @ np.linalg.inv(sigma) @ mu1)
                   + 0.5 * (mu0.T @ np.linalg.inv(sigma) @ mu0))
        theta = np.linalg.inv(sigma) @ (mu1 - mu0)
        self.theta = np.hstack([theta_0, theta])
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Calculate the sigmoid of the dot product of x and theta
        z = 1 / (1 + np.exp(-x @ self.theta))

        # Return 1 if z is greater than or equal to 0.5, otherwise return 0
        return (z >= 0.5).astype(int)
        # *** END CODE HERE
