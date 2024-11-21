import numpy as np


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape
        if self.theta is None:
            self.theta = np.zeros(dim)

        for i in range(self.max_iter):
            # First derivative
            z = np.dot(x, self.theta)
            h = 1/(1 + np.exp(-z))
            gradient = np.dot(x.T, (h-y)) / n_examples

            # Second derivative or Hessian
            H = (x.T * (h*(1-h)) @ x)/n_examples

            # Update theta
            old_theta = self.theta.copy()
            self.theta -= np.linalg.solve(H, gradient)

            # Check convergence with L1 norm on theta
            if np.linalg.norm(self.theta - old_theta, 1) < self.eps:
                break

            # I'll print J every 1000 iterations
            if self.verbose and i % 1000 == 0:
                # Doing the sum and then dividing by n_examples is the same as doing np.mean()
                J = -np.mean(y * np.log(h) + (1 - y)*np.log(1 - h))
                print(f"Loss of {J} at Iteration {i}")

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        z = np.dot(x, self.theta)
        h = 1 / (1 + np.exp(-z))
        return h
        # *** END CODE HERE ***
