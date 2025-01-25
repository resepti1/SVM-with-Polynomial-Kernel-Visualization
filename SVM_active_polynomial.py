import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers


class SVM_polynomial:
    """
    An implementation of an SVM classifier with a polynomial kernel
    using quadratic programming for optimization.

    Attributes:
        C (float): Regularization parameter.
    """
    def __init__(self, C):
        """
        Initialize the SVM with a regularization parameter.

        Args:
            C (float): Regularization parameter.
        """
        self.C = C

    def polynomial_kernel(self, x, z, gamma, r, degree):
        """
        Compute the polynomial kernel value between two points.

        Args:
            x (ndarray): First input vector.
            z (ndarray): Second input vector.
            gamma (float): Kernel coefficient.
            r (float): Kernel offset.
            degree (int): Degree of the polynomial.

        Returns:
            float: Kernel value.
        """
        return (gamma * np.dot(x, z) + r)**degree

    def compute_b(self, alphas, X, y, C):
        """
        Compute the bias for the SVM.

        Args:
            alphas (ndarray): Lagrange multipliers.
            X (ndarray): Training data.
            y (ndarray): Labels of the training data.
            C (float): Regularization parameter.

        Returns:
            float: Bias.
        """
        eps = 1e-7
        sv_indices = np.where((alphas > eps) & (alphas < C - eps))[0]

        # If no valid support vectors in the range, fallback to any non-zero alpha
        if len(sv_indices) == 0:
            sv_indices = np.where(alphas > eps)[0]

        b_values = []
        for i in sv_indices:
            s = 0
            for j in range(len(alphas)):
                s += alphas[j] * y[j] * \
                    self.polynomial_kernel(X[j], X[i], 0.5, 1, 3)
            b_i = y[i] - s
            b_values.append(b_i)

        print("b_values:", b_values)
        if not b_values:
            return 0
        return np.mean(b_values) if b_values else b_values[0]

    def get_prediction(self, alphas, b, X_train, x_grid, y):
        """
        Predict the value at a specific point using the trained model.

        Args:
            alphas (ndarray): Lagrange multipliers.
            b (float): Bias term.
            X_train (ndarray): Training data.
            x_grid (ndarray): Input data point to predict.
            y (ndarray): Labels of the training data.

        Returns:
            float: Predicted value.
        """
        result = np.sum(
            alphas * y * self.polynomial_kernel(X_train, x_grid, 0.5, 1, 3)
        )
        return result + b

    def train(self, X_train, y_train, C):
        """
        Train the SVM model using quadratic programming.

        Args:
            X_train (ndarray): Training data.
            y_train (ndarray): Labels for the training data.
            C (float): Regularization parameter.

        Returns:
            tuple: alphas and bias (b).
        """
        m = len(X_train)

        # Constructing the quadratic programming problem
        P = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                P[i, j] = y_train[i] * y_train[j] * \
                    self.polynomial_kernel(X_train[i], X_train[j], 0.5, 1, 3)

        q = -np.ones(m)  # Linear term in the objective function

        # Inequality constraints
        G_top = np.eye(m)
        G_bottom = -np.eye(m)
        G = np.vstack((G_top, G_bottom))

        h_top = np.ones(m) * C
        h_bottom = np.zeros(m)
        h = np.hstack((h_top, h_bottom))

        # Equality constraint
        A = y_train.reshape(1, -1)
        b = np.array([0.0])

        # Convert constraints and objective into cvxopt format
        P_cvx = matrix(P, tc='d')
        q_cvx = matrix(q, tc='d')
        G_cvx = matrix(G, tc='d')
        h_cvx = matrix(h, tc='d')
        A_cvx = matrix(A, tc='d')
        b_cvx = matrix(b, tc='d')

        # Solve the quadratic programming problem
        solution = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)

        alphas = np.array(solution['x']).ravel()
        print("alphas:", alphas)

        # Compute the bias term
        b = self.compute_b(alphas, X_train, y_train, C)

        return alphas, b

    def get_graph(self, X, y, grid_points):
        """
        Generate the decision boundary over a grid of points.

        Args:
            X (ndarray): Training data.
            y (ndarray): Labels for the training data.
            grid_points (ndarray): Points where predictions will be evaluated.

        Returns:
            ndarray: Predicted values at the grid points.
        """
        # Train the model to obtain alphas and bias
        alphas, b = self.train(X, y, self.C)

        # Compute predictions for the grid points
        Z = np.array([
            self.get_prediction(alphas, b, X, x_grid, y)
            for x_grid in grid_points
        ])

        return Z
