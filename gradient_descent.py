import numpy as np


def gradient_descent(A, b, step_size=0.01, tolerance=1e-6, max_iterations=1000):
    """
    Minimize f(x) = (1/2) ||Ax - b||^2 with respect to x using gradient descent.

    Args:
        A (numpy.ndarray): Matrix A of shape (m, n)
        b (numpy.ndarray): Vector b of shape (m)
        step_size (float): Step size for gradient descent (default: 0.01)
        tolerance (float): Tolerance for stopping criterion (default: 1e-6)
        max_iterations (int): Maximum number of iterations (default: 1000)

    Returns:
        numpy.ndarray: Optimal value of x
    """
    m, n = A.shape
    x = np.zeros(n)  # Initialize x with zeros
    steps = 0

    for i in range(max_iterations):
        steps = steps + 1
        gradient = A.T @ (A @ x - b)
        residual = np.linalg.norm(gradient)
        print("Iteration %d, Residual %f" % (steps, residual))
        if residual < tolerance:
            print(steps)
            break

        x = x - step_size * gradient
        loss(x, A, b)

    return x

def loss(x, A, b):
    loss = 1/2 * (A @ x - b).T @ (A @ x - b)
    print(f"The loss is {loss}")


# Example usage
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 11])

x_optimal = gradient_descent(A, b)
print(f"Optimal value of x: {x_optimal}")
