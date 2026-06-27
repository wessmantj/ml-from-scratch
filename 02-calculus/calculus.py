import math
from typing import Callable, List, Tuple


# approximates the derivative of f at x using the finite difference method
def derivative(f: Callable, x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x)) / h

# approximates the second derivative of f at x using the central finite difference method
def second_derivative(f: Callable, x: float, h: float = 1e-5) -> float:
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2

# approximates the partial derivative of f with respect to variable at index i
def partial_derivative(f: Callable, x: List[float], i: int, h: float = 1e-5) -> float:
    ...

# computes the gradient of f at point x as a list of partial derivatives
def gradient(f: Callable, x: List[float], h: float = 1e-5) -> List[float]:
    ...

# computes the directional derivative of f at x in the direction of unit vector d
def directional_derivative(f: Callable, x: List[float], d: List[float], h: float = 1e-5) -> float:
    ...

# computes the Jacobian matrix of a vector-valued function f at point x
def jacobian(f: Callable, x: List[float], h: float = 1e-5) -> List[List[float]]:
    ...

# computes the Hessian matrix (matrix of second partial derivatives) of f at point x
def hessian(f: Callable, x: List[float], h: float = 1e-5) -> List[List[float]]:
    ...

# performs one step of gradient descent, returning the updated parameters
def gradient_descent_step(f: Callable, x: List[float], lr: float) -> List[float]:
    ...

# runs gradient descent for n steps and returns the final parameter values
def gradient_descent(f: Callable, x: List[float], lr: float, n_steps: int) -> List[float]:
    ...

# finds a root of f near x0 using Newton's method in one dimension
def newtons_method(f: Callable, x0: float, n_steps: int) -> float:
    ...

# approximates the definite integral of f from a to b using the trapezoidal rule with n intervals
def trapezoidal_integral(f: Callable, a: float, b: float, n: int) -> float:
    ...

# approximates the definite integral of f from a to b using Simpson's rule with n intervals (n must be even)
def simpsons_integral(f: Callable, a: float, b: float, n: int) -> float:
    ...

# returns the nth-order Taylor series approximation of f around point a evaluated at x
def taylor_series(f: Callable, x: float, a: float, n: int) -> float:
    ...
