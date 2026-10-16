"""
MA3012/MA4013 Computer Practical Sheet 1, Problem 2a
Centred difference approximation of a derivative.
"""
import numpy as np


def cdiff(f, x, h):
    """Approximate f'(x) by the centred difference (f(x+h) - f(x-h)) / (2h).

    Parameters
    ----------
    f : callable
        Function of one variable (must accept a float, or a numpy array if
        x is an array).
    x : float or numpy.ndarray
        Point(s) at which to approximate the derivative.
    h : float
        Step size, must be non-zero.

    Returns
    -------
    fp : float or numpy.ndarray
        Approximation to f'(x).
    """
    if h == 0:
        raise ValueError("step size h must be non-zero")
    fp = (f(x + h) - f(x - h)) / (2 * h)
    return fp


if __name__ == "__main__":
    x0 = 5 * np.pi / 8
    exact = -np.sin(x0)  # d/dx cos(x) = -sin(x)

    print(f"Derivative of cos(x) at x = 5*pi/8 = {x0:.15f}")
    print(f"exact value -sin(5pi/8)           = {exact:.15f}")
    print(f"{'h':>10}  {'cdiff':>20}  {'|error|':>10}")
    for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]:
        fp = cdiff(np.cos, x0, h)
        print(f"{h:>10.0e}  {fp:>20.15f}  {abs(fp - exact):>10.3e}")
