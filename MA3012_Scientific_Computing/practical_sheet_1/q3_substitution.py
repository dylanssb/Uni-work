"""
MA3012/MA4013 Computer Practical Sheet 1, Problem 3
Forward substitution (Algorithm 2.1) and back substitution (Algorithm 2.2).
"""
import numpy as np
from scipy.linalg import solve_triangular


def forward_substitution(L, b):
    """Solve L x = b for x, where L is a square lower triangular matrix.

    x_i = (b_i - sum_{j<i} L_ij x_j) / L_ii,  for i = 1, ..., n
    """
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = L.shape[0]

    if L.ndim != 2 or L.shape[1] != n:
        raise ValueError("L must be a square matrix")
    if b.size != n:
        raise ValueError("b must have the same length as L")
    if not np.allclose(L, np.tril(L)):
        raise ValueError("L must be lower triangular")
    if np.any(np.diag(L) == 0):
        raise ValueError("L is singular (zero on the diagonal)")

    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
    return x


def back_substitution(U, b):
    """Solve U x = b for x, where U is a square upper triangular matrix.

    x_i = (b_i - sum_{j>i} U_ij x_j) / U_ii,  for i = n, ..., 1
    """
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = U.shape[0]

    if U.ndim != 2 or U.shape[1] != n:
        raise ValueError("U must be a square matrix")
    if b.size != n:
        raise ValueError("b must have the same length as U")
    if not np.allclose(U, np.triu(U)):
        raise ValueError("U must be upper triangular")
    if np.any(np.diag(U) == 0):
        raise ValueError("U is singular (zero on the diagonal)")

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def report(name, A, b, x, lower):
    """Print the solution and compare it with library solvers."""
    x_np = np.linalg.solve(A, b)
    x_sp = solve_triangular(A, b, lower=lower)
    print(name)
    print("  A =\n", A)
    print("  b =", b)
    print("  x =", x + 0.0)  # + 0.0 turns -0.0 into 0.0 for display
    print(f"  residual ||Ax - b||_2          = {np.linalg.norm(A @ x - b):.3e}")
    print(f"  ||x - numpy.linalg.solve||_2   = {np.linalg.norm(x - x_np):.3e}")
    print(f"  ||x - scipy solve_triangular|| = {np.linalg.norm(x - x_sp):.3e}\n")


if __name__ == "__main__":
    # a)  4x1 - 4 = 0
    #     5x1 - 2x2 - 5 = 0
    #    -3x2 + x3 + 1 = 0
    #     6x1 + 2x2 + 4x4 + 6 = 0
    # Move constants to the right-hand side to get L x = b.
    L = np.array([[4.0, 0.0, 0.0, 0.0],
                  [5.0, -2.0, 0.0, 0.0],
                  [0.0, -3.0, 1.0, 0.0],
                  [6.0, 2.0, 0.0, 4.0]])
    bL = np.array([4.0, 5.0, -1.0, -6.0])
    xL = forward_substitution(L, bL)
    report("3a) Forward substitution", L, bL, xL, lower=True)
    assert np.allclose(xL, [1, 0, -1, -3])

    # b)  3x1 - 3x2 + 2x4 = 2
    #     2x2 - x4 = 2
    #    -2x3 + 2x4 - 2 = 0
    #     3x4 + 6 = 0
    U = np.array([[3.0, -3.0, 0.0, 2.0],
                  [0.0, 2.0, 0.0, -1.0],
                  [0.0, 0.0, -2.0, 2.0],
                  [0.0, 0.0, 0.0, 3.0]])
    bU = np.array([2.0, 2.0, 2.0, -6.0])
    xU = back_substitution(U, bU)
    report("3b) Back substitution", U, bU, xU, lower=False)
    assert np.allclose(xU, [2, 0, -3, -2])
