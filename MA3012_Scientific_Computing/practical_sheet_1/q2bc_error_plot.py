"""
MA3012/MA4013 Computer Practical Sheet 1, Problem 2b and 2c
Error of the centred difference approximation of cos'(5*pi/8) as a function
of h = 2^-k, k = 2, ..., 52, with fitted truncation and round-off error lines.
"""
import numpy as np
import matplotlib.pyplot as plt

from q2a_cdiff import cdiff

x0 = 5 * np.pi / 8

# Exact derivative: cos'(x) = -sin(x), and sin(5pi/8) = sqrt(2 + sqrt(2)) / 2.
# NB: the sheet prints -sqrt(2 - sqrt(2))/2 (= -sin(pi/8) ~ -0.383), which is a
# typo; the correct value is -sqrt(2 + sqrt(2))/2 ~ -0.924.
exact = -np.sqrt(2 + np.sqrt(2)) / 2
assert np.isclose(exact, -np.sin(x0))

# ---------------------------------------------------------------------------
# b) Error as a function of h
# ---------------------------------------------------------------------------
k = np.arange(2, 53)
h = 2.0 ** (-k)
fp = np.array([cdiff(np.cos, x0, hi) for hi in h])
err = np.abs(fp - exact)

print(f"{'k':>3} {'h':>10} {'|error|':>10}")
for ki, hi, ei in zip(k, h, err):
    print(f"{ki:>3} {hi:>10.3e} {ei:>10.3e}")

# ---------------------------------------------------------------------------
# c) Fit E_t(h) = C_t h^p (large h) and E_r(h) = C_r h^q (small h)
# ---------------------------------------------------------------------------
# Truncation error dominates for large h, round-off error for small h.
trunc = k <= 12           # h >= 2^-12 ~ 2.4e-4
round_ = k >= 26          # h <= 2^-26 ~ 1.5e-8
nonzero = err > 0         # log(0) undefined (exact hits can occur)


def fit_power_law(hh, ee):
    """Least-squares fit of log(E) = log(C) + p log(h); returns (C, p)."""
    p, logC = np.polyfit(np.log(hh), np.log(ee), 1)
    return np.exp(logC), p


Ct_free, p_free = fit_power_law(h[trunc & nonzero], err[trunc & nonzero])
Cr_free, q_free = fit_power_law(h[round_ & nonzero], err[round_ & nonzero])
print(f"\nFree fit:  E_t ~ {Ct_free:.3e} h^{p_free:.3f},  E_r ~ {Cr_free:.3e} h^{q_free:.3f}")

# p and q must be integers: round the fitted slopes, then refit C with the
# slope fixed (geometric mean of E / h^p over the fitting region).
p = int(round(p_free))
q = int(round(q_free))
sel_t = trunc & nonzero
sel_r = round_ & nonzero
Ct = np.exp(np.mean(np.log(err[sel_t]) - p * np.log(h[sel_t])))
Cr = np.exp(np.mean(np.log(err[sel_r]) - q * np.log(h[sel_r])))
print(f"Integer exponents: p = {p}, q = {q}")
print(f"Fitted constants:  C_t = {Ct:.4e},  C_r = {Cr:.4e}")

# Theoretical values for comparison.
# Taylor: (f(x+h) - f(x-h))/(2h) = f'(x) + f'''(x) h^2/6 + O(h^4)
#   => E_t = |f'''(x)|/6 h^2, with f'''(x) = sin(x) for f = cos.
# Round-off: each f(x +- h) has relative error <= u = eps/2, so the numerator
# has absolute error <= 2u|f(x)| and the quotient <= u|f(x)| / h = eps|f(x)|/(2h).
eps = np.finfo(float).eps
Ct_theory = abs(np.sin(x0)) / 6
Cr_theory = eps * abs(np.cos(x0)) / 2
print(f"Theory:            C_t = |f'''(x)|/6 = {Ct_theory:.4e},  "
      f"C_r ~ eps|f(x)|/2 = {Cr_theory:.4e} (bound)")

# Optimal h where E_t + E_r is minimal: d/dh (C_t h^2 + C_r/h) = 0
h_opt = (Cr / (2 * Ct)) ** (1 / 3)
print(f"Optimal h ~ (C_r / 2C_t)^(1/3) = {h_opt:.3e}, "
      f"min error ~ {Ct * h_opt**2 + Cr / h_opt:.3e}")
i_min = np.argmin(err)
print(f"Observed minimum error {err[i_min]:.3e} at h = 2^-{k[i_min]} = {h[i_min]:.3e}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def sci(v):
    """Format v as mathtext a.bc x 10^n for plot labels."""
    mant, expo = f"{v:.2e}".split("e")
    return rf"{mant} \times 10^{{{int(expo)}}}"


fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(h[nonzero], err[nonzero], "o-", ms=4, label="|centred difference error|")
ax.loglog(h, Ct * h**p, "--", label=rf"$E_t(h) = {Ct:.3f}\,h^{{{p}}}$ (truncation)")
ax.loglog(h, Cr * h**float(q), "--",
          label=rf"$E_r(h) = {sci(Cr)}\,h^{{{q}}}$ (round-off)")
ax.axvline(h_opt, color="grey", ls=":", lw=1, label=rf"optimal $h \approx {sci(h_opt)}$")
ax.set_xlabel("step size $h$")
ax.set_ylabel(r"$|D_h f(x) - f'(x)|$")
ax.set_title(r"Error of centred difference approximation of $\cos'(5\pi/8)$")
ax.set_ylim(1e-14, 10)
ax.grid(True, which="major", ls=":")
ax.legend()
fig.tight_layout()
fig.savefig("q2_error_plot.png", dpi=150)
plt.show()
