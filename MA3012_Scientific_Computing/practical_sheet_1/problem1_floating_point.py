"""
MA3012/MA4013 Scientific Computing - Computer Practical Sheet 1, Problem 1
(not for submission): precision and magnitude limits of floating point numbers.

Python floats are IEEE 754 double precision (binary64):
    1 sign bit, 11 exponent bits, 52 stored fraction bits (53 bits of precision).
"""
import sys
import math
import warnings

import numpy as np


def smallest_change(x):
    """Return the smallest power of two delta such that x + delta != x.

    Starts at delta = 1 and keeps halving until adding it no longer changes x.
    """
    delta = 1.0
    while x + delta / 2 != x:
        delta /= 2
    # Halving any further would leave x unchanged
    return delta


# ---------------------------------------------------------------------------
# a) How small a value can be added to 1.0 and still change it?
# ---------------------------------------------------------------------------
print("a) Adding small values to 1.0")
for k in range(50, 55):
    d = 2.0 ** -k
    print(f"   1.0 + 2^-{k} == 1.0 ? {1.0 + d == 1.0}")

eps = smallest_change(1.0)
print(f"   smallest power of two that changes 1.0: {eps!r} = 2^{int(math.log2(eps))}")
print(f"   sys.float_info.epsilon               : {sys.float_info.epsilon!r}")
print(f"   np.finfo(float).eps                  : {np.finfo(float).eps!r}")
print(f"   unit roundoff u = eps/2              : {eps / 2!r}")
# Round-to-nearest-even: exactly eps/2 rounds back down to 1, but anything a
# little larger than eps/2 rounds up to 1 + eps.
print(f"   1 + eps/2 == 1 ? {1.0 + eps / 2 == 1.0}")
print(f"   1 + eps/2*(1+eps) == 1 ? {1.0 + (eps / 2) * (1 + eps) == 1.0}")
print(f"   next float after 1.0: {np.nextafter(1.0, 2.0)!r}  (gap = {np.spacing(1.0)!r})")

# ---------------------------------------------------------------------------
# b) The same for a different number, e.g. 1234.56
# ---------------------------------------------------------------------------
print("\nb) Adding small values to 1234.56")
x = 1234.56
dx = smallest_change(x)
print(f"   smallest power of two that changes {x}: {dx!r} = 2^{int(math.log2(dx))}")
print(f"   np.spacing({x})                      : {np.spacing(x)!r}")
print(f"   relative size dx/x                     : {dx / x!r}")
print(f"   compare with eps                       : {eps!r}")
# The gap between neighbouring floats is eps * 2^e where 2^e <= |x| < 2^(e+1)
e = math.floor(math.log2(x))
print(f"   2^{e} <= x < 2^{e + 1}, so gap = eps * 2^{e} = {eps * 2.0 ** e!r}")

# Relative error of storing a number: |fl(x) - x| / |x| <= u = eps/2
from decimal import Decimal
from fractions import Fraction
exact = Fraction("1234.56")
stored = Fraction(x)  # the exact value of the double actually stored
rel_err = abs(stored - exact) / exact
print(f"   1234.56 is actually stored as {Decimal(x)}")
print(f"   relative representation error      : {float(rel_err):.3e}  (<= u = {eps / 2:.3e})")

print("\n   gap / |x| for numbers of different size:")
for v in [1e-10, 1e-3, 1.0, 1234.56, 1e10, 1e300]:
    print(f"     x = {v:<8g}  spacing = {np.spacing(v):.3e}  spacing/x = {np.spacing(v) / v:.3e}")

# ---------------------------------------------------------------------------
# c) Largest and smallest numbers
# ---------------------------------------------------------------------------
print("\nc) Largest and smallest numbers")
fi = sys.float_info
tiny_sub = np.nextafter(0.0, 1.0)
print(f"   largest float         realmax : {fi.max!r}  = (2 - 2^-52) * 2^1023")
print(f"   smallest normal float realmin : {fi.min!r}  = 2^-1022")
print(f"   smallest subnormal            : {tiny_sub!r}  = 2^-1074")

print("\n   Exceeding the largest value (overflow):")
print(f"     max * 2          = {fi.max * 2}")
print(f"     max + 1          = {fi.max + 1}   (1 is far below the spacing at max)")
print(f"     max + max*eps/2  = {fi.max + fi.max * eps / 2}   (rounds up past max)")
print(f"     -max * 2         = {-fi.max * 2}")
print(f"     inf - inf        = {math.inf - math.inf}")
try:
    math.exp(1000)
except OverflowError as err:
    print(f"     math.exp(1000)   -> OverflowError: {err}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"     np.exp(1000)     = {np.exp(1000.0)}  (numpy returns inf with a warning)")
try:
    2.0 ** 1024
except OverflowError as err:
    print(f"     2.0 ** 1024      -> OverflowError: {err}")
print(f"     2**1024 (int)    = an exact Python int with {len(str(2**1024))} digits (ints are arbitrary precision)")
try:
    float(2**1024)
except OverflowError as err:
    print(f"     float(2**1024)   -> OverflowError: {err}")

print("\n   Going below the smallest value (underflow):")
print(f"     min / 2          = {fi.min / 2!r}  (subnormal, precision lost)")
print(f"     min / 2**52      = {fi.min / 2**52!r}  (smallest subnormal)")
print(f"     min / 2**53      = {fi.min / 2**53!r}  (underflows to zero)")
print(f"     1e-200 * 1e-200  = {1e-200 * 1e-200!r}")
print(f"     math.exp(-1000)  = {math.exp(-1000)!r}")

print("\n   Computing these values: repeated doubling / halving")
v, n = 1.0, 0
while not math.isinf(v * 2):
    v *= 2
    n += 1
print(f"     doubling 1.0: last finite value 2^{n} = {v!r}, next step gives inf")
v, n = 1.0, 0
while v / 2 != 0.0:
    v /= 2
    n += 1
print(f"     halving 1.0 : last non-zero value 2^-{n} = {v!r}, next step gives 0.0")

print("\n   Order of operations matters:")
a, b = 1e200, 1e200
print(f"     (a*b)/b  = {(a * b) / b}   (intermediate overflow)")
print(f"     a*(b/b)  = {a * (b / b)}")
print(f"     sqrt(a^2 + b^2) naive = {math.sqrt(a * a + b * b)},  math.hypot = {math.hypot(a, b)!r}")
