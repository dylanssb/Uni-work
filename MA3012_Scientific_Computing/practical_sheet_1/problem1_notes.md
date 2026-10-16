# Practical Sheet 1, Problem 1: floating point precision and range

This problem is not submitted. Run `python3 problem1_floating_point.py` to reproduce every number below.

Python `float` (and MATLAB `double`) is an IEEE 754 double: 1 sign bit, 11 exponent bits and 52 stored fraction bits. Including the implicit leading 1, that gives 53 bits of precision. A normalised number is

$$x = \pm (1.f_1 f_2 \dots f_{52})_2 \times 2^{e}, \qquad -1022 \le e \le 1023.$$

## a) Adding a small value to 1.0

| k | `1.0 + 2**-k == 1.0` |
|---|---|
| 51 | False |
| 52 | False |
| 53 | True |

- The next double after 1.0 is $1 + 2^{-52}$. The gap $2^{-52} \approx 2.22\times10^{-16}$ is the **machine epsilon** (`sys.float_info.epsilon`, `np.finfo(float).eps`, `eps` in MATLAB).
- With round-to-nearest, any $\delta \le \varepsilon/2 = 2^{-53} \approx 1.11\times10^{-16}$ leaves 1.0 unchanged. Exactly $\varepsilon/2$ is a tie, and ties go to the even neighbour, which is 1.0. Anything slightly larger than $\varepsilon/2$ rounds up to $1+\varepsilon$.
- So the largest value you can add without changing 1.0 is $u = \varepsilon/2 = 2^{-53}$, called the **unit roundoff**. Some texts use $u$ rather than the gap $\varepsilon$ as the definition of "machine precision". Check which one the lecture notes use.

## b) A different number, e.g. 1234.56

- $2^{10} \le 1234.56 < 2^{11}$, so the gap between neighbouring doubles here is $\varepsilon \cdot 2^{10} = 2^{-42} \approx 2.27\times10^{-13}$ (`np.spacing(1234.56)`). Adding anything up to about half of that leaves the number unchanged.
- The absolute gap grows with $|x|$, but the gap **relative** to $x$ stays the same size: $2.27\times10^{-13}/1234.56 \approx 1.8\times10^{-16}$, between $\varepsilon/2$ and $\varepsilon$. This holds at every scale:

  | x | spacing | spacing / x |
  |---|---|---|
  | 1e-10 | 1.29e-26 | 1.29e-16 |
  | 1 | 2.22e-16 | 2.22e-16 |
  | 1234.56 | 2.27e-13 | 1.84e-16 |
  | 1e300 | 1.49e284 | 1.49e-16 |

- **Relative error** is $|\tilde{x} - x|/|x|$, the error measured against the size of the true value. For any real $x$ in the normal range, the stored value $\mathrm{fl}(x)$ satisfies

  $$\frac{|\mathrm{fl}(x) - x|}{|x|} \le u = \frac{\varepsilon}{2}, \qquad \text{equivalently } \mathrm{fl}(x) = x(1+\delta),\ |\delta| \le u.$$

  For example, 1234.56 is stored as `1234.55999999999994543...`, a relative error of $4.4\times10^{-17} \le u$.
- How this relates to a): machine epsilon bounds the *relative* precision, so doubles carry about 15 to 16 significant decimal digits at any magnitude. Part a) is the special case $x = 1$, where the absolute and relative gaps are equal.

## c) Largest and smallest numbers

| Quantity | Value | Formula |
|---|---|---|
| Largest finite (`sys.float_info.max`, MATLAB `realmax`) | 1.7977e308 | $(2-2^{-52})\,2^{1023}$ |
| Smallest normal (`sys.float_info.min`, `realmin`) | 2.2251e-308 | $2^{-1022}$ |
| Smallest subnormal (`np.nextafter(0, 1)`) | 4.94e-324 | $2^{-1074}$ |

**Going above the largest value gives overflow.**
- `max * 2` gives `inf`, `-max * 2` gives `-inf`, and `inf - inf` gives `nan`.
- `max + 1` gives `max`, because 1 is far smaller than the gap near `max` (about $2^{971}$).
- What you see depends on how the value is computed:
  - Ordinary arithmetic quietly returns `inf`.
  - `math.exp(1000)` and `2.0**1024` raise `OverflowError`.
  - `np.exp(1000)` returns `inf` with a RuntimeWarning.
  - MATLAB returns `Inf` with no warning.
- Python `int` has arbitrary precision, so `2**1024` is exact. `float(2**1024)` raises `OverflowError`.

**Going below the smallest value gives underflow.**
- Below `realmin`, numbers become **subnormal** (denormalised). The leading bit is no longer 1, so precision drops gradually. For example, `min/2` is about 1.11e-308 with fewer significant bits.
- Below $2^{-1074}$ the result rounds to `0.0`. Examples: `min / 2**53`, `1e-200 * 1e-200` and `math.exp(-1000)`. No error or warning is raised.

**Computing the limits step by step.** Repeatedly doubling 1.0 reaches $2^{1023}$ as the last finite value; one more doubling gives `inf`. To reach `realmax` itself you have to fill in the mantissa, for example `(2 - 2**-52) * 2.0**1023`. Repeatedly halving reaches $2^{-1074}$, then 0.

**Intermediate results matter.** `(1e200*1e200)/1e200` gives `inf`, but `1e200*(1e200/1e200)` gives `1e200`. A naive $\sqrt{a^2+b^2}$ overflows where `math.hypot` does not. Rearranging or rescaling a calculation avoids these spurious overflows.

**Technical terms**
- Results that exceed the representable range are **overflow** errors.
- Results that fall below it are **underflow** errors. Gradual underflow happens through the subnormal numbers.
- Both are range errors, in contrast to the **rounding (round-off) errors** in a) and b), which are bounded by $u$.
