import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

from . import util

class CorrelationFunction:
    r"""The free correlation function of a reservoir is defined as
        C(t) = < B(t) B(0) >,
    where the expectation value is taken in the thermal equilibrium state
    and B is the bath coupling operator.

    It can be approximately represented as
        C(t) = \sum_j c_j exp(nu_j t) + 2 c_delta * delta(t)
    (for t>0), see e.g. [Kato and Tanimura, J. Chem. Phys. (2016)].
    The coefficients c_j and frequencies nu_j may be complex; however, c_delta must be real.

    The correlation functions for baths with some common spectral densities can be obtained
    using the following helper functions:
        * brownian_correlation_function
        * TODO: others
    Additionally, fit_correlation_function can be used for fitting an analytically given function
    with a multi-exponential expression.

    Member variables:
        * coeffs: list of coefficients c_j
        * freqs: list of frequencies nu_j
        * c_delta: as above
        * num_terms: number of coefficients (=number of frequencies)
    """

    def __init__(self, coeffs, freqs, c_delta=0):
        self.coeffs = np.array(coeffs)
        self.freqs = np.array(freqs)
        self.c_delta = c_delta

        if len(coeffs) != len(freqs):
            raise ValueError
        self.num_terms = len(coeffs)

    def value(self, time):
        r"""Gives C(t) at the time t specified by the argument.
        The delta-contribution is ignored.
        Negative times are handled correctly (assuming time-reversal symmetry C(-t) = C(t)*)."""

        if time < 0:
            return np.conjugate(self.value(-time))
        return (self.coeffs * np.exp(self.freqs * time)).sum()

    def plus(self, other):
        """Gives the sum of this correlation function and the other."""

        new_coeffs = []
        new_freqs = []
        # We go through both coefficient lists so that common frequencies are combined.
        # To this end, we create a list of the pairs (c_j, nu_j) of the other correlation function
        #   so that we can remove an element from that list if it is absorbed.
        other_freq_coeff = list(zip(other.freqs, other.coeffs))

        for freq, coeff in zip(self.freqs, self.coeffs):
            # pylint: disable=cell-var-from-loop, undefined-loop-variable
            other_index = util.first_index(other_freq_coeff, lambda pair: pair[0] == freq)
            if other_index is not None:
                coeff += other_freq_coeff[other_index][1]
                del other_freq_coeff[other_index]
            new_freqs.append(freq)
            new_coeffs.append(coeff)

        for freq, coeff in other_freq_coeff:
            new_freqs.append(freq)
            new_coeffs.append(coeff)

        return CorrelationFunction(new_coeffs, new_freqs, self.c_delta + other.c_delta)


def brownian_correlation_function(coup_strength, res_freq, half_width, beta, num_matsubara=0):
    r"""Correlation function corresponding to the spectral density of underdamped Brownian motion,
        J(w) = 2G l^2 w / ((w^2 - w0^2)^2 + 4G^2 w^2) ,
    where l is the coupling strength, w0 the resonance frequency and G < w0 the half width.

    The correlation function has two terms with frequencies -G (+-) W, where W^2 = w0^2 - G^2, in
        addition to an infinite number of Matsubara terms.
    The parameter num_matsubara specifies how many of these terms should be included.
    The parameter beta is the inverse temperature of the reservoir
        (given in the same units as w0 and G, note that hbar = kB = 1).

    The inverse temperature may not be zero, but it may be infinity. At beta=np.inf, the Matsubara
        frequencies approach a continuum. This function can then not calculate the Matsubara
        contribution, use fit_correlation_function instead. Numerically exact values for the
        zero-temperature Matsubara contribution can be obtained from brownian_matsubara_zeroT."""

    if (half_width >= res_freq) or (beta <= 0) or (beta == np.inf and num_matsubara != 0):
        raise ValueError

    # pylint: disable=invalid-name
    Omega = np.sqrt(res_freq**2 - half_width**2)

    if beta == np.inf:
        coeff1 = coup_strength**2 / (2 * Omega)
        freq1 = -half_width - 1j * Omega
        return CorrelationFunction([coeff1], [freq1])

    freq1 = -half_width + 1j * Omega
    freq2 = -half_width - 1j * Omega
    prefactor = coup_strength**2 / (4 * Omega)
    coeff1 = prefactor * (util.coth(-1j * beta * freq1 / 2) - 1)
    coeff2 = prefactor * (util.coth(1j * beta * freq2 / 2) + 1)

    # these are the exponents in the matsubara terms, i.e., negatives of
    # what is usually called the "matsubara frequencies":
    matsu_frequencies = np.linspace(1, num_matsubara, num=num_matsubara, endpoint=True) *\
                        (-2*np.pi / beta)
    denominators = (-freq1**2 + matsu_frequencies**2) * (-freq2**2 + matsu_frequencies**2)
    matsu_coeffs = 4 * coup_strength**2 * half_width / beta * matsu_frequencies / denominators

    return CorrelationFunction(np.concatenate([[coeff1, coeff2], matsu_coeffs]),
                               np.concatenate([[freq1, freq2], matsu_frequencies]))


def brownian_matsubara_zeroT(coup_strength, res_freq, half_width, time):
    r"""Zero-temperature Matsubara contribution M(t) to correlation function corresponding to
        underdamped Brownian spectral density (see brownian_correlation_function).
    M(t) is evaluated at the given time by numerically integrating the analytic expression taken
        from [Lambert et al., Nat. Commun. (2019)].
    For an explanation of the other parameters, see brownian_correlation_function."""
    
    if half_width > res_freq:
        raise ValueError

    # pylint: disable=invalid-name
    Omega = np.sqrt(res_freq**2 - half_width**2)
    freq1 = -half_width + 1j * Omega
    freq2 = -half_width - 1j * Omega

    prefactor = -2 * coup_strength**2 * half_width / np.pi
    integrand = lambda x: np.real(prefactor * (x * np.exp(-x * time)) /\
                                  ((-freq1**2 + x**2) * (-freq2**2 + x**2)))
    return quad(integrand, 0, np.inf)[0]


def fit_correlation_function(tlist, exact_function, coeff_guess, freq_guess,
                             coeff_bounds=None, freq_bounds=None, **kwargs):
    r"""Fit a multi-exponential correlation function to the given exact function
        (which must be a function of one parameter, time, taken from tlist). The number of
        exponentials is determined from the number of coefficients/frequencies in the initial guess.
    The named optional parameters specify the bounds of the allowed parameter intervals for both the
        coefficients and frequencies (given as lists of (lower, upper)). The kwargs are passed on to
        the scipy least_squares fitting routine."""

    # num(frequencies) must equal num(coefficients)
    num_exp = len(coeff_guess)
    if len(freq_guess) != num_exp:
        raise ValueError

    # use default (infinite) bounds for the parameters where no bounds are provided
    if coeff_bounds is None:
        coeff_bounds = [(-np.inf, np.inf)] * num_exp
    if freq_bounds is None:
        freq_bounds = [(-np.inf, np.inf)] * num_exp

    # the number of bounds must fit the number of frequencies and coefficients
    if (len(coeff_bounds) != num_exp) or (len(freq_bounds) != num_exp):
        raise ValueError

    # scipy expects bounds as ([... lower bounds ...], [... upper bounds ...])
    bounds = np.concatenate([coeff_bounds, freq_bounds]).transpose()

    # extract function values and normalize
    ydata = [exact_function(t) for t in tlist]
    avg_y = np.average(np.abs(ydata))
    function_data = list(zip(tlist, ydata / avg_y))

    # perform fit. exp_fit_residuals returns a list of differences between the exact function values
    #   and those resulting from the fit with the given fit_params in the forms [coeffs : freqs]
    def exp_fit_residuals(fit_params):
        coeffs = fit_params[:num_exp]
        freqs = fit_params[num_exp:]
        return [(coeffs * np.exp(freqs * t)).sum() - y
                for t, y in function_data]
    result = least_squares(exp_fit_residuals,
                           np.concatenate([coeff_guess, freq_guess]),
                           bounds=bounds,
                           **kwargs)

    return CorrelationFunction(result.x[:num_exp] * avg_y, result.x[num_exp:])
