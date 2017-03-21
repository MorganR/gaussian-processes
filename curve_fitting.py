import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare

# VGP training times, no pca
x = [5, 10, 15, 25, 50, 75, 150]
y = [13.3, 78.2, 157.9, 450.4, 2026.1, 5052.9, 27476]

def power_fit(x, coeffs, index):
    if index == 0:
        return coeffs[0]
    return coeffs[index]*(x**index) + power_fit(x, coeffs, index - 1)

def n_cubed(n, p0, p1, p2, p3):
    return power_fit(n, [p0, p1, p2, p3], 3)

def m_n_squared(mn, m1, n0, n1, n2):
    return mn[0]*m1 + power_fit(mn[1], [n0, n1, n2], 2)

def fit_n_cubed(n, y, guess=[1, 1, 1, 1]):
    p_coeffs, _pcov = curve_fit(n_cubed, n, y, p0=guess, bounds=([0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
    err = np.sqrt(np.diag(_pcov))
    y_guess = n_cubed(n, p_coeffs[0], p_coeffs[1], p_coeffs[2], p_coeffs[3])
    chi2 = chisquare(y, y_guess, ddof=4)
    return p_coeffs, err, chi2

def fit_m_n_squared(m, n, y, guess=[1, 1, 1, 1]):
    p_coeffs, _pcov = curve_fit(m_n_squared, [m, n], y, p0=guess, bounds=([-np.inf, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
    err = np.sqrt(np.diag(_pcov))
    y_guess = m_n_squared([m, n], p_coeffs[0], p_coeffs[1], p_coeffs[2], p_coeffs[3])
    chi2 = chisquare(y, y_guess, ddof=4)
    return p_coeffs, err, chi2
