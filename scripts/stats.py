# -*- coding: utf-8 -*-
"""Statistical tests.

Example:

Notes:

Todo:

@author: Annette Stellema
@email: a.stellema@unsw.edu.au
@created: Mon Aug  1 15:01:41 2022

"""
import math
import numpy as np
from scipy import stats


def precision(var):
    """Determine the precision to print based on the number of digits.

    Values greater than ten: the precision will be zero decimal places.
    Values less than ten but greater than one: print one decimal place.
    Values less than one: print two decimal places.


    Args:
        var (DataArray): Transport dataset

    Returns:
        p (list): The number of decimal places to print
    """
    # List for the number of digits (n) and decimal place (p).
    n, p = 1, 1

    tmp = abs(var.item())
    n = int(math.log10(tmp)) + 1
    if n == 1:

        p = 1 if tmp >= 1 else 2
    elif n == 0:
        p = 2
    elif n == -1:
        p = 3
    return p


def format_pvalue_str(p):
    """Format significance p-value string to correct decimal places.

    p values greater than 0.01 rounded to two decimal places.
    p values between 0.01 and 0.001 rounded to three decimal places.
    p values less than 0.001 are just given as 'p>0.001'
    Note that 'p=' will also be included in the string.
    """
    if p <= 0.001:
        sig_str = 'p<0.001'
    elif p <= 0.01 and p >= 0.001:

        sig_str = 'p=' + str(np.around(p, 3))
    else:
        if p < 0.05:
            sig_str = 'p<' + str(np.around(p, 2))
        else:
            sig_str = 'p=' + str(np.around(p, 2))

    return sig_str


def test_signifiance(x, y):
    def resample(ds):
        return ds.resample(rtime="Y").mean("rtime", keep_attrs=True)
    times = [slice('2012'), slice('2070', '2101')]
    x, y = x.sel(rtime=times[0]), y.sel(rtime=times[1])
    x, y = resample(x), resample(y)
    t, p = stats.wilcoxon(x, y)
    p = format_pvalue_str(p)
    return p


def weighted_bins_fd(ds, weights):
    """Weighted Freedman Diaconis Estimator bin width (number of bins).

    Bin width:
        h = 2 * IQR(values) / cubroot(values.size)

    Number of bins:
        nbins = (max(values) - min(values)) / h

    """
    # Sort data and weights.
    ind = np.argsort(ds).values
    d = ds[ind]
    w = weights[ind]

    # Interquartile range.
    pct = 1. * w.cumsum() / w.sum() * 100  # Convert to percentiles?
    iqr = np.interp([25, 75], pct, d)
    iqr = np.diff(iqr)

    # Freedman Diaconis Estimator (h=bin width).
    h = 2 * iqr / np.cbrt(ds.size)
    h = h[0]

    # Input data max/min.
    data_range = [ds.min().item(),  ds.max().item()]

    # Numpy conversion from bin width to number of bins.
    nbins = int(np.ceil(np.diff(data_range) / h)[0])
    return h, nbins, data_range



# # Man whitney test (histogram diff) returns zero while ttest_ind p = 0.158.
# # r sample sizes above ~20, approximation using the normal distribution is fairly good
# binned data -mwu=value=0.00035 ttest_ind=0.158
# Whitney U test can have inflated type I error rates even in large samples (especially if the variances of two populations are unequal and the sample sizes are different)
# Brunner-Munzel (n <50 samples?)and the Fligner–Policello test
# Brunner-Munzel and the Fligner–Policello test
# Kolmogorov–Smirnov test
from fncs import source_dataset
z = 3
ds = source_dataset(165, sum_interior=True)
dx = ds.sel(zone=z)['z_f']
dx = [dx.isel(exp=i).dropna('traj') for i in range(2)]

stats.mannwhitneyu(dx[0], dx[1], use_continuity=False)
stats.brunnermunzel(dx[0], dx[1])
stats.ttest_ind(dx[0], dx[1], equal_var=False)
