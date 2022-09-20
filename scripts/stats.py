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
import statsmodels.stats.weightstats as wtd


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
    tdim = x.dims[0]
    def resample(ds):
        return ds.resample({tdim: "Y"}).mean(tdim, keep_attrs=True)
    x, y = x.dropna(tdim, 'all'), y.dropna(tdim, 'all')
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

# Kolmogorov–Smirnov test
if __name__ == "__main__":
    from fncs import source_dataset
    from tools import idx
    import matplotlib.pyplot as plt

    z = 3
    var = 'age'
    ds = source_dataset(165, sum_interior=True)
    dx = [ds.sel(zone=z).isel(exp=i).dropna('traj') for i in range(2)]
    dxx = [dx[i][var] for i in range(2)]

    bins = 'fd'
    color=['r', 'r']
    kwargs = dict(histtype='stepfilled', density=0, range=None, stacked=False, alpha=0.4)
    weights = [dx[i].u  for i in [0, 1]]  # sum(ds.rtime > np.datetime64('2020-01-06'))
    h0, _, r0 = weighted_bins_fd(dx[0][var], weights[0])
    h1, _, r1 = weighted_bins_fd(dx[1][var], weights[1])
    r = [min(np.floor([r0[0], r1[0]])), max(np.ceil([r0[1], r1[1]]))]
    kwargs['range'] = r
    bins = int(np.ceil(np.diff(r) / min([h0, h1])))

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    x1, bins1, _ = ax.hist(dx[0][var], bins, weights=weights[0], color='b', **kwargs)
    x2, bins2, _ = ax.hist(dx[1][var], bins, weights=weights[1], color='k', **kwargs)
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        # ax.axvline(np.quantile(dx[0][var], q), c='b')
        ax.axvline(bins1[sum(np.cumsum(x1) < (sum(x1)*q))], c='b')
        ax.axvline(bins2[sum(np.cumsum(x2) < (sum(x2)*q))], c='k')
    ax.set_xlim(0, 1000)

    stats.mannwhitneyu(*dxx, use_continuity=False)
    stats.mannwhitneyu(*dxx)
    stats.brunnermunzel(*dxx)
    stats.ttest_ind(*dxx, equal_var=False)
    stats.ks_2samp(*dxx)
    stats.epps_singleton_2samp(*dxx)
    wtd.ttest_ind(*dxx, weights=tuple(weights), usevar='unequal')

    stats.ttest_ind(x1, x2)
    stats.ttest_ind(x1, x2, equal_var=False)
    stats.mannwhitneyu(x1, x2)
    stats.wilcoxon(x1, x2)
    stats.wilcoxon(x1, x2, zero_method='pratt')
    stats.wilcoxon(x1, x2, zero_method='zsplit')


    x, y = dxx
    for func in [np.mean, np.median]:
        fx, fy = func(x).item(), func(y).item()
        print('{:} {:.2f}, {:.2f}, {:.2f} {:.1%}'.format(func.__name__[:4], fx, fy, fy-fx, (fy-fx)/fx))

    for q in [0.25, 0.5, 0.75]:
        fx, fy = [np.quantile(v, q).item() for v in [x,y]]
        print('qq{:} {:.2f}, {:.2f}, {:.2f} {:.1%}'.format(int(q*100), fx, fy, fy-fx, (fy-fx)/fx))

    for q in [0.25, 0.5, 0.75]:
        fx, fy = [b[sum(np.cumsum(v) < (sum(v)*q))] for v, b in zip([x1,x2], [bins1, bins2])]
        print('qb{:} {:.2f}, {:.2f}, {:.2f} {:.1%}'.format(int(q*100), fx, fy, fy-fx, (fy-fx)/fx))

    # fx, fy = [stats.mode(v, keepdims=False)[0]for v in [x,y]]
    fx, fy = [stats.mode(v)[0][0] for v in [x,y]]
    print('{} {:.2f}, {:.2f}, {:.2f} {:.1%}'.format('mode', fx, fy, fy-fx, (fy-fx)/fx))
    fx, fy = bins1[idx(x1, np.max(x1))], bins2[idx(x2, np.max(x2))]
    print('{} {:.2f}, {:.2f}, {:.2f} {:.1%}'.format('hmax', fx, fy, fy-fx, (fy-fx)/fx))
    fx, fy = [np.average(v[var], weights=v.u).item() for v in dx]
    print('{} {:.2f}, {:.2f}, {:.2f} {:.1%}'.format('wavg', fx, fy, fy-fx, (fy-fx)/fx))
