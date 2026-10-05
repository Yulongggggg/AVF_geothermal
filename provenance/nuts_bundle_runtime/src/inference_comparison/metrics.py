"""Marginal predictive scores. No KDE fitted to test truth or to each method."""
import numpy as np
from scipy.special import logsumexp, ndtr


def empirical_crps(draws, truth):
    x = np.sort(np.asarray(draws, dtype=np.float64), axis=0)
    n = len(x)
    coefficient = (2*np.arange(1, n+1)-n-1).reshape((n,)+(1,)*(x.ndim-1))
    return np.mean(np.abs(x-truth), axis=0)-np.sum(coefficient*x, axis=0)/n**2


def score(draws, truth, epistemic_sd=None):
    """Return pointwise scores of raw samples or a fixed Gaussian mixture.

    Mixture NLL and quantiles are evaluated directly. CRPS uses the standard
    Gaussian-mixture formula, in chunks to bound memory. Instrument noise is
    excluded when the target is a noiseless future FE response.
    """
    x = np.asarray(draws, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if epistemic_sd is None:
        lo, med, hi = np.quantile(x, [.05, .5, .95], axis=0)
        crps = empirical_crps(x, truth)
        nll = None  # An empirical delta distribution has no continuous density.
    else:
        sd = np.broadcast_to(epistemic_sd, truth.shape).astype(np.float64)
        if np.any(sd <= 0): raise ValueError('Positive validation-fitted scale required')
        normalized = (truth-x)/sd
        nll = -(logsumexp(-.5*normalized**2-np.log(sd)-.5*np.log(2*np.pi), axis=0)-np.log(len(x)))

        def quantile(q):
            low, high = x.min(0)-10*sd, x.max(0)+10*sd
            for _ in range(40):
                mid = (low+high)/2
                below = ndtr((mid-x)/sd).mean(0) < q
                low, high = np.where(below, mid, low), np.where(below, high, mid)
            return (low+high)/2

        lo, med, hi = (quantile(q) for q in (.05, .5, .95))
        def absolute_normal(delta, scale):
            z = delta/scale
            return 2*scale*np.exp(-.5*z*z)/np.sqrt(2*np.pi)+delta*(2*ndtr(z)-1)
        first = absolute_normal(x-truth, sd).mean(0)
        second = np.zeros_like(truth)
        for i in range(len(x)):
            second += absolute_normal(x[i]-x, np.sqrt(2)*sd).mean(0)/len(x)
        crps = first-.5*second
    result = dict(crps=crps, p50_absolute_error=np.abs(med-truth),
                  coverage90=((truth >= lo)&(truth <= hi)).astype(float),
                  width90=hi-lo, p05=lo, p50=med, p95=hi)
    if nll is not None: result['nll'] = nll
    return result
