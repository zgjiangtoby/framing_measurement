"""Two-stage estimators and paired Monte Carlo summaries."""

import numpy as np


def two_stage_estimate(z_pilot, z_audit, q_pilot, q_audit, population_size,
                       prediction_sum, prediction_variance):
    """Only budgeted labels enter this function; rows are Monte Carlo repetitions."""
    m, r = z_pilot.shape[1], z_audit.shape[1]
    assert m >= 2 and r > 0 and m + r <= population_size
    assert z_pilot.shape == q_pilot.shape and z_audit.shape == q_audit.shape
    covariance = ((z_pilot-z_pilot.mean(axis=1, keepdims=True)) *
                  (q_pilot-q_pilot.mean(axis=1, keepdims=True))).sum(axis=1)/(m-1)
    coefficient = np.clip(covariance/prediction_variance, 0, 1) if prediction_variance > 0 else np.zeros(len(z_pilot))
    complement_size = population_size-m
    complement_prediction_mean = (prediction_sum-q_pilot.sum(axis=1))/complement_size
    estimate = z_pilot.sum(axis=1)/population_size + complement_size/population_size * (
        coefficient*complement_prediction_mean +
        (z_audit-coefficient[:, None]*q_audit).mean(axis=1))
    return estimate, coefficient


def mcse(values):
    return float(np.std(values, ddof=1)/np.sqrt(len(values)))


def rmse_comparison(error_new, error_baseline):
    squared_new, squared_baseline = error_new**2, error_baseline**2
    a, b = np.sqrt(squared_new.mean()), np.sqrt(squared_baseline.mean())
    assert a > 0 and b > 0
    difference_influence = squared_new/(2*a)-squared_baseline/(2*b)
    ratio_influence = squared_new/(2*a*b)-a*squared_baseline/(2*b**3)
    return dict(rmse_difference=float(a-b), paired_mcse_rmse_difference=mcse(difference_influence),
                rmse_ratio=float(a/b), paired_mcse_rmse_ratio=mcse(ratio_influence),
                mean_error_difference=float((error_new-error_baseline).mean()),
                paired_mcse_mean_error_difference=mcse(error_new-error_baseline))
