import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d


def extract_envelope(signal):
    """
    Extract oscillation envelope using Hilbert transform and Gaussian smoothing.
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    smoothed = gaussian_filter1d(envelope, sigma=5)  # smooth for clean peak detection
    return smoothed


def estimate_bp_points(envelope, pressure_range):
    """
    Estimate SBP, MAP, DBP using the standard oscillometric percent method.
    pressure_range: simulated pressure decreasing from ~180 to 40 mmHg.
    """
    max_env = np.max(envelope)
    max_idx = np.argmax(envelope)

    MAP = pressure_range[max_idx]

    # Find SBP: first point before MAP where envelope is ~55% of max
    sbp_idx = np.where(envelope[:max_idx] <= 0.55 * max_env)[0]
    SBP = pressure_range[sbp_idx[-1]] if len(sbp_idx) else pressure_range[0]

    # Find DBP: first point after MAP where envelope is ~85% of max
    dbp_idx = np.where(envelope[max_idx:] <= 0.85 * max_env)[0]
    DBP = pressure_range[max_idx + dbp_idx[0]] if len(dbp_idx) else pressure_range[-1]

    return {
        "SBP": round(SBP, 2),
        "MAP": round(MAP, 2),
        "DBP": round(DBP, 2)
    }
